import torch
import os
import sys

import torch.optim as optim
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, Sampler

from models.restnet_custom import resnet50_baseline
from models.model_Full import Partial_Net
from models.attention_model import Attention_MB
from learning.logger import EarlyStopping, Accuracy_Logger
from .train import calculate_error
from arguments.partial_training_selections import RandomSelection, SolidSelection, HandPickedSelection, AttentionSelection

from tqdm import tqdm
import copy
from utils.helper import compare_models

class Patient_Dataset(Dataset):
    """
    A class which inherits Dataset to use DataLoader

    Attributes
    ----------
    X : WholeSlideImage Object
        The Whole Slide Image Object for this patient
    i : int
        The Tile Property index for this WSI to use

    Methods
    -------
    __len__()
        Number of items to process - number of Tiles
    __getitem__(idx)
        Return current image
    """
    def __init__(self, X, i):
        """
        X : WholeSlideImage Object
            The Whole Slide Image Object for this patient
        i : int
            The Tile Property index for this WSI to use
        marked_elements: boolean
            Boolean to decide if only marked Tiles shall be returned
        """
        # WSI
        self.X = X
        # Tile property index
        self.i = i
        #marking search

    def __len__(self):
        """ Return number of items to process

        Returns
        -------
        int 
            Number of items
        """
        # Number of available tiles
        return len(self.X.get_tiles_list()[self.i])

    def __getitem__(self, idx):
        """ Return Tile Image at idx

        Parameters
        ----------
        idx : int
            Index of Tile
        
        Returns
        -------
        TorchTensor
            The Tiles image casted to TorchTensor
        """
        # Get augmented image at index
        a, mark = self.X.get_tiles_list()[self.i][idx].get_image(self.X.get_image(), True)                        
        # Transpose for channel first
        img = np.transpose(a, (2, 0, 1))
        # Cast to float
        img = img.astype(np.float32)
        # Check if has three dimensions
        if len(np.shape(img)) == 3:
            img = np.expand_dims(img, axis=0)
        # To torch tensor
        all_p = torch.from_numpy(img)
        
        return {"all_p":all_p, "mark":mark, "tile_pos":idx}

def collate_features(batch):
    """ Concatenate batch as TorchTensor

    Parameters
    ----------
    batch : list [TorchTensor]
        Batch of images 

    Returns
    -------
    TorchTensor
        Concatenated batch
    """
    # concatenate batch
    img = torch.cat([item for item in batch], dim=0)

    return img

def create_cleaned_dataset(dataset):
    """creates a list of images that are ordered according to mark or not marked

    Parameters
    ----------
    dataset : list of dictionaries
        one entry contains an image of a tile and if this tile was marked as well as its position

    Returns
    -------
    list of images
        list of images ordered according to mark or not marked
    """    
    clean_dict_dataset = []
    j = 0
    #first loop sorts the marked datasets first
    for ele in tqdm(iter(dataset)):
        if ele['mark']:
            clean_dict_dataset = [{"img":ele['all_p'],"pos":ele['tile_pos']}] + clean_dict_dataset
            j += 1
        else:
            #print('no mark')
            clean_dict_dataset.append({"img":ele['all_p'],"pos":ele['tile_pos']})
    
    #splits the sorted dataset into tile pos and img
    pos_list = []
    clean_dataset = []
    for data in clean_dict_dataset:
        clean_dataset.append(data['img'])
        pos_list.append(data['pos'])

    list_of_marked = []
    for i in range(j):
        list_of_marked.append(clean_dataset[i])

    return clean_dataset, list_of_marked, pos_list

class CleanDataset(Dataset):
    """Dataset class for the cleaned dataset

    Parameters
    ----------
    Dataset : list of ordered images of tiles

    Attributes
    ----------
    clean_dataset : list of images

    Functions
    ---------
    __len__()
        returns the number of images in the dataset
    __getitem__(idx:int)
        returns the image at the specified index
    """    
    def __init__(self, clean_dataset) -> None:
        self.clean_dataset = clean_dataset

    def __len__(self):
        """returns the number of images in the dataset

        Returns
        -------
        int
            number of images
        """        
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        """returns the image at the specified index

        Parameters
        ----------
        idx : int
            index of the image

        Returns
        -------
        image
            returns image at the specified index
        """        
        return self.clean_dataset[idx]

def set_label(label, device):
    """Set the label of the dataset

    Parameters
    ----------
    label : int
        label of the dataset
    device : torch.device
        cuda or cpu
    Returns
    -------
    torch.Tensor
    """
    if label == 0:
        label = torch.zeros(1)
        label = label.to(device)
    else:
        label = torch.ones(1)
        label = label.to(device)

    return label
def select_selection_mode(selection_mode, feature_setting, wsi, epoch=None):
    """returns the selection mode

    Parameters
    ----------
    selection_mode : string
        contains the selection mode

    Returns
    -------
    Selection
        mode that selects the  tiles to be marked
    """    
    if selection_mode == "random":
        return RandomSelection(setting=feature_setting, wsi=wsi)
    elif selection_mode == "solid":
        return SolidSelection(setting=feature_setting, wsi=wsi)
    elif selection_mode == "handpicked":
        return HandPickedSelection(setting=feature_setting, wsi=wsi)
    elif selection_mode == "attention":
        return AttentionSelection(setting=feature_setting, wsi=wsi, epoch=epoch)
    else:
        return RandomSelection()

def create_dataset(wsi, i):
    """Create the dataset for the patient

    Parameters
    ----------
    wsi : WholeSlideImage Object
        The Whole Slide Image Object for this patient
    i : int
        The Tile Property index for this WSI to use

    Returns
    -------
    data : DataSet
        Dataset for the WSI
    marked_tensors : list of TorchTensor
        contains the tensors of the marked tiles
    pos_list : list of int
        contains the positions of the tiles marked
    """    
    data = Patient_Dataset(wsi, i)
    data, marked_tensors, pos_list = create_cleaned_dataset(data)
    data = CleanDataset(data)

    return data, marked_tensors, pos_list

def partial_training(patients, patients_val, setting, fold, selection_mode):
    """ Create features for patients and save them
    Parameters
    ----------
    patients : list [[Patient]]
        list of list per class of Patient to train
    setting : Setting
        Setting as defined by class
    patients_val : list [[Patient]]
        list of patients to use for validation
    fold : int
        Fold number
    """

    # Get feature setting
    feature_setting = setting.get_feature_setting()
    # Number of classes
    n_classes = setting.get_class_setting().get_n_classes()
    # Get Encoder - default pretrained ResNet50
    param_Encoder = resnet50_baseline(pretrained=True)
    param_Encoder.train()
    # Get Decoder
    param_Decoder = Attention_MB(setting)
    param_Decoder.train()
    # get Partial Model
    model = Partial_Net(encoder=param_Encoder, decoder=param_Decoder)
    # To GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    #define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Early stopping checker
    early_stopping = EarlyStopping(patience=setting.get_network_setting().get_patience(), stop_epoch=setting.get_network_setting().get_stop_epoch(), verbose=setting.get_network_setting().get_verbose())
    # Folder to save model parameters to
    model_folder = setting.get_network_setting().get_model_folder()
    # File to save model parameters to
    model_file = 's_{}_checkpoint.pt'.format(fold)
    # Number of patients to process
    n_patients = sum([len(p) for p in patients])
    for epoch in range(setting.get_network_setting().get_epochs()):
        model.getEncoder().train()
        model.getDecoder().train()
        model.train()
        # Iterate patient classes
        for patient in patients:
            # Iterate patients in class
            for p in patient:
                # Iterate image properties
                label = p.get_diagnosis().get_label()
                label = set_label(label, device)
                #print(p.get_wsis())
                for wp in p.get_wsis():
                    # Iterate WSIs with image property
                    for wsi in wp:
                        # Iterate Tile properties
                        for i in range(len(wsi.get_tile_properties())):
                            # If no tiles then nothing to be done
                            if len(wsi.get_tiles_list()) != 0:
                                #trains the encoder for one wsi property
                                features_wsi, encoder_pre_train, unmarked_outputs, marked_output = train_stage_one(wsi, i, feature_setting, selection_mode, epoch, device, model)
                                features_wsi.retain_grad()
                                #sets the features of a wsi property
                            set_features(features_wsi, wsi, i)
                        #train the decoder
                        train_decoder(model, loss_fn, features_wsi,  optimizer, marked_output, unmarked_outputs, encoder_pre_train, setting, device, label, p, wsi)
        #validate the model
        stop = validate_partial_net_epoch(epoch, model, n_classes, loss_fn, early_stopping, patients_val, feature_setting, ckpt_name=model_folder + model_file)
        if stop and setting.get_network_setting().get_early_stopping():
            break

def set_features(features_wsi, wsi, i):
    """sets the features of a patient for a wsi

    Parameters
    ----------
    features_wsi : feature matrix
        contains the concatenated features of a wsi
    wsi : WSI object
        contains the WSI object
    i : int
        contains the Tile Property index for this WSI to use
    """    
    #create features
    features_numpy = features_wsi.cpu().detach().numpy()
    # Create directory to save features and keys
    feature_directory = {}
    # Fill directory
    for j, tile in enumerate(wsi.get_tiles_list()[i]):
    # Key Tile position, value feature of Tile
        feature_directory[tile.get_position()] = features_numpy[j]
    # Save features
    wsi.set_features(feature_directory, i)

def train_stage_one(wsi, i, feature_setting, selection_mode, epoch, device, model):
    """runs the encoder and concatenates the computed features per tile

    Parameters
    ----------
    wsi : WSI object
        contains the WSI object
    i : int
        contains the Tile Property index for this WSI to use
    feature_setting : Feature_Setting
        contains the feature setting
    selection_mode : String
        contains the mode that selects the tiles to be marked
    epoch : int
        contains the current epoch
    device : torch.device
        contains the current device
    model : Full_Net
        contains the current model

    Returns
    -------
    features_wsi : torch.Tensor
        contains the concatenated features of a wsi
    encoder_pre_train : torch.state_dict
        contains the  encoder before training
    unmarked_outputs : [torch.Tensor]
        contains a list of torch.Tensor that contain the unmarked outputs
    marked_output : [torch.Tensor]
        contains a list of torch.Tensor that contain the marked outputs
    """    
    # Open WSI
    wsi.load_wsi()
    #select the selection mode according to the settings
    selection = select_selection_mode(selection_mode, feature_setting, wsi, epoch)
    #mark tiles as defined by the selection mode
    selection.tile_marking()
    # Create Dataset
    data, marked_tensors, pos_list = create_dataset(wsi, i)
    # Create Dataloader
    loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
    marked_loader = DataLoader(dataset=marked_tensors, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(marked_tensors))
    marked_tensors_list = []
    for ele in marked_loader:
        marked_tensors_list.append(ele)
    i = 0
    #Encoder before training
    encoder_pre_train = copy.deepcopy(model.getEncoder()).state_dict()
    #used to determine whether a gradient was computed for a marked output and None for an unmarked
    #iterates per batch in the Dataloader
    unmarked_outputs, marked_output, features_wsi = train_encoder(loader, model, device, marked_tensors_list, selection.get_marked_batches())
    #reset tiles
    selection.tile_resetting()
    #close WSI
    wsi.close_wsi()
    #retain the gradients for proofs
    return features_wsi, encoder_pre_train, unmarked_outputs, marked_output

def set_error(loss_parent, error_parent, loss_child, error_child, counter, ret_counter=True):
    """set an error for the validation of an epoch

    Parameters
    ----------
    loss_parent : float
        loss parent
    error_parent : float
        parent error
    loss_child : float
        loss added to the parent
    error_child : float
        error to be added to the parent
    counter : int
        counter of errors
    ret_counter : bool, optional
        set if the counter is to be returned, by default True

    Returns
    -------
    loss_parent : float
        new parent loss
    error_parent : float
        new error of parent
    counter : int
        counter of errors
    """    
    if ret_counter:
        loss_parent += loss_child / counter
        error_parent += error_child / counter
        counter += 1
        return loss_parent, error_parent, counter
    else:
        loss_parent += loss_child / counter
        error_parent += error_child / counter
        return loss_parent, error_parent
    

def validate_partial_net_epoch(epoch, model, n_classes, loss_fn, early_stopping, patients_val, feature_setting, ckpt_name,):
    """validates the model for one epoch

    Parameters
    ----------
    epoch : int
        contains the current epoch
    model : Full_Net
        contains the current model
    n_classes : int
        contains the number of classes
    loss_fn : torch.nn.Module
        contains the loss function
    early_stopping : bool
        contains the early stopping setting
    patients_val : [Patient]
        contains the patients to be validated
    feature_setting : Feature_Setting
        contains the feature setting
    ckpt_name : String
        contains the name of the checkpoint file

    Returns
    -------
    stop : bool
        set to True if the validation was stopped, by default False
    """    
    #get the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set the model to evaluation mode
    model.getDecoder().eval()
    model.getEncoder().eval()
    model.eval()
    #define the validation loss and error
    val_loss = 0.
    val_error = 0.
    # Just forward pass needed
    with torch.no_grad():
        val_error_pat = 0.
        val_loss_pat = 0.
        patient_counter = len(patients_val)
        #iterate patients
        for p in patients_val:
            wsi_prop_counter = len(p[0].get_wsis())
            val_error_wsi_prop = 0.
            val_loss_wsi_prop = 0.
            #get label
            label = p[0].get_diagnosis().get_label()
            label = set_label(label, device)
            # Iterate image properties
            for wsi_prop in p[0].get_wsis():
                wsi_counter = len(wsi_prop)
                val_error_wsi = 0.
                val_loss_wsi = 0.
                #iterate WSIs in image property
                for wsi in wsi_prop:
                    tileprop_counter = 0
                    val_loss_tile_prop = 0.
                    val_error_tile_prop = 0.
                    #iterate tileproperties in WSI
                    for i in range(len(wsi.get_tile_properties())):
                        val_error_tile = 0. 
                        val_loss_tile = 0.
                        if len(wsi.get_tiles_list()) != 0:
                            wsi.load_wsi()
                            #create data
                            data, marked_tensors, pos_list = create_dataset(wsi, i)
                            loader = DataLoader(dataset= data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                            #compute features    
                            features = test_encoder(loader, model, device)
                            #compute error and loss of decoder
                            error, loss = test_decoder(model, features, label, device, False)
                            # pass error and loss to parent errors
                            val_error_tile += error
                            val_loss_tile += loss
                            wsi.close_wsi()

                            val_loss_tile_prop, val_error_tile_prop = set_error(val_loss_tile_prop, val_error_tile_prop, val_loss_tile, val_error_tile, 1, False)
                            tileprop_counter += 1
                    
                    #compute error per wsi
                    val_loss_wsi, val_error_wsi = set_error(val_loss_wsi, val_error_wsi, val_loss_tile_prop, val_error_tile_prop, tileprop_counter, False)
                    wsi_counter += 1
                #compute error per wsi_prop
                val_loss_wsi_prop, val_error_wsi_prop = set_error(val_loss_wsi_prop, val_error_wsi_prop, val_loss_wsi, val_error_wsi, wsi_counter, False)
                wsi_prop_counter += 1
            #compute error per patient
            val_loss_pat, val_error_pat = set_error(val_loss_pat, val_error_pat, val_loss_wsi_prop, val_error_wsi_prop, wsi_prop_counter, False)
        #compute error overall
        val_loss, val_error = set_error(val_loss, val_error, val_loss_pat, val_error_pat, patient_counter, False)
    # Compute Early stopping
    early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)

    if early_stopping.early_stop:
        return True

    return False

def train_encoder(dataloader, model, device, marked_tens, number_markings, randomList = None, selection = None):
    """trains the encoder for one epoch

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        contains the dataloader
    model : Full_Net
        contains the current model
    device : torch.device
        contains the device
    marked_tens : [torch.Tensor]
        contains the marked tensors
    number_markings : int
        contains the number of markings
    randomList : [torch.Tensor]
        contains the random tensors
    selection : Selection
        contains the selection

    Returns
    -------
    unmarked_output : torch.Tensor
        contains the unmarked output
    marked_output : torch.Tensor
        contains the marked output
    features_wsi : torch.Tensor
        contains the features of the WSIs
    """
    i = 0
    batchlist = []
    features_wsi = None
    for batch in tqdm(dataloader):
        if i<=number_markings:
            batchlist.append(batch)
            batch = batch.to(device, non_blocking=True)
            # Compute features
            features = model.getEncoder()(batch)
            print('marked')
        if i == 0:
            features_wsi = features
            marked_output = features
            marked_output.retain_grad()
        else:
            with torch.no_grad():
                batchlist.append(batch)
                batch = batch.to(device, non_blocking=True)
                
                # Compute features
                features = model.getEncoder()(batch)
                if i == 2:
                    unmarked_output = features
        if i == 4:
            #used to confirm that the first two batches are identical with the marked batches 
            print(torch.equal(batchlist[0], marked_tens[0]))
            print(not torch.equal(batchlist[0].to('cpu'),marked_tens[2].to('cpu')))
            print(not torch.equal(batchlist[1].to('cpu'),marked_tens[2].to('cpu')))
            print(torch.equal(batchlist[1], marked_tens[1]))
        # Concatenate features
        features_wsi = torch.cat((features_wsi, features), 0)
        i += 1

    return unmarked_output, marked_output, features_wsi

def check_feat_dir_contains_files(wsi):
    
    for ele in wsi.feature_file_names:
        if not os.path.isfile(ele):
            print('pen')
            return False

    return True

def train_decoder(model, loss_fn, features, optimizer, marked_output, unmarked_output, encoder_pre_train, setting, device, label, patient, wsi, draw_map=True):
    """trains the decoder for one epoch

    Parameters
    ----------
    model : Full_Net
        contains the current model
    loss_fn : torch.nn.modules.loss
        contains the loss function
    features : torch.Tensor
        contains the features of the WSIs
    optimizer : torch.optim
        contains the optimizer
    marked_output : torch.Tensor
        contains the marked output
    unmarked_output : torch.Tensor
        contains the unmarked output
    encoder_pre_train : torch.Tensor
        contains the features of the WSIs
    setting : Setting
        contains the setting
    device : torch.device
        contains the device
    label : torch.Tensor
        contains the label of the patient
    patient : Patient
        contains the patient
    wsi : WSI
        contains the WSI
    draw_map : bool
        contains the draw attention map

    Returns
    -------
    """
    #preperations for the decoder run
    n_classes = setting.get_class_setting().get_n_classes()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    features.to(device)
    print('begun decoder run')
    logits, Y_prob, Y_hat, A = model.getDecoder()(features, label)
    
    logits.retain_grad()
    
    #conversions of Tensor types for loss calculation
    label = label.type(torch.LongTensor)
    label = label.to(device)
    logits = logits.to(device)
    
    acc_logger.log(Y_hat, label)
    loss = loss_fn(logits, label)
    loss_value = loss.item()
    #print the current loss
    print(loss_value)
    error = calculate_error(Y_hat, label)
    print(error)

    total_loss= loss
    
    total_loss.backward()
    print(logits.grad)
    print(marked_output.grad)
    optimizer.step()
    print('-------------------------')
    print('marked gradient')
    print(marked_output.grad)
    print('-------------------------')
    print('unmarked gradient')
    print(unmarked_output.grad)
    
    optimizer.zero_grad()
    # Get attention map for predicted class
    if check_feat_dir_contains_files(wsi):
        features, keys = patient.get_features(wsi)
        print('featget')
        A = A[Y_hat]
        A = A.view(-1, 1).cpu().detach().numpy()
        # Get tile keys in attention map
        if draw_map:
            # Save attention map
            patient.set_map(A, keys, wsi)
    #encoder post train
    encoder_post_train = model.getEncoder().state_dict()

    #compare models
    compare_models(encoder_pre_train, encoder_post_train)

def test_encoder(dataloader, model, device):
    """tests the encoder with the given dataloader

    Parameters
    ----------
    model : Full_Net
        contains the current model
    dataloader : Dataloader
        contains the current dataloader

    Returns
    -------
    features_wsi : 
        _description_
    """
    i = 0

    features_wsi = None
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            features = model.getEncoder()(batch)
            if i == 0:
                features_wsi = features
            else:
                features_wsi = torch.cat((features_wsi, features), 0)
    
    return features_wsi

def test_decoder(model, features, label, device, draw_map=True):
    """tests the decoder with the given features and label

    Parameters
    ----------
    model : Full_Net
        contains the current model
    features : torch.Tensor
        contains the current features of a wsi
    label : _type_
        _description_
    device : _type_
        _description_
    draw_map : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """    
    loss_fn = torch.nn.CrossEntropyLoss()
    label = label.type(torch.LongTensor)
    features, label = features.to(device), label.to(device)
    with torch.no_grad():
        logits, Y_prob, Y_hat, A = model.getDecoder()(features, label)
    error = calculate_error(Y_hat, label)
    label, logits = label.to(device), logits.to(device)

    loss = loss_fn(logits, label)
    return error, loss