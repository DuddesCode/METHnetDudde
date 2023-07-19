import torch
import os
import sys

import torch.optim as optim
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn

from models.rest_net_model_paral import resnet50_baseline_paral
from models.restnet_custom import resnet50_baseline
from models.model_Full import Partial_Net
from models.attention_model import Attention_MB
from learning.logger import EarlyStopping, Accuracy_Logger
from .train import calculate_error
from arguments.partial_training_selections import RandomSelection, SolidSelection, HandPickedSelection, AttentionSelection

from timeit import default_timer as timer
from tqdm import tqdm
import copy
from utils.helper import compare_models
import torch.backends.cudnn


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ":16:8")
print(os.environ['CUBLAS_WORKSPACE_CONFIG'])

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
        timer_start = timer()
        a, mark = self.X.get_tiles_list()[self.i][idx].get_image(self.X.get_image(), True)
        timer_stop = timer()
        time = timer_stop-timer_start                          
        # Transpose for channel first
        img = np.transpose(a, (2, 0, 1))
        # Cast to float
        img = img.astype(np.float32)
        # Check if has three dimensions
        if len(np.shape(img)) == 3:
            img = np.expand_dims(img, axis=0)
        # To torch tensor
        all_p = torch.from_numpy(img)
        
        return {"all_p":all_p}, time

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

def create_cleaned_dataset(dataset, index):
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
    times = 0 
    for ele, time in dataset:
        times += time
        clean_dict_dataset.append(ele['all_p'])
    if index is not None:
        clean_dict_dataset = np.asarray(clean_dict_dataset)
        temp_marked = clean_dict_dataset[index]
        clean_dict_dataset = np.delete(clean_dict_dataset, index)

        clean_dict_dataset = np.concatenate((temp_marked, clean_dict_dataset))
        clean_dict_dataset = clean_dict_dataset.tolist()

    clean_dataset = clean_dict_dataset
    #first loop sorts the marked datasets first
    """for ele in tqdm(iter(dataset)):
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
        list_of_marked.append(clean_dataset[i])"""

    return clean_dataset, times


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

def select_selection_mode(selection_mode, feature_setting, wsi, epoch=None, json_path=None):
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
        return RandomSelection(setting=feature_setting, wsi=wsi, json_path=json_path)
    elif selection_mode == "solid":
        return SolidSelection(setting=feature_setting, wsi=wsi, epoch=epoch, json_path=json_path)
    elif selection_mode == "hand_picked":
        return HandPickedSelection(setting=feature_setting, wsi=wsi, json_path=json_path)
    elif selection_mode == "attention":
        return AttentionSelection(setting=feature_setting, wsi=wsi, epoch=epoch, json_path=json_path)
    else:
        return RandomSelection()

def create_dataset(wsi, i, index=None):
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
    
    data, time_loading_img = create_cleaned_dataset(data, index)
    data = CleanDataset(data)

    return data, time_loading_img
    

def partial_training_parallel(patients, patients_val, setting, fold, selection_mode, draw_map=False, json_path = None):
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
    param_Encoder.eval()
    # Get Decoder
    param_Decoder = Attention_MB(setting)

    decoder_test = copy.deepcopy(param_Decoder)
    param_Decoder.eval()

    encoder_test = resnet50_baseline(pretrained=True)
    encoder_test.eval()
    decoder_test.eval()
    full_model_test = Partial_Net(encoder=encoder_test, decoder=decoder_test)
    full_model_test.eval()
    # get Partial Model
    model = Partial_Net(encoder=param_Encoder, decoder=param_Decoder)
    # To GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    full_model_test.to(device)
    # Loss function
    loss_fn = torch.nn.L1Loss()
    #define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, nesterov=False)
    optimizer_test = optim.SGD(full_model_test.parameters(), lr=0.001, nesterov=False)
    # Early stopping checker
    early_stopping = EarlyStopping(patience=setting.get_network_setting().get_patience(), stop_epoch=setting.get_network_setting().get_stop_epoch(), verbose=setting.get_network_setting().get_verbose())
    # Folder to save model parameters to
    model_folder = setting.get_network_setting().get_model_folder()
    # File to save model parameters to
    model_file = 's_{}_checkpoint.pt'.format(fold)
    #array to store the losses for wll wsi. One iteration means one wsi
    train_losses = []
    val_losses = []
    wsi_list = []
    epoch_timelist = []
    time_spent_imgs = []
    time_spent_dataset = []

    enc_pre_train_normal = copy.deepcopy(model.getEncoder()).state_dict()
    enc_pre_train_test = copy.deepcopy(full_model_test.getEncoder()).state_dict()
    compare_models(model.cpu().state_dict(), full_model_test.cpu().state_dict())
    model.to(device)
    full_model_test.to(device)
    for epoch in range(setting.get_network_setting().get_epochs()):
        running_loss = 0.0
        loss_count = 0
        epoch_timer_start = timer()
        model.getEncoder().eval()
        model.getDecoder().eval()
        model.eval()
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
                        wsi_list.append(wsi)
                        # Iterate Tile properties
                        for i in range(len(wsi.get_tile_properties())):
                            # If no tiles then nothing to be done
                            if len(wsi.get_tiles_list()) != 0:
                                #trains the encoder for one wsi property
                                features_wsi, num_batches_marked, time_spent_img , time_spent_ds= train_stage_one(wsi, i, feature_setting, selection_mode, epoch, device, model, json_path=json_path)
                                features_wsi_test, num_batches_marked, time_spent_img , time_spent_ds= train_stage_one_parallel(wsi, i, feature_setting, selection_mode, epoch, device, full_model_test, json_path=json_path)
                                if num_batches_marked != 0:
                                    features_wsi.retain_grad()
                                    features_wsi_test.retain_grad()
                                time_spent_imgs.append(time_spent_img)
                                time_spent_dataset.append(time_spent_ds)
                                #sets the features of a wsi property
                            set_features(features_wsi, wsi, i)
                        #train the decoder
                        loss_wsi, error = train_decoder(model, loss_fn, features_wsi,  optimizer, setting, device, label, p, wsi, draw_map=draw_map)
                        
                        loss_wsi_test, error = train_decoder(full_model_test, loss_fn, features_wsi_test,  optimizer_test, setting, device, label, p, wsi, draw_map=draw_map, para=True)
                        enc_post_train = model.getEncoder().state_dict()
                        #full_model_test.getEncoder().set_ResNet_Baseline_device('cuda:0')
                        enc_test_post_train = full_model_test.getEncoder().state_dict()
                        print('same base encoders models test')
                        compare_models(enc_pre_train_normal, enc_pre_train_test)
                        print('same trained encoder models test')
                        compare_models(enc_post_train, enc_test_post_train)
                        print('same models post train test')
                        compare_models(model.cpu().state_dict(), full_model_test.cpu().state_dict())
                        running_loss += loss_wsi
                        loss_count += 1

        running_loss = running_loss / loss_count
        train_losses.append(running_loss)
        #validate the model
        stop, val_loss = validate_partial_net_epoch(epoch, model, early_stopping, patients_val, feature_setting, ckpt_name=model_folder + model_file, json_path=json_path)
        val_losses.append(val_loss.cpu())
        if stop and setting.get_network_setting().get_early_stopping():
            break
        epoch_timer_stop = timer()
        epoch_timelist.append(epoch_timer_stop - epoch_timer_start)
        print('epochtime')
        print(epoch_timelist)

    train_losses = np.array(train_losses, dtype=np.float32)
    val_losses = np.array(val_losses, dtype=np.float32)
    print(train_losses)
    epoch_timelist = np.array(epoch_timelist, dtype=np.float32)
    time_spent_img = np.array(time_spent_imgs, dtype = np.float32)
    time_spent_dataset= np.array(time_spent_dataset, dtype = np.float32)
    if json_path is not None:
        save_it_losses(train_losses, json_path, 'train_losses')
        save_it_losses(epoch_timelist, json_path, 'epoch_time')
        save_it_losses(val_losses, json_path, 'val_losses')
        save_it_losses(time_spent_imgs, json_path, 'time_spent_img')
        save_it_losses(time_spent_dataset, json_path, 'time_spent_dataset')

    #if selection_mode == 'solid':
        #print('doenerbanana')
        #wsi_list = list(set(wsi_list))
        #solid_tile_reset(wsi_list)
    return num_batches_marked

def save_it_losses(it_losses, json_path, file_name):
    import json
    with open(json_path+'setup.json', 'r') as outfile:
        setup_file = json.load(outfile)
    fn = setup_file['test_score_folder']+file_name+'.npy'
    np.save(fn, it_losses)

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

def train_stage_one(wsi, i, feature_setting, selection_mode, epoch, device, model, json_path=None):
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
    selection = select_selection_mode(selection_mode, feature_setting, wsi, epoch, json_path=json_path)
    #mark tiles as defined by the selection mode
    index_array = selection.tile_marking(i)
    # Create Dataset
    timer_spent_ds_start = timer()
    data, time_spent_img = create_dataset(wsi, i, index_array)
    timer_spent_ds_stop = timer()
    time_spent_ds = timer_spent_ds_stop-timer_spent_ds_start
    # Create Dataloader
    loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
    #used to determine whether a gradient was computed for a marked output and None for an unmarked
    #iterates per batch in the Dataloader
    features_wsi = train_encoder(loader, model, device, selection.get_marked_batches())
    #reset tiles
    #close WSI
    wsi.close_wsi()
    #retain the gradients for proofs
    return features_wsi, selection.get_marked_batches(), time_spent_img, time_spent_ds

def train_stage_one_parallel(wsi, i, feature_setting, selection_mode, epoch, device, model, json_path=None):
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
    selection = select_selection_mode(selection_mode, feature_setting, wsi, epoch, json_path=json_path)
    #mark tiles as defined by the selection mode
    index_array = selection.tile_marking(i)
    # Create Dataset
    timer_spent_ds_start = timer()
    data, time_spent_img = create_dataset(wsi, i, index_array)
    timer_spent_ds_stop = timer()
    time_spent_ds = timer_spent_ds_stop-timer_spent_ds_start
    # Create Dataloader
    loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
    #used to determine whether a gradient was computed for a marked output and None for an unmarked
    #iterates per batch in the Dataloader
    features_wsi = train_encoder_parallel(loader, model, device, selection.get_marked_batches())
    #reset tiles
    #close WSI
    wsi.close_wsi()
    #retain the gradients for proofs
    return features_wsi, selection.get_marked_batches(), time_spent_img, time_spent_ds


def validate_partial_net_epoch(epoch, model, early_stopping, patients_val, feature_setting, ckpt_name, json_path=None):
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
    val_losses = 0
    val_errors = 0
    # Just forward pass needed
    with torch.no_grad():
        #iterate patients
        for p in patients_val:
            #get label
            label = p[0].get_diagnosis().get_label()
            label = set_label(label, device)
            # Iterate image properties
            for wsi_prop in p[0].get_wsis():
                #iterate WSIs in image property
                for wsi in wsi_prop:
                    #iterate tileproperties in WSI
                    for i in range(len(wsi.get_tile_properties())):
                        if len(wsi.get_tiles_list()) != 0:
                            wsi.load_wsi()
                            #create data
                            data, time_placeholder = create_dataset(wsi, i)
                            loader = DataLoader(dataset= data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                            #compute features    
                            features = test_encoder(loader, model, device)
                            #compute error and loss of decoder
                            error, loss, Y_prob = test_decoder(model, features, label, device, False)
                            # pass error and loss to parent errors
                            print('error')
                            print(error)
                            print('loss')
                            print(loss)
                            
                            val_error += error
                            val_errors += 1
                            val_loss += loss
                            val_losses += 1
                            wsi.close_wsi()

    val_loss = val_loss / val_losses
    # Compute Early stopping
    early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)

    if early_stopping.early_stop:
        return True, val_loss

    return False, val_loss

def train_encoder_parallel(dataloader, model, device, number_markings):
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
    i = 1
    batchlist = []
    features_wsi = None
    #model = model.to(device)
    parallelNet = nn.DataParallel(model.getEncoder(), device_ids=[0,1,2,3])
    for batch in tqdm(dataloader):
        if i<=number_markings:
            batchlist.append(batch)
            batch = batch.to(device, non_blocking=True)
            # Compute features
            features = parallelNet(batch)
            print(features.requires_grad)
            #features = features.to('cuda:1', non_blocking=True)
            print('marked')
        else:
            with torch.no_grad():
                batchlist.append(batch)
                batch = batch.to(device, non_blocking=True)
                
                # Compute features
                features = parallelNet(batch)
                #features = features.to('cuda:1', non_blocking=True)
        if i == 1:
            features_wsi = features
            features_wsi = features_wsi.to('cuda:1', non_blocking=True)

        #if i == 4:
            #used to confirm that the first two batches are identical with the marked batches 
            #print(torch.equal(batchlist[0], marked_tens[0]))
            #print(not torch.equal(batchlist[0].to('cpu'),marked_tens[2].to('cpu')))
            #print(not torch.equal(batchlist[1].to('cpu'),marked_tens[2].to('cpu')))
            #print(torch.equal(batchlist[1], marked_tens[1]))
        # Concatenate features
        features = features.to('cuda:1', non_blocking=True)
        features_wsi = torch.cat((features_wsi, features), 0)
        i += 1

    return features_wsi.to('cuda:0', non_blocking=True)

def train_encoder(dataloader, model, device, number_markings):
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
    i = 1
    batchlist = []
    features_wsi = None
    for batch in tqdm(dataloader):
        if i<=number_markings:
            batchlist.append(batch)
            batch = batch.to(device, non_blocking=True)
            # Compute features
            features = model.getEncoder()(batch)
            print(features.requires_grad)
            features = features.to('cuda:1', non_blocking=True)
            print('marked')
        else:
            with torch.no_grad():
                batchlist.append(batch)
                batch = batch.to(device, non_blocking=True)
                
                # Compute features
                features = model.getEncoder()(batch)
                features = features.to('cuda:1', non_blocking=True)
        if i == 1:
            features_wsi = features
            features_wsi = features_wsi.to('cuda:1', non_blocking=True)

        #if i == 4:
            #used to confirm that the first two batches are identical with the marked batches 
            #print(torch.equal(batchlist[0], marked_tens[0]))
            #print(not torch.equal(batchlist[0].to('cpu'),marked_tens[2].to('cpu')))
            #print(not torch.equal(batchlist[1].to('cpu'),marked_tens[2].to('cpu')))
            #print(torch.equal(batchlist[1], marked_tens[1]))
        # Concatenate features
        features = features.to('cuda:1', non_blocking=True)
        features_wsi = torch.cat((features_wsi, features), 0)
        i += 1

    return features_wsi.to('cuda:0', non_blocking=True)

def train_encoder_distr_parallel(dataloader, model, device, number_markings):
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
    i = 1
    batchlist = []
    features_wsi = None
    #model = model.to(device)
    rank = 3
    dist.init_process_group('nccl', world_size =4)
    model = model.getEncoder().to()
    ddp_model = DDP(model, device_ids = [4])

    for batch in tqdm(dataloader):
        if i<=number_markings:
            batchlist.append(batch)
            batch = batch.to(rank, non_blocking=True)
            # Compute features
            features = ddp_model(batch)
            print(features.requires_grad)
            #features = features.to('cuda:1', non_blocking=True)
            print('marked')
        else:
            with torch.no_grad():
                batchlist.append(batch)
                batch = batch.to(rank, non_blocking=True)
                
                # Compute features
                features = ddp_model(batch)
                #features = features.to('cuda:1', non_blocking=True)
        if i == 1:
            features_wsi = features
            features_wsi = features_wsi.to('cuda:1', non_blocking=True)

        #if i == 4:
            #used to confirm that the first two batches are identical with the marked batches 
            #print(torch.equal(batchlist[0], marked_tens[0]))
            #print(not torch.equal(batchlist[0].to('cpu'),marked_tens[2].to('cpu')))
            #print(not torch.equal(batchlist[1].to('cpu'),marked_tens[2].to('cpu')))
            #print(torch.equal(batchlist[1], marked_tens[1]))
        # Concatenate features
        features = features.to('cuda:1', non_blocking=True)
        features_wsi = torch.cat((features_wsi, features), 0)
        i += 1

    return features_wsi.to('cuda:0', non_blocking=True)

def check_feat_dir_contains_files(wsi):
    
    for ele in wsi.feature_file_names:
        if not os.path.isfile(ele):
            print('pen')
            return False

    return True

def train_decoder(model, loss_fn, features, optimizer, setting, device, label, patient, wsi, draw_map, para=False):
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
    features = features.to(device)
    print('begun decoder run')
    print(features.device)

    print(next(model.getDecoder().parameters()).device)
    logits, Y_prob, Y_hat, A = model.getDecoder()(features, label)
    logits.retain_grad()
    
    #conversions of Tensor types for loss calculation
    label = label.type(torch.LongTensor)
    print(para)
    if para:
        label = label.to('cuda:0')
    else:
        label = label.to(device)
    print(label.device)

    logits = logits.to(device)
    
    acc_logger.log(Y_hat, label)
    logits = logits[0][0].view(1)
    print(logits)
    print(label)
    loss = loss_fn(logits, label)
    loss_value = loss.item()
    #print the current loss
    print(loss_value)
    error = calculate_error(Y_hat, label)
    print(error)

    total_loss= loss
    print(total_loss)
    total_loss.backward()
    #print(logits.grad)
    #print(marked_output.grad)
    optimizer.step()
    #print('-------------------------')
    #print('marked gradient')
    #print(marked_output.grad)
    #print('-------------------------')
    #print('unmarked gradient')
    #print(unmarked_output.grad)
    
    optimizer.zero_grad()
    # Get attention map for predicted class
    if check_feat_dir_contains_files(wsi):
        features, keys = patient.get_features(wsi)
        A = A[Y_hat]
        A = A.view(-1, 1).cpu().detach().numpy()
        # Get tile keys in attention map
        if draw_map:
            # Save attention map
            patient.set_map(A, keys, wsi)
    #encoder post train
    encoder_post_train = model.getEncoder().state_dict()

    #compare models
    #compare_models(encoder_pre_train, encoder_post_train)
    return loss_value, error

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
    return error, loss, Y_prob