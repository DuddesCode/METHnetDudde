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
    
    clean_dict_dataset = []
    #first loop sorts the marked datasets first
    for ele in tqdm(iter(dataset)):
        if ele['mark']:
            clean_dict_dataset = [{"img":ele['all_p'],"pos":ele['tile_pos']}] + clean_dict_dataset
            
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
    for i in range(768):
        list_of_marked.append(clean_dataset[i])

    return clean_dataset, list_of_marked, pos_list

class CleanDataset(Dataset):

    def __init__(self, clean_dataset) -> None:
        self.clean_dataset = clean_dataset

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        return self.clean_dataset[idx]


def partial_training(patients, patients_val, setting, fold):
    """ Create features for patients and save them
    Parameters
    ----------
    patients : list [[Patient]]
        list of list per class of Patient to create features for
    setting : Setting
        Setting as defined by class
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
        print(n_patients)
        for patient in patients:
            # Iterate patients in class
            for p in patient:
                # Iterate image properties
                label = p.get_diagnosis().get_label()
                if label == 0:
                    label = torch.zeros(1)
                    label = label.to(device)
                else:
                    label = torch.ones(1)
                    label = label.to(device)
                #print(p.get_wsis())
                for wp in p.get_wsis():
                    
                    print(wp)
                    # Iterate WSIs with image property
                    for wsi in wp:
                        print('ED')
                        
                        # Iterate Tile properties
                        for i in range(len(wsi.get_tile_properties())):
                            # If no tiles then nothing to be done
                            if len(wsi.get_tiles_list()) != 0:
                                # Open WSI
                                wsi.load_wsi()
                                #give random tiles of the WSI a marking
                                selection = AttentionSelection(setting=feature_setting, wsi=wsi, epoch=epoch)
                                #selection = SolidSelection(setting=feature_setting, wsi=wsi)
                                #selection = RandomSelection(setting = feature_setting, wsi = wsi)
                                #randomlist = selection.tile_marking()
                                selection.tile_marking()
                                # Create Dataset
                                data = Patient_Dataset(wsi, i)
                                print('data length')
                                print(len(data))
                                data, marked_tensors, pos_list = create_cleaned_dataset(data)
                                data = CleanDataset(data)
                                print(len(data))
                                # Create Dataloader
                                loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                                marked_loader = DataLoader(dataset=marked_tensors, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(marked_tensors))
                                marked_tensors_list = []
                                for ele in marked_loader:
                                    marked_tensors_list.append(ele)
                                # patient features for tile property
                                patient_features = torch.zeros((0, feature_setting.get_feature_dimension()))
                                # Pass batches
                                i = 0
                                #Encoder before training
                                encoder_pre_train = copy.deepcopy(model.getEncoder()).state_dict()
                                #used to determine whether a gradient was computed for a marked output and None for an unmarked
                                #iterates per batch in the Dataloader
                                unmarked_outputs, marked_output, features_wsi = train_encoder(loader, model, device, marked_tensors_list, selection.get_marked_batches())
                                # Close WSI
                                selection.tile_resetting()
                                wsi.close_wsi()
                                features_wsi.retain_grad()
                                
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
                        #train the decoder
                        train_decoder(model, loss_fn, features_wsi,  optimizer, marked_output, unmarked_outputs, encoder_pre_train, setting, device, label, p, wsi)
        stop = validate_partial_net_epoch(epoch, model, n_classes, loss_fn, early_stopping, patients_val, feature_setting, ckpt_name=model_folder + model_file)
        if stop and setting.get_network_setting().get_early_stopping():
            break
                            
                        
def validate_partial_net_epoch(epoch, model, n_classes, loss_fn, early_stopping, patients_val, feature_setting, ckpt_name,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.getDecoder().eval()
    model.getEncoder().eval()
    model.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.
    # Just forward pass needed
    with torch.no_grad():
        val_error_pat = 0.
        val_loss_pat = 0.
        patient_counter = len(patients_val)
        print(patients_val)
        for p in patients_val:
            print(p)
            wsi_prop_counter = len(p[0].get_wsis())
            val_error_wsi_prop = 0.
            val_loss_wsi_prop = 0.
        # Iterate image properties
            label = p[0].get_diagnosis().get_label()
            if label == 0:
                label = torch.zeros(1)
                label = label.to(device)
            else:
                label = torch.ones(1)
                label = label.to(device)
            for wsi_prop in p[0].get_wsis():
                wsi_counter = len(wsi_prop)
                val_error_wsi = 0.
                val_loss_wsi = 0.
                for wsi in wsi_prop:
                    tileprop_counter = 0
                    val_loss_tile_prop = 0.
                    val_error_tile_prop = 0.
                    for i in range(len(wsi.get_tile_properties())):
                        val_error_tile = 0. 
                        val_loss_tile = 0.
                        if len(wsi.get_tiles_list()) != 0:
                            wsi.load_wsi()
                            data = Patient_Dataset(wsi, i)
                            data, marked_tens, pos_list = create_cleaned_dataset(data)
                            data = CleanDataset(data)
                            loader = DataLoader(dataset= data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                            for batch in tqdm(loader):
                                
                                features = test_encoder(loader, model, device)
                                error, Y_prop = test_decoder(model, features, label, p[0], False)
                                data, label = data.to(device), label.to(device)
                                # Forward pass
                                logits, Y_prob, Y_hat, A = model(data, label=label)

                                acc_logger.log(Y_hat, label)

                                loss = loss_fn(logits, label)

                                val_loss_tile += loss.item()

                                error = calculate_error(Y_hat, label)

                                val_error_tile += error
                            wsi.close_wsi()
                            #compute error of tileprop
                            val_error_tile /= len(loader)
                            val_loss_tile /= len(loader)
                            val_error_tile_prop += val_error_tile
                            val_loss_tile_prop += val_loss_tile
                        tile_prop_counter = i
                    
                    #compute error per wsi
                    val_loss_wsi += val_loss_tile_prop / tileprop_counter
                    val_error_wsi += val_error_tile_prop / tileprop_counter
                    wsi_counter += 1
                #compute error per wsi_prop
                val_error_wsi_prop += val_error_wsi / wsi_counter
                val_loss_wsi_prop += val_loss_wsi / wsi_counter
                wsi_prop_counter += 1
            #compute error per patient
            val_error_pat += val_error_wsi_prop / wsi_prop_counter
            val_loss_pat += val_loss_wsi_prop / wsi_prop_counter
        #compute error overall
        val_error = val_error_pat / patient_counter
        val_loss= val_loss_pat / patient_counter
    # Compute Early stopping
    early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)

    if early_stopping.early_stop:
        return True

    return False

def train_encoder(dataloader, model, device, marked_tens, number_markings, randomList = None, selection = None):
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
        print('catted features')

    return unmarked_output, marked_output, features_wsi

def check_feat_dir_contains_files(wsi):
    
    for ele in wsi.feature_file_names:
        if not os.path.isfile(ele):
            print('pen')
            return False

    return True

def train_decoder(model, loss_fn, features, optimizer, marked_output, unmarked_output, encoder_pre_train, setting, device, label, patient, wsi, draw_map=True):
    
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
    #compare_models(encoder_pre_train, encoder_post_train)

def test_encoder(dataloader, model, device):
    i = 0

    features_wsi = None
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            features = model.getEncoder(batch)
            if i == 0:
                features_wsi = features
            else:
                features_wsi = torch.cat((features_wsi, features), 0)
    
    return features_wsi

def test_decoder(model, features, label, patient, draw_map=True):
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, A = model.getDecoder()(features, label)

    error = calculate_error(Y_hat, label)
    A = A[Y_hat]
    A = A.view(-1, 1).cpu().numpy()
    # Get tile keys in attention map
    keys = keys.cpu().numpy()[0]
    if draw_map:
        # Save attention map
        patient.set_map(A, keys)

    return error, Y_prob