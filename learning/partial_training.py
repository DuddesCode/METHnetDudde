import torch

from models.restnet_custom import resnet50_baseline
from progress.bar import IncrementalBar
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from models.model_Full import Partial_Net
from models.attention_model import Attention_MB
from learning.logger import EarlyStopping, Accuracy_Logger
from .train import calculate_error
import sys
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
        """
        # WSI
        self.X = X
        # Tile property index
        self.i = i

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
        a = self.X.get_tiles_list()[self.i][idx].get_image(self.X.get_image(), True)
        # Transpose for channel first
        a = np.transpose(a, (2, 0, 1))
        # Cast to float
        a = a.astype(np.float32)
        # Check if has three dimensions
        if len(np.shape(a)) == 3:
            a = np.expand_dims(a, axis=0)
        # To torch tensor
        all_p = torch.from_numpy(a)
        
        return all_p

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

def construct_features(patients, setting, fold):
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
    # Get Encoder - default pretrained ResNet50
    param_Encoder = resnet50_baseline(pretrained=True)
    param_Encoder.train()
    # Get Decoder
    param_Decoder = Attention_MB(setting)
    param_Decoder.train()
    # get Partial Model
    model = Partial_Net(encoder=param_Encoder, decoder=param_Decoder)
    # To GPU if available
    print(torch.cuda.is_available())
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
    print('at epoch')
    for epoch in range(setting.get_network_setting().get_epochs()):
    #Track progress
        
        print('at bar')
        # Iterate patient classes
        for patient in patients:
            # Iterate patients in class
            for p in patient:
                # Iterate image properties
                print(device)
                label = p.get_diagnosis().get_label()
                if label == 0:
                    label = torch.zeros(1)
                    label = label.to(device)
                    print(label.device)
                else:
                    label = torch.ones(1)
                    label = label.to(device)
                    print(label.device)
                
                for wp in p.get_wsis():
                    # Iterate WSIs with image property
                    for wsi in wp:
                        
                        # Iterate Tile properties
                        bar_wsi = IncrementalBar('WSIbar ', max=len(wsi.get_tile_properties()))
                        for i in range(len(wsi.get_tile_properties())):
                            # If no tiles then nothing to be done
                            if len(wsi.get_tiles_list()) != 0:
                                # Open WSI
                                wsi.load_wsi()
                                # Create Dataset
                                data = Patient_Dataset(wsi, i)
                                # Create Dataloader
                                loader = DataLoader(dataset=data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                                # patient features for tile property
                                patient_features = torch.zeros((0, feature_setting.get_feature_dimension()))
                                # Pass batches
                                i = 0
                                #Encoder before training
                                encoder_pre_train = copy.deepcopy(model.getEncoder()).state_dict()
                                features_wsi = None
                                #used to determine whether a gradient was computed for a marked output and None for an unmarked
                                marked_output = None
                                unmarked_outputs = None
                                bar_batches = IncrementalBar('batchBar', max = len(loader)) 
                                for batch in tqdm(loader):
                                    if i<=2:
                                        batch = batch.to(device, non_blocking=True)
                                        features = model.getEncoder()(batch)
                                        print('marked')
                                        #features.to(device)
                                    if i == 0:
                                        features_wsi = features
                                        marked_output = features
                                        #print(marked_output.grad)
                                        marked_output.retain_grad()
                                        print(marked_output.grad)
                                    else:
                                        with torch.no_grad():
                                            batch = batch.to(device, non_blocking=True)
                                            # Compute features
                                            features = model.getEncoder()(batch)
                                            #features = features.cpu()
                                    if i == 4:
                                        unmarked_outputs = features
                                    # Concatenate features
                                    features_wsi = torch.cat((features_wsi, features), 0)
                                    i += 1
                                    print('catted features')

                                # Close WSI
                                wsi.close_wsi()
                                features_wsi.retain_grad()
                            
                            """
                            # Create directory to save features and keys
                            feature_directory = {}
                            # Fill directory
                            for j, tile in enumerate(wsi.get_tiles_list()[i]):
                                # Key Tile position, value feature of Tile
                                feature_directory[tile.get_position()] = patient_features[j]
                            # Save features
                            wsi.set_features(feature_directory, i)"""
                        #TODO integrate the Decoder Process depending on the architecture
                        n_classes = setting.get_class_setting().get_n_classes()
                        acc_logger = Accuracy_Logger(n_classes=n_classes)
                        features_wsi.to(device)
                        logits, Y_prob, Y_hat, A = model.getDecoder()(features_wsi, label)
                        logits.retain_grad()
                        print(logits.grad)
                        label = label.type(torch.LongTensor)
                        label = label.to(device)
                        logits = logits.to(device)
                        #logits.retain_grad()
                        print('begun decoder run')
                        acc_logger.log(Y_hat, label),
                        loss = loss_fn(logits, label)
                        print(marked_output.grad)
                        loss_value = loss.item()
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
                        print(unmarked_outputs.grad)
                        
                        optimizer.zero_grad()
                        #encoder post train
                        encoder_post_train = model.getEncoder().state_dict()

                        #compare models
                        compare_models(encoder_pre_train, encoder_post_train)
                        sys.exit()
                        bar_wsi.next()
                
                    bar_wsi.finish()