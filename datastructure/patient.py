import os

from datastructure.wsi import WholeSlideImage as WholeSlideImage
import numpy as np

from scipy.stats import rankdata

class Patient(object):
    """
    A class to represent one patient with diagnosis and histological slides.

    Attributes
    ----------
    identifier : string
        (Anonymized) identifier of patient, should also be part of WSI name
    diagnosis : Diagnosis
        diagnosis of patient as specified by class
    setting : Setting
        A setting as specified by the class
    age : int
        Age of the patient at biopsy
    sex : string
        Sex of the patient 
    wsis : list
        A list containing a list per image property containing all WSIs for this patient available as WholeSlideImage

    Methods
    -------
    load_wsis()
        Check existing WSIs and create the list wsis according to image properties in data setting
    get_possible_wsis(data_folder, image_property)
        Check in data folder for WSIs with patient identifier and corrent image properties in file name
    get_identifier()
        Getter patient identifier
    get_diagnosis()
        Getter patient diagnosis
    get_wsis()
        Getter patient wsis
    get_features()
        Getter encoded patient features
    save_predicted_scores(folder)
        Save prediction scores stored in patient diagnosis
    set_map(A, keys)
        Set attention map for each WSI of patient
    save_map()
        Save attention maps of each WSI
    """
    def __init__(self, setting, identifier, diagnosis, age=-1, sex='?'):
        """
        Parameters
        ----------
        setting : Setting
            A setting file as specified by the class
        identifier : string
            The identifier of the patient should be of form "ABC-DE"
        diagnosis : Diagnosis
            The diagnosis of the patient as specified by the class
        age : int
            The age of the patient at biopsy
        sex : string
            The sex of the patient
        """ 
        # Identifier
        self.identifier = identifier
        # Diagnosis of patient
        self.diagnosis = diagnosis

        self.setting = setting
        # Age of patient 
        self.age = age

        # Sex of patient
        self.sex = sex
        # Load WSIs of patient
        self.wsis = self.load_wsis()

    def load_wsis(self):
        """ Checks data folder and creates a WholeSlideImage object for each WSI file with matching identifier and image properties in file name
        Image properties and data folders as specified in data setting. If Use only stamp specified in data setting filtered for those slides where
        diagnosis is from.

        Returns
        -------
        list
            A list containing a list per image property containing all possible WholeSlideImage object for this patient and image property
        """
        # List of list of WSIs
        wsis = []
        # Get wanted image properties
        image_properties = self.setting.get_data_setting().get_image_properties()

        # Iterate image properties
        for image_property in image_properties:
            # List of WSIs for this property
            wsis_property = []
            # Get data folders
            data_folders = self.setting.get_data_setting().get_data_folders()
            # Iterate data folders
            for data_folder in data_folders:
                # Get all possible WSIs
                wsi_identifiers = self.get_possible_wsis(data_folder, image_property)
                # Try to Load each WSI for this patient for this property
                for wsi_identifier in wsi_identifiers:
                    # Create WSI object
                    wsi = WholeSlideImage(self.setting, wsi_identifier, image_property, data_folder)
                    # If using only such from a stamp check
                    if self.setting.get_data_setting().get_use_only_stamp():
                        # Get stamp locations
                        stamp_blocks = self.get_diagnosis().get_stamp_blocks()
                        stamp_subblocks = self.get_diagnosis().get_stamp_subblocks()
                        # Check wheter it is within
                        for i in range(len(stamp_blocks)):
                            if wsi.get_block() == stamp_blocks[i] and wsi.get_subblock() == stamp_subblocks[i]:
                                wsi.set_tiles(wsi.load_tiles())
                                wsis_property.append(wsi)
                                break
                    else:
                        wsi.set_tiles(wsi.load_tiles())
                        wsis_property.append(wsi)

            # Apppend to list of lists
            wsis.append(wsis_property)
 
        return wsis

    def get_possible_wsis(self, data_folder, image_property):
        """ Checks data_folder for matching WSIs according to file name. 
        File name of a WSI therefore must be as following : 
        Patient identifier followed by (optional) WSI identifier
        Afterwards seperated by underscore ('_') in following order as specified in image property
        staining, scanner and magnification followed by 'x' and valid file ending for scanner

        Parameters
        ----------
        data_folder : string
            path to data folder to check for WSIs
        image_property : ImageProperty
            ImageProperty object defining properties of WSI (staining, scanner, magnification) to search for

        Returns
        -------
        list
            List of matching WSI identifiers as strings
        """
        # Get correct file ending
        file_ending = image_property.get_file_ending()
        # All WSIs in folder with correct ID
        possible_wsis = [f for f in os.listdir(data_folder) if f.startswith(str(self.identifier))]
        # Filter for those with correct file format 
        possible_wsis = [f for f in possible_wsis if f.endswith(file_ending)]
        # Check if additional information is in file name
        
        # Filter for those with correct staining
        possible_wsis = [f for f in possible_wsis if f.split('_')[1] == image_property.get_staining()]
        # Filter for those with correct scanner
        possible_wsis = [f for f in possible_wsis if f.split('_')[2] == image_property.get_scanner()]
        # Filter for those with correct magnification
        possible_wsis = [f for f in possible_wsis if f.split('_')[3] == str(image_property.get_magnification())+'x'+file_ending]
        # Get WSI identifiers
        possible_wsis = [f.split('_')[0] for f in possible_wsis]

        return possible_wsis

    def get_identifier(self):
        """ Getter patient identifier

        Returns
        -------
        string
            patient identifier
        """
        return self.identifier

    def get_diagnosis(self):
        """ Getter patient diagnosis
        
        Returns
        -------
        Diagnosis
            patients diagnosis object
        """
        return self.diagnosis

    def get_wsis(self):
        """ Getter patients wsis

        Returns
        -------
        list
            A list containing a list per image property containing a list of matching WholeSlideImage objects
        """
        return self.wsis

    def get_features(self, wsi):
        """ Getter encoded patient features, concatenates all feature vectors of all WholeSlideImage objects and the keys of the tiles contained in them

        Returns
        -------
        numpy array
            A array of dimensionality (n, X) containing n feature vectors of size X as defined by network setting. One feature vector per tile. 
        list
            A list of tuples one per tile in the WholeSlideImage objects in the same order as their feature vectors
        """
        print('a')
        print(self.wsis)
        patient_features = np.zeros((0,self.setting.get_network_setting().get_F()))
        # Get all features for this WSI
        features, keys = wsi.get_all_features()
        features = np.concatenate((patient_features, features), axis = 0)        

        return features, keys

    def save_predicted_scores(self, folder):
        """ Save predicted scores in diagnosis object of patient

        Parameters
        ----------
        folder : string
            Path to folder to store scores to
        """
        self.diagnosis.save_predicted_scores(self.identifier, folder)

    def set_map(self, A, keys, wsi):
        """ Set attention map per WSI of patient

        Parameters
        ----------
        A : numpy array
            Numpy array with one value per tile of all WSIs
        keys : list
            List of tuples with a key per tile in the same order as A
        wsi : WSI Obj
            contains the wsi object for which the map needs to be set MD
        """
        # Compute relative Attention map over all Attention maps of this patient
        A_relative = rankdata(A, "dense")
        max_rank = np.max(np.array(A_relative))
        A_relative = np.array(A_relative) / max_rank

        tile_counter = 0
        # Count number of tiles for this WSI object (multiple tile properties might be possible)
        n_keys = sum([len(k) for k in wsi.get_tiles_list()])

        # Get keys for current WSI
        wsi_keys = keys[tile_counter:tile_counter+n_keys]
        # Get attention values for current WSI
        wsi_A = A[tile_counter:tile_counter+n_keys]
        # Get relative attention_values for current WSI
        wsi_A_relative = A_relative[tile_counter:tile_counter+n_keys]

        # Increment tile counter
        tile_counter += n_keys

        # Set map for WSI
        wsi.set_map(wsi_A, wsi_keys, wsi_A_relative)

    def save_map(self):
        """ Save attention map statistics to folder as specified by data setting and according to its image property for each WSI

        """
        for i, wp in enumerate(self.wsis):
            folder = self.setting.get_data_setting().get_attention_statistics_folder() + str(self.setting.get_data_setting().get_image_properties()[i].get_used_magnification()) + "/"

            for wsi in wp:
                wsi.save_map(folder)