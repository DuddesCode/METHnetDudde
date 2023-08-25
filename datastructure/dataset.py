
from cgi import test
import pandas as pd
from progress.bar import IncrementalBar

from datastructure.diagnosis import Diagnosis as Diagnosis
from datastructure.patient import Patient as Patient
import os
import numpy as np

class Dataset(object):
    """
    A class to represent the datasets handeled, internally splitted into train/validation and test
    
    Attributes
    -------------
    setting : Setting
        A setting as specified by the class
    patients : list
        A list of all patients as Patient loaded from csv file, applied filtering reduces this list
    n_classes : int
        Number of classes for the classification problem taken from setting file
    train_set : list
        A list containing a list per class containing the patients as patient assigned to train set
    validation_set : list
        A list containing a list per class containing the patients as Patient assigned to validation set
    test_set : list
         A list containing a list per class containing the patients as Patient assigned to test set

    Methods
    -------------
    load_patients()
        Loads list of patients from csv file
    filter_excluded_patients()
        Filters excluded identifiers as specified in data setting from patients
    filter_wsi_existence()
        Filters patients for patients without a valid WSI for each image property
    assign_label()
        Sets label for each patient according to setting file for the current classification problem
    filter_inconsistent_methylation()
        Filters patients for patients with inconsistent methylation class & subclass 
    filter_methylation_confidence()
        Filters patients for patients with too low probability score as specified in setting in methylation class and/or subclass
    set_fold(k=0)
        Splits patients list into train/validation/test for current split k
    split_dataset_narrow()
        Splitting to append all patients to test set
    split_dataset_random(k=0)
        Splitting for Monte Carlo cross validation
    split_dataset(k=0)
        Splitting for k-fold cross validation
    get_n_classes()
        Getter n_classes
    get_train_set()
        Getter train set
    get_validation_set()
        Getter validation set
    get_test_set()
        Getter test set
    """

    def __init__(self, setting):
        """
        Parameters
        ----------
        setting : Setting
            A setting file as specified by the class
        """
        self.setting = setting

        # Load patients for dataset
        self.patients = self.load_patients()

        # Filter out excluded patients
        self.filter_excluded_patients()
        # Check patients if they have all required WSI available
        self.filter_wsi_existence()
        # Assign correct class label to patient
        self.assign_label()

        # Filter inconsistent methylation prediction if necessary
        if self.setting.get_class_setting().get_filter_inconsistent_methylation():
            self.filter_inconsistent_methylation()

        # Filter too low methylation confidence if necessary  
        if self.setting.get_class_setting().get_filter_methylation_confidence():
            self.filter_methylation_confidence()

        self.preload_imagedata()

        # number of classes equals amount of groups of cancer subtypes
        self.n_classes = self.setting.get_class_setting().get_n_classes()
        # Set fold default to first


        self.train_set = []
        self.validation_set = []
        self.test_set = []

        if len(self.setting.get_data_setting().get_folds()) != 0:
            self.set_fold(self.setting.get_data_setting().get_folds()[0])
        else:
            self.set_fold(0)

    def load_patients(self):
        """ Loads and creates patients as specified in the .csv file

        """
        # List of patients
        patients = []
        # Get CSV file of patients
        csv_file = self.setting.get_data_setting().get_csv_patients()
        df = pd.read_csv(csv_file)

        # Track loading of patients each row one patient
        bar = IncrementalBar('Loading patients: ', max=len(df))

        #Iterate dataframe
        for _, row in df.iterrows():
            # Identifier
            identifier = row['Histological ID']
            # Get class attribute to use
            class_attribute = row[self.setting.get_class_setting().get_class_attribute()]

            # Methylation class MNG and score
            methylation_class = row['Methylation Classes Meningioma-Classifier']
            methylation_class_score = row['probability class']
            # Methylation subclass MNG and score
            methylation_subclass = row['Methylation class family member (EPIC) Meningioma-Classifier']
            methylation_subclass_score = row['probability subclass']

            stamp_blocks = ['']
            if 'Block' in df.columns:
                stamp_blocks = row['Block'].split(" ")
            stamp_subblocks = ['']
            if 'Subblock' in df.columns:
                stamp_subblocks = row['Subblock'].split(" ")

            who_grade = row['WHO Grade']

            diagnosis = Diagnosis(self.setting, class_attribute, methylation_class, methylation_class_score, methylation_subclass, methylation_subclass_score, who_grade, stamp_blocks, stamp_subblocks)

            age = row['Age']
            sex = row['Sex']

            patient = Patient(self.setting, identifier, diagnosis, age, sex)

            patients.append(patient)

            bar.next()

        bar.finish()

                
        print("Initial {} patients".format(len(patients)))
        return patients

    def filter_excluded_patients(self):
        """ Filters patients for identifiers in excluded list in data setting
        """
        patients = []
        # Get ids to exclude
        excluded_ids = self.setting.get_data_setting().get_excluded_patients()
        for patient in self.patients:
            identifier = patient.get_identifier()
            # If patient id not to exclude append
            if not(identifier in excluded_ids):
                patients.append(patient)

        self.patients = patients

    def filter_wsi_existence(self):
        """ Filters patients which do not have a valid WSI object for each image property
        """
        # New list of patients
        patients = []
        # Track progress
        bar = IncrementalBar('Filtering patients for existing WSI: ', max=len(self.patients))
        # Iterate patients
        for patient in self.patients:

            valid_properties = 0
            # Check each image property
            for i in range(len(patient.get_wsis())):
                # If patient has WSI for image property increase number of valid properties
                if len(patient.get_wsis()[i]) > 0:
                    valid_properties += 1
            # Append patient if it has WSI for each image property
            if valid_properties == len(patient.get_wsis()):
                patients.append(patient)

            bar.next()

        bar.finish()
        print("Resulting in {} patients".format(len(patients)))
        self.patients = patients

    def assign_label(self):
        """ Assigns a label for each patient to the diagnosis object of each patient according to the class labels defined in class setting
        """
        patients = []
        # Track progress
        bar = IncrementalBar('Filtering for relevant classes: ', max=len(self.patients))
        for patient in self.patients:
            # Get patients diagnosis
            diagnosis = patient.get_diagnosis()
            # Check if patient class attribute has a label in class setting
            if diagnosis.get_class_attribute() in self.setting.get_class_setting().get_class_labels():
                # Set label according to class setting
                diagnosis.set_label(self.setting.get_class_setting().get_class_labels()[diagnosis.get_class_attribute()])         
                patients.append(patient)
            bar.next()
        
        bar.finish()
        print("Resulting in {} patients".format(len(patients)))
        self.patients = patients

    def filter_inconsistent_methylation(self):
        """ Filters patients with inconsistent methylation diagnosis (not matching class/subclass) 
        """
        patients = []
        #Track progress
        bar = IncrementalBar('Filtering inconsistent methylation class: ', max=len(self.patients))
        for patient in self.patients:
            # Check if patient diagnosis is consistent
            if patient.get_diagnosis().is_consistent():
                patients.append(patient)
            bar.next()

        bar.finish()
        print("Resulting in {} patients".format(len(patients)))
        self.patients = patients

    def filter_methylation_confidence(self):
        """ Filters patients with too low methylation confidence as specified in class setting in class and/or subclass
        """
        patients = []
        #Track progress
        bar = IncrementalBar('Filtering too low methylation confidence: ', max=len(self.patients))
        for patient in self.patients:
            # Get patient diagnosis
            diagnosis = patient.get_diagnosis()
            # Check if methylation class score is over confidence level for this class
            if diagnosis.get_methylation_class_score() >= self.setting.get_class_setting().get_confidence_scores_class()[diagnosis.get_methylation_class()]:
                # Check if methylation subclass score is over confidence level for this subclass
                if diagnosis.get_methylation_subclass_score() >= self.setting.get_class_setting().get_confidence_scores_subclass()[diagnosis.get_methylation_subclass()]:
                    patients.append(patient)
            bar.next()

        bar.finish()
        print("Resulting in {} patients".format(len(patients)))
        self.patients = patients

    def preload_imagedata(self):

        bar = IncrementalBar('Loading image data: ', max=len(self.patients))
        for patient in self.patients:
            wsis = patient.get_wsis()
            for wp in wsis:
                for wsi in wp:
                    tps = wsi.get_tiles_list()
                    wsi.load_wsi()
                    for tp in tps:
                        for tile in tp:
                            tile.get_image(wsi.get_image(),True)
                    wsi.close_wsi()

            bar.next()

        bar.finish()

    def set_fold(self, k=0):
        """ Sets the current split for train/validation/test set. Splitting can be done different ways according to parameters in data setting.
        If narrow validation is set all are assigned to test set. Else if Monte Carlo is set splitting is done in random Monte Carlo manner. 
        Else k-fold cross validation is performed.

        Parameters
        ----------
        k : int
            current split or iteration of monte carlo cross validation
        """
        # If narrow validation all in test set
        if self.setting.get_data_setting().get_narrow_validation():
            self.test_set = self.split_dataset_narrow()
        else:
            # If monte carlo cross-validation use random data split
            if self.setting.get_data_setting().get_monte_carlo():
                self.train_set, self.validation_set, self.test_set = self.split_dataset_random(k=k)
            # Else use fixed split
            else:    
                self.train_set, self.validation_set, self.test_set = self.split_dataset(k=k)

    def split_dataset_narrow(self):
        """ Assigns all patients to test set.

        Returns
        -------
        list 
            A list containining a list per class containing Patient objects, all patients are assigned to those list
        """
        test_set = []

        # Iterate over classes
        for c in range(self.setting.get_class_setting().get_n_classes()):
            # Get patients for this class
            class_patients = [patient for patient in self.patients if patient.get_diagnosis().get_label() == c]
                        # Count number of patients for class

            test_set.append(class_patients)

        return test_set

    def split_dataset_random(self, k=0):
        """ Assigns patients to train/validation and test set. Assignment is based on random shuffling and one .csv with the order of identifiers per class 
        is saved to rememember the shuffling. If split was already done before .csv file order is assessed instead.
        Splitting proportion is done per class and according to the splits specified in
        data setting. All three sets are mutually exclusive. 

        Parameters
        ----------
        k : int
            Current split to be performed
            
        Raises
        ------
        Exception
            If not at least 3 patients per class are available, no split can be performed then

        Returns
        -------
        
        list 
            A list containining a list per class containing Patient objects resembling the train set
        list 
            A list containining a list per class containing Patient objects resembling the validation set
        list
            A list containining a list per class containing Patient objects resembling the test set
        """
        # Get split proportion
        split = self.setting.get_data_setting().get_split()
        
        # Initialize Sets
        train_set = []
        validation_set = []
        test_set = []

        # Iterate over classes
        for c in range(self.setting.get_class_setting().get_n_classes()):
            # Get patients for this class
            class_patients = [patient for patient in self.patients if patient.get_diagnosis().get_label() == c]
                        # Count number of patients for class
            n_in_class = len(class_patients)
            # File to memorize this split
            indices_file_name = self.setting.get_data_setting().get_monte_carlo_folder() + str(k)+'_'+str(c)+'.csv'
            # If file already existing use it
            if os.path.exists(indices_file_name):
                df = pd.read_csv(indices_file_name)
                identifications = df['Order'].to_list()
            else:
                all_indices = np.arange(n_in_class)
                # Randomly shuffle patients
                np.random.shuffle(all_indices)
                identifications = []
                # Write patient identifiers in this order
                for i in range(len(all_indices)):
                    identification = class_patients[all_indices[i]].get_identifier()
                    identifications.append(identification)
                # Write to csv
                df = pd.DataFrame({'Order':identifications})
                df.to_csv(indices_file_name)

            if n_in_class < 3:
                raise Exception('Not enough patients (n>=3) in class')
            # Number of patients in test according to split
            n_test = max(int(n_in_class * split[2]),1)
            # Number of patients in validation according to split
            n_validation = max(int(n_in_class * split[1]),1)

            # Initialize subsets
            train_set_class = []
            validation_set_class = []
            test_set_class = []

            for i in range(n_in_class):
                # Get patient with correct identifier
                cur_p = [p for p in class_patients if p.get_identifier() == identifications[i]][0]
                # Append to correct subset
                if i < n_test:
                    test_set_class.append(cur_p)
                elif i < n_test+n_validation:
                    validation_set_class.append(cur_p)
                else:
                    train_set_class.append(cur_p)
            
            # Append class set
            train_set.append(train_set_class)
            validation_set.append(validation_set_class)
            test_set.append(test_set_class)
            
        return train_set, validation_set, test_set

    def split_dataset(self, k=0):
        """ Assigns patients to train/validation and test set. Iterates through subsets for split.
        Splitting proportion is done per class and according to the splits specified in
        data setting. All three sets are mutually exclusive. 

        Parameters
        ----------
        k : int
            Current split to be performed

        Raises
        ------
        Exception
            If not at least 3 patients per class are available, no split can be performed then

        Returns
        -------
        
        list 
            A list containining a list per class containing Patient objects resembling the train set
        list 
            A list containining a list per class containing Patient objects resembling the validation set
        list
            A list containining a list per class containing Patient objects resembling the test set
        """

        split = self.setting.get_data_setting().get_split()

        train_set = []
        validation_set = []
        test_set = []

        for c in range(self.setting.get_class_setting().get_n_classes()):
            class_patients = [patient for patient in self.patients if patient.get_diagnosis().get_label() == c]
            n_in_class = len(class_patients)

            if n_in_class < 3:
                raise Exception('Not enough patients (n>=3) in class')
            # Number of patients in test according to split
            n_test = max(int(n_in_class * split[2]),1)

            test_indices = np.arange(k*n_test, min(k*n_test+n_test,len(class_patients)))
            test_set_class = np.array(class_patients)[test_indices].tolist()
            test_set.append(test_set_class)
        
            n_validation = max(int(n_in_class * split[1]),1)
            validation_indices = np.arange((k*n_test+n_test)%n_in_class,min((k*n_test+n_test)%n_in_class + n_validation,len(class_patients)))
            validation_set_class = np.array(class_patients)[validation_indices].tolist()
            validation_set.append(validation_set_class)

            train_indices = np.array([n for n in np.arange(0,n_in_class) if n not in test_indices and n not in validation_indices])
            train_set_class = np.array(class_patients)[train_indices].tolist()
            train_set.append(train_set_class)


        return train_set, validation_set, test_set

    
    def get_n_classes(self):
        """ Getter n_classes

        Returns
        -------
            int 
                number of classes in classification problem
        """
        return self.n_classes

    def get_train_set(self):
        """ Getter train_set
        
        Returns
        -------
            list
                List containing a list per class containing Patient objects assigned to train set
        """
        return self.train_set

    def get_validation_set(self):       
        """ Getter validation_set
        
        Returns
        -------
            list
                List containing a list per class containing Patient objects assigned to validation set
        """
        return self.validation_set

    def get_test_set(self):
        """ Getter test_set
        
        Returns
        -------
            list
                List containing a list per class containing Patient objects assigned to test set
        """
        return self.test_set
    
    def set_settings(self, settings):
        """MD set settings to new settings due to image preload condition"""
        self.setting = settings

