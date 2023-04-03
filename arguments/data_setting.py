from tkinter import Image
import utils.helper as helper
from albumentations import (RandomCrop, Normalize, Compose)

class ImageProperty(object):
    """ Class to store parameters of one WholeSlideImage type to use

    Attributes
    ----------
    staining : string
        staining to use same as in WSI file name
    scanner : string
        scanner to use same as in WSI file name
    magnification : int
        magnification to use same as in WSI file name
    preparation : string
        Preparation type to use currently not considered
    used_magnification : int
        magnification to use from image pyramid - closest will be used

    Methods
    -------
    get_staining()
        Return staining
    get_scanner()
        Return scanner
    get_magnification()
        Return magnification
    get_preparation()
        Return preparation
    get_file_ending()
        Return fie type of WSI of used scanner
    get_used_magnification()
        Return used_magnification
    """
    def __init__(self, staining, scanner, magnification, preparation, used_magnification):
        """
        Parameters
        ----------
        staining : string
            Staining to use same as in WSI file name
        scanner : string
            Scanner to use same as in WSI file name
        magnification : int
            Magnification of WSI to use same as in WSI file name
        preparation : string
            Preparation type to use - currently unused
        used_magnification : int
            magnification to use from image pyramid - closest will be used
        """
        # Staining of WSI to use
        self.staining = staining
        # Scanner to use
        self.scanner = scanner
        # Magnification to use
        self.magnification = magnification
        # Fixation to use
        self.preparation = preparation
        # Magnification factor in image pyramid to use <= magnification
        self.used_magnification = used_magnification

    def get_staining(self):
        """ Return staining
        Returns
        -------
        string
            staining
        """
        return self.staining

    def get_scanner(self):
        """ Return scanner
        Returns
        -------
        string
            scanner
        """
        return self.scanner

    def get_magnification(self):
        """ Return magnification
        Returns
        -------
        int
            magnification
        """
        return self.magnification

    def get_preparation(self):
        """ Return preparation
        Returns
        -------
        string
            preparation
        """
        return self.preparation

    def get_file_ending(self):
        """ Return file type for used scanner
        Returns
        -------
        string
            file type of scanner extend for new scanners
        """
        # Extend for new Scanners
        file_endings = {
            'Leica':'.svs',
            'Hamamatsu':'.ndpi'
        }

        return file_endings[self.scanner]

    def get_used_magnification(self):
        """ Return used_magnification
        Returns
        -------
        int
            used_magnification
        """
        return self.used_magnification

class TileProperty(object):
    """ Class to store parameters for Tiles to generate

    Attributes
    ----------
    tile_size : tuple (int, int)
        Size of Tile in pixel (x, y)
    tile_overlap : tuple (int, int)
        Overlap between two adjacent tiles in pixel (x, y)
    input_size : tuple (int, int)
        Expected size of image for Encoder in pixel as tuple (x, y) should be smaller or equal to tile_size
    augmentations : Albumentations object
        Augmentation sequence to apply to Tiles

    Methods
    -------
    get_tile_size()
        Return tile_size
    get_tile_overlap()
        Return tile_overlap
    get_input_size()
        Return input_size
    get_augmentations
        Return augmentations
    """
    def __init__(self, tile_size, tile_overlap, input_size, augmentations):
        """
        Parameters
        ----------
        tile_size : tuple (int, int)
            Size of tile in pixel (x, y)
        tile_overlap : tuple (int, int)
            Overlap between two adjacent tiles in pixel (x, y)
        input_size : tuple (int, int)
            Expected size for Encoder in pixel (x, y)
        augmentations : Albumentations object
            Augmentation sequence to apply to Tiles
        """
        # Size of tile in pixel as tuple
        self.tile_size = tile_size
        # Overlap of tiles in pixel
        self.tile_overlap = tile_overlap
        # Size of input for Encoder network in pixel as tuple <= tile_size
        self.input_size = input_size
        # Augmentation sequence to use - see albumentations
        
        self.augmentations = augmentations

    def get_tile_size(self):
        """ Return tile_size
        Returns
        -------
        tuple (int, int)
            tile_size
        """
        return self.tile_size

    def get_tile_overlap(self):
        """ Return tile_overlap
        Returns
        -------
        tuple (int, int)
            tile_overlap
        """
        return self.tile_overlap

    def get_input_size(self):
        """ Return input_size
        Returns
        -------
        tuple (int, int)
            input_size
        """
        return self.input_size

    def get_augmentations(self):
        """ Return augmentations
        Returns
        -------
        Albumentations object
            augmentations
        """
        return self.augmentations

class DataSetting(object):
    """ Class to hold parameters for data used

    Attributes
    ----------
    excluded patients : list [string]
        List of patient identifiers to manually exclude from data
    data_folders : list [string]
        Folders where WSI data is stored
    csv_patients : string
        .csv file holding patient information
    json_tiling_folder : string
        Folder where .json with Tiling information should be stored
    attention_folder : string
        Folder where .json with Attention map information should be stored
    image_properties : list [ImageProperty]
        ImageProperty Objects used to define WSIs loaded
    use_only_stamp : bool
        True if want to exclude WSIs that do not come from stamp area - stamp area need to be define in .csv and in WSI file name
    filter_non_stamp : bool
        True if want to filter Tiles outside a marked area - expects an existing label map for the WSI
    label_map_folder : string
        Folder where label maps with marked area are stored
    tile_properties : list [TileProperty]
        TileProperty Objects used to define extracted Tiles for each WSI
    skip_existing : bool
        True if want to skip existing Tilings and Features and not redo them
    filter_background : bool
        True if want to substract background from tissue for minimum tissue percentage check on Tile
    filter_blood : bool
        True if want to substract blood from tissue for minimum tissue percentage check on Tile
    min_tissue_percentage : float
        Minimum percentage of tissue on Tile to be valid and not filtered
    split : tuple (float, float, float)
        Percentage of data to use for training/validation/test set should sum up to 1.0
    folds : list
        Folds to run if using k-fold cross-validation
    monte_carlo : bool
        True if using Monte-Carlo cross-validation instead k-fold
    monte_carlo_folder : string
        Folder where order of random shuffling in Monte-Carlo cross validation will be stored
    narrow_validation : bool
        True if want to predict only - all valid patients will be in test set
    feature_folder : string
        Folder where encoded features will be stored
    results_folder : string
        Folder where prediction results for each patient will be stored
    attention_statistics_folder : string
        Folder where attention statistic information will be stored


    Methods
    -------
    get_excluded_patients()
        Return excluded_patients
    get_data_folders()
        Return data_folders
    get_json_tiling_folder()
        Return json_tiling_folder
    get_attention_folder()
        Return attention_folder
    get_csv_patients()
        Return csv_patients
    get_image_properties()
        Return image_properties
    get_use_only_stamp()
        Return use_only_stamp
    get_filter_non_stamp()
        Return filter_non_stamp
    get_label_map_folder()
        Return label_map_folder
    get_tile_properties()
        Return tile_properties
    get_skip_existing()
        Return skip_existing
    get_filter_background()
        Return filter_background
    get_filter_blood()
        Return filter_blood
    get_min_tissue_percentage()
        Return min_tissue_percentage
    get_split()
        Return split
    get_folds()
        Return folds
    get_monte_carlo()
        Return monte_carlo
    get_monte_carlo_folder()
        Return monte_carlo_folder
    get_narrow_validation()
        Return narrow_validation
    get_feature_folder()
        Return feature_folder
    get_results_folder()
        Return results_folder
    get_attention_statistics_folder()
        Return attention_statistics_folder
    """
    def __init__(self, data_directories, csv_file, working_directory):
        """ 
        Parameters
        ----------
        data_directories : list [string]
            Paths to data directories with WSI
        csv_file : string
            Csv storing patients and their attributes
        working_directory : string
            working directory for data
        """
        
        # Identifier of patients to manually exclude
        self.excluded_patients = []

        # Data folder - Folder where WSI Images are
        self.data_folders = data_directories

        # Csv of patients with attributes
        self.csv_patients = csv_file
        # JSON Tiling folder
        self.json_tiling_folder = working_directory+'Tiling'
        helper.create_folder(self.json_tiling_folder)
        # Attentio Map folder
        self.attention_folder = working_directory+'Attention'
        helper.create_folder(self.attention_folder)

        # Choose which WSIs to use - detailed explanation in class
        self.image_properties = [
            ImageProperty('HE', 'Hamamatsu', 40, 'FFPE', 40),
            ImageProperty('HE', 'Hamamatsu', 40, 'FFPE', 20)
        ]

        # Set True if just want to use WSIs where EPIC comes from        
        self.use_only_stamp = False
        
        # Set True if want to filter for marked area
        self.filter_non_stamp = True
        self.label_map_folder = ''
        if self.filter_non_stamp:
            self.label_map_folder = working_directory+'/stamps/'
        

        # Choose which Tiles to generate - detailed explanation in class 
        self.tile_properties = [
            TileProperty((256, 256), (0, 0), (256, 256), 
            Compose([RandomCrop(256, 256, always_apply=True, p=1.0),
            Normalize(always_apply=True, p=1.0)]))
        ]

        # Set True if want to Skip existing Tilings - Rerun necessary if changes in tiling/filtering were made
        self.skip_existing = True

        # Set True if want to filter background from single tiles
        self.filter_background = True
        # Set True if want to filter blood from single tiles
        self.filter_blood = True
        # Minimum Percentage of tissue on a single tile
        self.min_tissue_percentage = 0.75

        # Split to use as percent 0..1 (Train, Validation, Test)
        self.split = (0.7, 0.1, 0.2)

        # Folds to run --> Not necessary if want to run Monte-Carlo
        self.folds = [0, 1, 2, 3, 4]
        # Set True if want to run Monte Carlo Folds instead of k-Fold
        self.monte_carlo = True
        # Folder to memorize Monte Carlo splits
        self.monte_carlo_folder = working_directory+'splits/'
        if self.monte_carlo:
            helper.create_folder(self.monte_carlo_folder)

        self.narrow_validation = False
        # Feature folder
        self.feature_folder = working_directory+'Features/'
        helper.create_folder(self.feature_folder)


        self.results_folder = working_directory+'Results/'
        helper.create_folder(self.results_folder)

        self.attention_statistics_folder = working_directory+'Attention_Statistics/'
        helper.create_folder(self.attention_statistics_folder)

    def get_excluded_patients(self):
        """ Return excluded_patients
        Returns
        -------
        list [string]
            excluded_patients
        """
        return self.excluded_patients

    def get_data_folders(self):
        """ Return data_folders
        Returns
        -------
        list [string]
            data_folders
        """
        return self.data_folders

    def get_json_tiling_folder(self):
        """ Return json_tiling_folder
        Returns
        -------
        string
            json_tiling_folder
        """
        return self.json_tiling_folder

    def get_attention_folder(self):
        """ Return attention_folder
        Returns
        -------
        string
            attention_folder
        """
        return self.attention_folder

    def get_csv_patients(self):
        """ Return csv_patients
        Returns
        -------
        string
            csv_patients
        """
        return self.csv_patients

    def get_image_properties(self):
        """ Return image_properties
        Returns
        -------
        list [ImageProperty]
            image_properties
        """
        return self.image_properties

    def get_use_only_stamp(self):
        """ Return use_only_stamp
        Returns
        -------
        bool
            use_only_stamp
        """
        return self.use_only_stamp

    def get_filter_non_stamp(self):
        """ Return filter_non_stamp
        Returns
        -------
        bool
            filter_non_stamp
        """
        return self.filter_non_stamp

    def get_label_map_folder(self):
        """ Return label_map_folder
        Returns
        -------
        string
            label_map_folder
        """
        return self.label_map_folder

    def get_tile_properties(self):
        """ Return tile_properties
        Returns
        -------
        list [TileProperty]
            tile_properties
        """
        return self.tile_properties

    def get_skip_existing(self):
        """ Return skip_existing
        Returns
        -------
        bool
            skip_existing
        """
        return self.skip_existing

    def get_filter_background(self):
        """ Return filter_background
        Returns
        -------
        bool
            filter_background
        """
        return self.filter_background

    def get_filter_blood(self):
        """ Return filter_blood
        Returns
        -------
        bool
            filter_blood
        """
        return self.filter_blood

    def get_min_tissue_percentage(self):
        """ Return min_tissue_percentage
        Returns
        -------
        float
            min_tissue_percentage
        """
        return self.min_tissue_percentage

    def get_split(self):
        """ Return split
        Returns
        -------
        tuple (float, float, float)
            split
        """
        return self.split

    def get_folds(self):
        """ Return folds
        Returns
        -------
        list [int]
            folds
        """
        return self.folds
        
    def get_monte_carlo(self):
        """ Return monte_carlo
        Returns
        -------
        bool
            monte_carlo 
        """
        return self.monte_carlo

    def get_monte_carlo_folder(self):
        """ Return monte_carlo_folder
        Returns
        -------
        string
            monte_carlo_folder
        """
        return self.monte_carlo_folder

    def get_narrow_validation(self):
        """ Return narrow_valdiation
        Returns
        -------
        bool
            narrow_validation
        """
        return self.narrow_validation

    def get_feature_folder(self):
        """ Return feature_folder
        Returns
        -------
        string
            feature_folder
        """
        return self.feature_folder

    def get_results_folder(self):
        """ Return results_folder
        Returns
        -------
        string
            results_folder
        """
        return self.results_folder

    def get_attention_statistics_folder(self):
        """ Return attention_statistics_folder
        Returns
        -------
        string
            attention_statistics_folder
        """
        return self.attention_statistics_folder