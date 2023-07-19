from itertools import count
import preprocessing.reader as reader
import preprocessing.filter as filter

import os
import numpy as np

from utils import helper as helper

import json

import multiprocessing
from functools import partial

from datastructure.tile import Tile as Tile

from progress.bar import IncrementalBar
import pickle5 as pickle

from scipy.stats import rankdata

from matplotlib import pyplot as plt

def multiprocess_filtering(x, otsu_value, filter_background, filter_blood, tissue_percentage, tile_property, level):
    """ Check wheter at a position a valid is tile is possible. Filtering may be applied to check for background and blood.

    Parameters
    ----------
    x : tuple
        (Image, Position) where Image is a pillow Image of the Tile and Position is a tuple of its location in pixel (x,y) 
    otsu_value : int
        The original otsu value as obtained by filtering the overview image
    filter_background : bool
        True if want to exclude background from tissue percentage
    filter_blood : bool
        True if want to exclude blood from tissue percentage
    tissue_percentage : float
        Value (should be between 0 and 1) of minimum amount of tissue on Tile
    tile_property : TileProperty
        Properties of the tile to generate
    level : int
        Level of the image pyramid where tile was extracted

    Returns
    -------
    Tile
        Tile object if minimum percentage of tissue is in image else None
    """
    # Get image region
    region = x[0]
    # Get position of image region
    position = x[1]
    # If want to filter for background or blood
    if filter_background or filter_blood:
        # Check single tile if it contains enough tissue
        is_tissue = filter.check_tissue_tile(region, otsu_value, tissue_percentage, filter_background, filter_blood)
    else:
        # Else it is tissue
        is_tissue = True
    
    # If valid Tile create it
    if is_tissue:
        # Create Tile object
        tile_obj = Tile(position[0], position[1], tile_property.get_tile_size(),
        tile_property.get_input_size(), tile_property.get_augmentations(), level)
        # Return tile object
        return tile_obj
    # Else return None
    return None

class WholeSlideImage(object):
    """
    A class to represent a digitized histological slide of a patient

    Attributes
    ----------
    setting : Setting
        A Setting object as specified by the class
    identifier : string
        The identifier of the Whole Slide image
    block : string
        The block where the WSI comes from
    subblock : string
        The subblock where the WSI comes from
    cut : int
        The number of the cut from the paraffin block if z remains unknown
    z_level : int
        The z level of the cut from the paraffin block
    image_property : ImageProperty
        The image properties of the WSI
    tile_properties : list
        A list of TileProperty objects for which Tiles should be generated from the WSI
    data_folder : string
        The path to the folder where the WSI is in
    file_name : string
        The file name of the WSI in the following form : Identifier_Staining_Scanner_Magnification x
    file_path : string
        The complete file path
    image : OpenSlideObject
        The WSI as OpenSlide Object
    json_folder : string
        Path to folder where json files of Tiling should be stored
    json_file_name_scheme : string
        File name scheme for Tiling .json Files
    attention_folder : string
        Path to folder where json files of Attention Maps should be stored
    attention_file_name_scheme : string
        File name scheme for Attention Maps .json Files
    used_level : int
        Used level of image pyramid best match according to image property
    overview_image : Image
        Pillow image of full WSI at lowest resolution 
    downsampling_factor : float
        factor between used and lowest resolution of image pyramid
    factor : float
        factor between highest and lowest resolution of image pyramid
    overview_threshold : numpy array
        Thresholded (Otsu) overview image, valid tissue at low resolution
    otsu_value : int
        Value determined by thresholding overview image
    label_file_name : string
        Name of the label map exported from QuPath (optional)
    label_map : Image
        The label map exported from QuPath (optional) will be multiplied with overview_threshold
    size : tuple
        The size of the image at the used level
    tiles : list
        A list containing a list per TileProperty containing Tile object - all valid Tiles
    features : numpy array
        The econded features for all Tiles for all TileProperties     
    keys : list
        A list of tuples identifying the Tile object belonging to a feature (same order as feature)
    feature_folder : string
        Path to folder where features should be saved to
    feature_file_names : list
        List of names of feature files, one per TileProperty
    tiles_inside : list
        A list containing a list per TileProperty containing Tile objects - all tiles inside the region marked by label map
    tiles_outside : list
        A list containing a list per TileProperty containing Tile objects - all tiles outside the region marked by label map
    
    Methods
    -------
    load_wsi()
        Open OpenSlideObject
    close_wsi()
        Close OpenSlideObject
    load_tiles()
        Perform tiling per TileProperty and maintain tiles list
    load_tiles_json(json_file_name, tile_property)
        Load tiles into tiles list according to saved tiling in json file
    get_identifier()
        Getter wsi identifier
    get_block()
        Getter block
    get_subbblock()
        Getter subblock
    get_cut()
        Getter cut
    get_z_level()
        Getter z-level
    get_image_property()
        Getter used image property
    get_tiles_properties()
        Getter used tile properties
    get_data_folder()
        Getter path to data folder of WSI
    get_file_name()
        Getter file name of WSI
    get_file_path()
        Getter file path of WSI
    get_image()
        Getter OpenSlide Object
    get_overview_image()
        Getter overview image
    set_tiles(tiles)
        Set list of tiles
    get_tiles_list()
        Get list of tiles
    set_features(features, tile_property_index)
        Set encoded features for WSI at specific TileProperty
    get_feature_file_names()
        Getter names of all feature files for this WSI
    get_features(tile_property_index)
        Getter encoded features for specific TileProperty
    get_all_features()
        Getter concatenated feature vector over all TileProperties
    set_map(A, keys, A_relative)
        Set attention maps for WSI
    value_to_rgb(value)
        Compute RGB value for a value according to Colormap
    save_map(folder)
        Save attention maps
    get_attention_values(tile_list, nested=True)
        Get list of attention values for tile_list
    get_attention_values_relative(tile_list, nested=True)
        Get list relative (ranking over this property) attention values for tile_list 
    set_inside_outside()
        Compute which tiles are inside and outside of label map
    get_inside()
        returns a list of Tiles that are inside the marked area
    get_outside()
        returns a list of Tiles that are outside the marked area
    """
    def __init__(self, setting, identifier, image_property, data_folder):
        """
        Parameters
        ----------
        setting : Setting
            A Setting object specified a given by class
        identifier : string
            The identifier of the WSI - optional splitted by '-' with block,subblock,cut and z-level
        image_property : ImageProperty
            The ImageProperty object used for this WSI
        data_folder : string
            Path to folder where WSI file is within
        """
        self.setting = setting

        self.identifier = identifier
        # Check wheter Block information is given
        if len(identifier.split('-')) > 2:
            # Block WSI was taken from
            self.block = identifier.split('-')[3]
            # Subblock WSI was taken from
            self.subblock = identifier.split('-')[4]
            # Cut of WSI
            self.cut = identifier.split('-')[5]
            # z-Level of WSI
            self.z_level = identifier.split('-')[6]
        else:
            self.block = ''
            self.subblock = ''
            self.cut = 0
            self.z_level = 0

        # Image properties of WSI
        self.image_property = image_property
        # Tile properties of Tiles to generate
        self.tile_properties = self.setting.get_data_setting().get_tile_properties()
        print(self.tile_properties)
        # Data Folder
        self.data_folder = data_folder

        # File Name according to naming scheme
        self.file_name = self.identifier +\
            '_' + self.image_property.get_staining() +\
            '_' + self.image_property.get_scanner() +\
            '_' + str(self.image_property.get_magnification()) +\
            'x' + self.image_property.get_file_ending()

        # File path of WSI
        self.file_path = self.data_folder + '/' + self.file_name

        # WSI Image Object
        self.image = None

        # JSON Tiling Folder and File Name
        self.json_folder = self.setting.get_data_setting().get_json_tiling_folder()

        self.json_file_name_scheme = self.json_folder + '/' + self.file_name.split('.')[0] +\
            '_' + str(self.image_property.get_used_magnification())


        # JSON Attention Folder and File Name
        self.attention_folder = self.setting.get_data_setting().get_attention_folder()

        self.attention_file_name_scheme = self.attention_folder + '/' + self.file_name.split('.')[0] +\
            '_' + str(self.image_property.get_used_magnification())

        # Open WSI
        self.load_wsi()

        # Get best Level of image pyramid to use according to used magnification and native magnification
        self.used_level = self.image.get_best_level_for_downsample(float(self.image_property.get_magnification())/float(self.image_property.get_used_magnification()))
        
        # Get overview image
        self.overview_image, downsampling = reader.get_overview(self.image)

        # Downsampling factor of overview compared to extracted regions
        self.downsampling_factor = downsampling / self.image.level_downsamples[self.used_level]
        # Factor between highest resolution and overview
        self.factor = downsampling / self.image.level_downsamples[0]

        # Threshold overview image
        self.overview_threshold, self.otsu_value = filter.filter_overview_image(self.overview_image)
        self.label_file_name = self.identifier +\
            '_' + self.image_property.get_staining() +\
            '_' + self.image_property.get_scanner() +\
            '_' + str(self.image_property.get_magnification()) +\
            'x-labels.png'
        if os.path.exists(self.setting.get_data_setting().get_label_map_folder()+self.label_file_name):
            from PIL import Image
            self.label_map = Image.open(self.setting.get_data_setting().get_label_map_folder()+self.label_file_name)
        else:
            # Default map is complete WSI
            self.label_map = np.ones(np.shape(self.overview_threshold))
        # Check wheter tissue outside marked area should be filtered
        if self.setting.get_data_setting().get_use_only_stamp():
            
            # Filter tissue according to label map
            self.overview_threshold = self.overview_threshold * self.label_map




        # Size of WSI at used level
        self.size = self.image.level_dimensions[0]

        # Load Tiles for WSI
        self.tiles = []
        # Close WSI    
        self.close_wsi()

        # Create single feature vector
        feature_dim = self.setting.get_network_setting().get_F()
        self.features = np.zeros((0, feature_dim))
        # Keys for each feature vector - Position to feature for Map
        self.keys = []

        # Create Folder for features
        self.feature_folder = self.setting.get_data_setting().get_feature_folder() +\
            self.image_property.get_staining() + '_' +\
            str(self.image_property.get_magnification()) + '_' +\
            str(self.image_property.get_used_magnification()) + '/'
    
        helper.create_folder(self.feature_folder)

        # Generate Feature file name per tile property
        self.feature_file_names = []
        for tile_property in self.tile_properties:
            # Same naming scheme as JSON
            feature_file_name = self.feature_folder + str(self.identifier) +\
                '_' + str(tile_property.get_tile_size()[0]) +\
                '_' + str(tile_property.get_tile_size()[1]) +\
                '_' + str(tile_property.get_tile_overlap()[0]) +\
                '_' + str(tile_property.get_tile_overlap()[1]) + '.pkl'
            
            self.feature_file_names.append(feature_file_name)
        self.tiles_inside = []
        self.tiles_outside = []

        #MD
        #list used for solid selection during training
        self.solid_selection_list = []*len(self.tile_properties)
        print(len(self.solid_selection_list))

    #MD
    def get_solid_selection_list(self, tile_property_index):
        """Returns the list for solid selection"""
        try:
            return self.solid_selection_list[tile_property_index]
        except IndexError:
            return []

    #MD
    def set_solid_selection_list(self, solid_list, tile_property_index):
        """set the solid list to the tile property"""
        self.solid_selection_list.append(solid_list)

    def load_wsi(self):
        """ Opens OpenSlideObject
        """
        # Open Openslide WSI
        self.image = reader.read_wsi(self.file_path)
    
    def close_wsi(self):
        """ Closes OpenSlideObject
        """
        # Close Openslide WSI
        if not (self.image is None):
            self.image.close()

        self.image = None

    def load_tiles(self):
        """ Creates Tile Objects for WSI according to Tile properties and chosen filter settings. 
        Saves the Tiling to a JSON File containing, x and y positions of valid Tiles.

        Returns 
        -------
        list [[Tile]]
            A list containing a list per TileProperty containing all Tile Objects created 
        """
        self.load_wsi()
        # List of list of tiles
        tiles = []
        
        # Create tiles for each tile property
        for tile_property in self.tile_properties:

            # JSON File Name for specific property
            json_file_name = self.json_file_name_scheme +\
                '_' + str(tile_property.get_tile_size()[0]) +\
                '_' + str(tile_property.get_tile_size()[1]) +\
                '_' + str(tile_property.get_tile_overlap()[0]) +\
                '_' + str(tile_property.get_tile_overlap()[1]) +\
                '.json'

            # Check if JSON already existing and should skip already tiled WSIs
            if os.path.exists(json_file_name) and self.setting.get_data_setting().get_skip_existing():
                # Append Tiles loaded from JSON
                tiles.append(self.load_tiles_json(json_file_name, tile_property))
                # No Tiling necessary for property
                continue

            # List of tiles for property
            tiles_property = []

            # Get step size accoring to overlap
            step_size_x = (1+self.used_level)*(tile_property.get_tile_size()[0] - tile_property.get_tile_overlap()[0])
            step_size_y = (1+self.used_level)*(tile_property.get_tile_size()[1] - tile_property.get_tile_overlap()[1])

            # Get possible positions which are not already background according to overview image and minimum tissue percentage
            positions = [(sx, sy) for sx in range(0, self.size[0], step_size_x)\
                for sy in range(0, self.size[1], step_size_y)\
                if filter.check_tissue(self.overview_threshold,(sx, sy), tile_property.get_tile_size(), self.downsampling_factor, self.factor, self.setting.get_data_setting().get_min_tissue_percentage())]
            
            # Progress bar
            bar = IncrementalBar('Loading Tiles:', max=len(positions))
            images = []
            # Load image tiles
            for position in positions:
                # Image loading only necessary if want to filter each tile individually again
                if self.setting.get_data_setting().get_filter_background():
                    # Read image tile at position
                    image = reader.read_region(self.image, corner=position, size=tile_property.get_tile_size(), level_downsamples=self.used_level)
                else:
                    # Else no image needed, this speeds up
                    image = None
                # Append image and its position
                images.append((image, position))
                bar.next()

            bar.finish()

            # Create Pool for multiprocessing of filtering
            pool = multiprocessing.Pool()
            # Create multiprocessing filtering function, otsu value according to overview
            partial_f = partial(multiprocess_filtering, otsu_value=self.otsu_value,
                filter_background=self.setting.get_data_setting().get_filter_background(),
                filter_blood=self.setting.get_data_setting().get_filter_blood(),
                tissue_percentage=self.setting.get_data_setting().get_min_tissue_percentage(),
                tile_property=tile_property,
                level = self.used_level
            )
            # Execute multiprocessing
            resulting_tiles = pool.map(partial_f, images)
            # Save positions of Tiles for JSON
            indices = []

            # Iterate tiles from multiprocessing
            for i in range(len(resulting_tiles)):
                # If valid Tile
                if resulting_tiles[i] != None:
                    # Append to list
                    tiles_property.append(resulting_tiles[i])
                    # Append to positions to save in JSON
                    indices.append({"x":resulting_tiles[i].get_position()[0],\
                         "y":resulting_tiles[i].get_position()[1]})

            # Write JSON
            json_obj = json.dumps({"tiles":indices})

            with open(json_file_name, 'w') as json_file:
                json.dump(json_obj, json_file)

            # Append tiles for property
            tiles.append(tiles_property)

        self.close_wsi()
        return tiles

    def load_tiles_json(self, json_file_name, tile_property):
        """ Loads a list of Tiles from the positions stored in a JSON file

        Parameters
        ----------
        json_file_name : string
            Path to the JSON file where Tiling is stored
        tile_property : TileProperty
            TileProperty for which loaded Tiles should be created

        Returns
        -------
        list [Tile]
            A list containing all Tiles stored in the JSON file
        """
        # List of tiles for property
        tiles = []
        # Open JSON file
        with open(json_file_name) as json_file:
            json_obj = json.load(json_file)
            # Get valid positions
            indices = json.loads(json_obj)["tiles"]
            # Iterate valid positions
            for i in range(0, len(indices)):
                x = indices[i]["x"]
                y = indices[i]["y"]
                # Create tile object
                tile_obj = Tile(x, y, tile_property.get_tile_size(), 
                    tile_property.get_input_size(), tile_property.get_augmentations(), self.used_level)
                # Append to tiles for property
                tiles.append(tile_obj)

        return tiles

    def get_identifier(self):
        """ Getter WSI identifier attribute

        Returns
        -------
        string
            The WSIs identifier
        """
        return self.identifier

    def get_block(self):
        """ Getter WSI block attribute

        Returns
        -------
        string
            The WSIS block
        """
        return self.block

    def get_subblock(self):
        """ Getter WSI subblock attribute

        Returns
        -------
        string
            The WSIS subblock
        """
        return self.subblock

    def get_cut(self):
        """ Getter WSI cut attribute

        Returns
        -------
        int
            The WSIs cut
        """
        return self.cut

    def get_z_level(self):
        """ Getter WSI z_level attribute

        Returns
        -------
        int
            The WSIs z-level
        """
        return self.z_level

    def get_image_property(self):
        """ Returns the image property of the WSI

        Returns
        -------
        ImageProperty
            The image property of the WSI
        """
        return self.image_property

    def get_tile_properties(self):
        """ Returns the Tile properties of the WSI

        Returns
        -------
        list [TileProperty]
            The tile properties for this WSI
        """
        return self.tile_properties

    def get_data_folder(self):
        """ Returns the data folder where the WSI is stored

        Returns
        -------
        string
            The WSIs data folder
        """
        return self.data_folder

    def get_file_name(self):
        """ Returns the file name of the WSI

        Returns
        -------
        string
            The WSIs file name 
        """
        return self.file_name

    def get_file_path(self):
        """ Returns the complete file path of the WSI

        Returns
        -------
        string
            The WSIs file path
        """
        return self.file_path

    def get_image(self):
        """ Return the OpenSlide Object belonging to the WSI

        Returns
        -------
        OpenSlideObject
            The WSIs OpenSlide Object
        """
        return self.image
    
    def get_overview_image(self):
        """ Return the highest level of the image pyramid

        Returns
        -------
        Image
            The overview image of the WSI
        """
        return self.overview_image

    def set_tiles(self, tiles):
        """ Set list of tiles

        Parameters
        ----------
        tiles : list [[Tile]]
            List containing a list per TileProperty containing Tiles 
        """
        self.tiles = tiles
    
    def get_tiles_list(self):
        """ Return the tile attribute

        Returns
        -------
        list [[Tile]]
            A list containing a list per TileProperty containing the Tiles belonging to the WSI
        """
        return self.tiles
    
    def set_features(self, features, tile_property_index):
        """ Set encoded features for a TileProperty and saves them to a pickle file

        Parameters
        ----------
        features : dict {(int, int) : numpy array}
            Encoded features for TileProperty, keys of dicitionary are positions of Tiles
        tile_property_index : int
            Index of the TileProperty
        """
        # Open correct feature file
        print(self.feature_file_names)
        with open(self.feature_file_names[tile_property_index], 'wb') as f:
            # Write features to it
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

    def get_feature_file_names(self):
        """ Return list of all feature file names for this WSI

        Returns
        -------
        list [string]
            List of all feature file names
        """
        return self.feature_file_names
  
    def get_features(self, tile_property_index):
        """ Return the features stored in a file for this TileProperty

        Parameters
        ----------
        tile_property_index : int
            Index of the TileProperty

        Returns
        -------
        dict {(int, int) : numpy array} 
            Encoded features for TileProperty, keys of dictionary are positions of Tiles
        """
        # Open correct feature file
        with open(self.feature_file_names[tile_property_index], 'rb') as f:
            return pickle.load(f)

    def get_all_features(self):
        """ Returns all features for all TileProperties of the WSI

        Returns
        -------
        numpy array
            Concatenated features - one TileProperty after another
        list [(int, int)]
            Concatenated keys of features, keys are positons of Tiles
        """
        # If features already loaded return
        if len(self.features) != 0:
            return self.features, self.keys

        # Iterate all Feature vectors - one per tile property
        for i, feature_file_name in enumerate(self.feature_file_names):
            # Get feature vector
            feature_dict = self.get_features(i)
            feature = np.array(list(feature_dict.values()))
            # Check for non-existing feature vector
            if len(feature) == 0:
                feature = np.zeros((0, 1024))
            
            # Concatenate
            self.features = np.concatenate((self.features, feature), axis=0)
            # Append keys to list
            key = list(feature_dict.keys())
            if key != None:
                self.keys += key

        return self.features, self.keys

    def set_map(self, A, keys, A_relative):
        """ Set attention map of WSI and store it to a json.

        Parameters
        ----------
        A : numpy array
            Numpy array with one value per tile of all WSIs
        keys : list
            List of tuples with a key per tile in the same order as A
        A_relative : numpy array
            Numpy array with one value per tile of all WSIs, relative rank was computed over all WSIs of patient
        """
        counter_tiles = 0
        # Iterate tile properties
        for i, tile_property in enumerate(self.tile_properties):
            # Get feature dictionary for this property
            feature_dict = self.get_features(i)
            # Get number of tiles for property
            n_tiles = len(list(feature_dict.values()))
            
            # Get keys for tile property
            tiles_keys = keys[counter_tiles:counter_tiles+n_tiles]
            # Get Attention_map for tile property
            tiles_A = A[counter_tiles:counter_tiles+n_tiles]
            # Get relative Attention map for tile property
            A_relative_patient = A_relative[counter_tiles:counter_tiles+n_tiles]

            # Compute relative attention map for tile property
            tiles_A_relative = rankdata(tiles_A, "dense")
            max_rank = np.max(np.array(tiles_A_relative))
            tiles_A_relative = np.array(tiles_A_relative)/ max_rank
            # Increment tile counter
            counter_tiles += n_tiles

            attention_file_name = self.attention_file_name_scheme +\
                '_' + str(tile_property.get_tile_size()[0]) +\
                '_' + str(tile_property.get_tile_size()[1]) +\
                '_' + str(tile_property.get_tile_overlap()[0]) +\
                '_' + str(tile_property.get_tile_overlap()[1]) +\
                '.json'

            attention_objects = []

            tiles_dict = {t.get_position() : t for t in self.tiles[i]}

            for j in range(n_tiles):
                r, g, b, value = self.value_to_rgb(A_relative_patient[j])

                r_relative, g_relative, b_relative, value_relative = self.value_to_rgb(tiles_A_relative[j])

                attention_objects.append({"x":int(tiles_keys[j][0]),\
                    "y":int(tiles_keys[j][1]),\
                    "R":r, "G":g, "B":b,\
                    "class":value,\
                    "R_relative":r_relative, "G_relative":g_relative, "B_relative":b_relative,\
                    "class_relative":value_relative\
                    })

                current_tile = tiles_dict[(tiles_keys[j][0], tiles_keys[j][1])]
                
                current_tile.add_attention_value(value)
                current_tile.add_attention_value_relative(value_relative)

            
            json_obj = json.dumps({"tiles":attention_objects})

            with open(attention_file_name, 'w') as json_file:
                json.dump(json_obj, json_file)

    def value_to_rgb(self, value):
        """ Computes RGB for a value with colormap 'coolwarm'

        Parameters
        ----------
        value : float
            Rank of value inside attention map normed to [0...1]

        Returns
        -------
        int 
            Red value
        int
            Green value
        int
            Blue value
        float
            input value
        """
        cmap = plt.get_cmap('coolwarm')

        color_value = cmap(value)
        r = int(color_value[0]*255)
        g = int(color_value[1]*255)
        b = int(color_value[2]*255)

        return r, g, b, value

    def save_map(self, folder):
        """ Save attention maps for statistics to numpy arrays. Relative and normal attention maps will be stored. For each all attention values and those inside/outside will
        be stored.

        Parameters
        ----------
        folder : string
            Path were attention values should be stored
        """
        self.set_inside_outside()

        for i, tp in enumerate(self.tile_properties):
            attention_values = np.array(self.get_attention_values(self.tiles[i], nested=True))
            attention_values_inside = np.array(self.get_attention_values(self.tiles_inside[i], nested=True))
            attention_values_outside = np.array(self.get_attention_values(self.tiles_outside[i], nested=True))

            attention_values_relative = np.array(self.get_attention_values_relative(self.tiles[i], nested=True))
            attention_values_relative_inside = np.array(self.get_attention_values_relative(self.tiles_inside[i], nested=True))
            attention_values_relative_outside = np.array(self.get_attention_values_relative(self.tiles_outside[i], nested=True))

            subfolder = folder + str(tp.get_tile_size()[0]) + '_' + str(tp.get_tile_size()[1]) + '_' + str(tp.get_tile_overlap()[0]) + '_' + str(tp.get_tile_overlap()[1]) + '/'
            helper.create_folder(subfolder)

            save_folders = [subfolder + a for a in ['attention/', 'attention_inside/', 'attention_outside/', 'attention_relative/', 'attention_relative_inside/', 'attention_relative_outside/']]
            save_values = [attention_values, attention_values_inside, attention_values_outside, attention_values_relative, attention_values_relative_inside, attention_values_relative_outside]

            for save_folder, save_value in zip(save_folders, save_values):
                helper.create_folder(save_folder)

                np.save(save_folder+self.identifier+'.npy', save_value)

    def get_attention_values(self, tile_list, nested=True):
        """ Returns all attention values stored for Tiles

        Parameters
        ----------
        tile_list : list [Tile]
            List of tiles for which attention values should be returned
        nested : bool
            True if list should be nested for each Tile

        Returns
        -------
        nested = True list [float] 
        nested = False list [[float]]
            list of attention values nested if True
        """
        attention_values = []


        for tile in tile_list:
            if nested:
                attention_values.append(tile.get_attention_values())
            else:
                attention_values += tile.get_attention_values()

        return attention_values
           
    def get_attention_values_relative(self, tile_list, nested=True):
        """ Returns all relative attention values stored for Tiles

        Parameters
        ----------
        tile_list : list [Tile]
            List of tiles for which relative attention values should be returned
        nested : bool
            True if list should be nested for each Tile

        Returns
        -------
        nested = True list [float] 
        nested = False list [[float]]
            list of attention values nested if True
        """
        attention_values = []


        for tile in tile_list:
            if nested:
                attention_values.append(tile.get_attention_values_relative())
            else:
                attention_values += tile.get_attention_values_relative()

        return attention_values

    def set_inside_outside(self):
        """ Computes for all Tiles if they are inside or outside marked area according to label map and sets those attributes.
        """
        outside_lists = []
        inside_lists = []

        for i, tile_property in enumerate(self.tile_properties):
            
            step_size_x = (1+self.used_level)*(tile_property.get_tile_size()[0] - tile_property.get_tile_overlap()[0])
            step_size_y = (1+self.used_level)*(tile_property.get_tile_size()[1] - tile_property.get_tile_overlap()[1])
            positions = [(sx, sy) for sx in range(0, self.size[0], step_size_x)\
                for sy in range(0, self.size[1], step_size_y)\
                if filter.check_tissue(self.overview_threshold*self.label_map, (sx, sy), tile_property.get_tile_size(), self.downsampling_factor, self.factor, self.setting.get_data_setting().get_min_tissue_percentage())]

            outside_list = []
            inside_list = []

            for tile in self.tiles[i]:
                if tile.get_position() in positions:
                    
                    inside_list.append(tile)
                else:

                    outside_list.append(tile)
            print("dkfdksf")
            print(len(self.tiles[i]))
            print(len(inside_list))
            print(len(outside_list))
            outside_lists.append(outside_list)
            inside_lists.append(inside_list)

       
        self.tiles_inside = inside_lists
        self.tiles_outside = outside_lists

    #MD?
    def get_inside(self):

        """returns all tiles inside the marked area

        
        Returns
        -------
        [Tile]
        """
        return self.tiles_inside
    #MD?
    def get_outside(self):
        """returns all tiles outside the marked area

        Returns
        -------
        [Tile]
        """
        return self.tiles_outside