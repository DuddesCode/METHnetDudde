"""contains functions and a class that holds the selected settings for a train run"""

import random
import numpy as np
from abc import ABC, abstractmethod

class Selection(ABC):
    """Base class for selectiing tiles for end to end training

    Parameters
    ----------
    ABC : ABC
        absract base class

    Attributes
    ----------
    marked_batches : int
        contains the number of batches that have to be marked
    initial_batches : int
        contains the number of batches that were initially designated to be marked
    batch_size : int
        contains the batch size of the training run
    number_of_tiles : int
        contains the number of tiles to be marked
    wsi : WSI
        contains the WSI object of the current wsi to train
    
    Functions
    -------
    tile_marking()
        abstract method of how to mark tiles
    tile_resetting()
        resets all tiles to non marked
    get_marked_batches()
        returns the number of batches that have to be marked
    get_number_marked()
        returns the number of tiles that have to be marked
    get_batch_size()
        returns the batch size of the training run
    set_batch_size(batch_size: int)
        sets the batch size of the training run
    set_num_tiles(num: int)
        sets the number of tiles to be marked
    set_marked_batches(num_marked: int)
        sets the number of batches that have to be marked
    """    
    def __init__(self, setting, wsi, json_path=None) -> None:
        self.marked_batches = 3 
        self.batch_size = setting.get_batch_size()
        self.number_of_tiles = self.marked_batches * setting.get_batch_size()
        self.wsi = wsi
        if json_path is not None:
            import json
            with open(json_path+'setup.json', 'r') as j_file:
                setup_file = json.load(j_file)
                self.marked_batches = setup_file['num_batches']
        self.number_of_tiles = self.marked_batches * setting.get_batch_size()
        self.initial_batches = self.marked_batches

    @abstractmethod
    def tile_marking(self, i):
        """abstract method to decide how the tiles have to be marked
        """
        pass

    def get_marked_batches(self):
        """returns the number of batches that have to be marked

        Returns
        -------
        int: number of marked batches
        """        
        return self.marked_batches
    
    def tile_resetting(self):
        """resets all tiles to non marked
        """        
        for tile_list in self.wsi.get_tiles_list():
            for tile in tile_list:
                tile.set_mark(False)
    
    def get_number_marked(self):
        """returns the number of tiles that have to be marked

        Returns
        -------
        int: number of tiles
        """        
        return self.number_of_tiles
    
    def set_marked_batches(self, num_marked):
        """sets the number of batches that have to be marked

        Parameters
        ----------
        num_marked : int
            contains the number of batches that have to be marked
        """        
        self.marked_batches = num_marked
    
    def set_num_tiles(self, num):
        """sets the number of tiles to be marked

        Parameters
        ----------
        num : number of tiles to be marked
        """        
        self.number_of_tiles = num
    
    def get_batch_size(self):
        """returns the batch size

        Returns
        -------
        int: batch size
            _description_
        """        
        return self.batch_size

    def tile_number_check(self, tile_list):
        """checks if the number of tiles to be marked is not less than the length of param:tilelist

        If it is less then the function fit_to_tile_num is called

        :param tile_list: contains Tile objects
        :type tile_list: list
        """
        
        if not self.number_of_tiles < len(tile_list):
            self.fit_to_tile_num(len(tile_list))

    def batchsize_check(self):
        if len(self.wsi.get_tiles_list()[0])< self.batch_size:
            self.marked_batches = 1
            return False
        else:
            return True

    def fit_to_tile_num(self, tiles):
        """fits the number of marked_batches to the number of tiles in param tiles

        :param tiles: contains how many tiles exist in a list 
        :type tiles: int
        """    
        print('came to fit_to_tile_num')
        print(tiles)
        print(self.batch_size)
        possible_batches = tiles // self.batch_size
        print(possible_batches)
        self.set_marked_batches(possible_batches)
        self.set_num_tiles(possible_batches * self.batch_size)

    def reset_to_initial_batch_count(self):
        """Resets the number of marked batches and the number of tiles
        """    
        self.set_marked_batches(num_marked=self.initial_batches)
        self.set_num_tiles(self.get_marked_batches() * self.get_batch_size())

class RandomSelection(Selection):
    """Selects the tiles randomly

    Parameters
    ----------
    Selection : Selection
        base class for all selection subtypes

    Functions
    -------
    tile_marking()
        overwrites the base class function tile_marking()
    """
    def __init__(self, setting, wsi, json_path=None) -> None:
        super().__init__(setting, wsi, json_path=json_path)

    def tile_marking(self, i):
        """changes the marked attribute of wsi Tiles of the given wsi until the number of batches is satisfied"""

        print('worked')
        tile_list = self.wsi.get_tiles_list()[i]
        self.tile_number_check(tile_list)
        indices_array = np.arange(0, len(tile_list)-1, 1, dtype=int)
        print(indices_array)
        rng = np.random.default_rng()
        rng.shuffle(indices_array)
        print(indices_array)
        randomised_tile_list = indices_array[0:self.number_of_tiles-1:1]
        self.reset_to_initial_batch_count()
        if len(tile_list) < self.batch_size:
            self.set_marked_batches(0)
            return None
        return randomised_tile_list  

class SolidSelection(Selection):
    """Selects the tiles based on the number of tiles to mark. Fixed Random.

    Parameters
    ----------
    Selection : Selection
        base class for all selection subtypes

    Functions
    -------
    tile_marking()
        overwrites the base class function tile_marking()
    """    
    def __init__(self, wsi, setting, epoch, json_path=None) -> None:
        super().__init__(setting, wsi, json_path=json_path)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch
    
    def tile_marking(self, i):
        """changes the first self.number_of_tiles of a wsi to marked"""
        #iterate over tilelists
        print(self.get_epoch())
        rng = np.random.default_rng()
        

        if self.wsi.get_solid_selection_list(0) == []:
            print('worked')
            
            tile_list = self.wsi.get_tiles_list()[i]
            if len(tile_list) < self.batch_size:
                self.set_marked_batches(1)
                return None
            self.tile_number_check(tile_list)
            indices_array = np.arange(0, len(tile_list)-1, 1, dtype=int)
            print(indices_array)
            rng = np.random.default_rng()
            rng.shuffle(indices_array)
            print(indices_array)
            final_indices = indices_array[0:self.number_of_tiles-1:1]
            self.wsi.set_solid_selection_list(final_indices, i)
            #self.reset_to_initial_batch_count()
        
        else:
            final_indices = self.wsi.get_solid_selection_list(i)

        rng.shuffle(final_indices)

        return final_indices

class HandPickedSelection(Selection):
    """Marks tiles that are within an area marked by a specialist

    Parameters
    ----------
    Selection : Selection
        Base class for all selection subtypes
    
    Functions
    -------
    tile_marking()
        overwrites the base class function tile_marking()
    """
    def __init__(self, wsi, setting, json_path=None) -> None:
        super().__init__(setting, wsi, json_path=json_path)
        self.wsi.set_inside_outside()


    def tile_marking(self, idx):
        """overwrites the abstract method of the base class

        Only tiles are selected that are within an area marked by a specialist.
        If not enough tiles lie within the marked area to fill the requested batch size,
        then it is filled up by other tiles outside the marked region.

        """ 
        #check if enough tiles in the marked region to satisfy the requested batch size
        tile_list_final = []
        rng = np.random.default_rng()
        if 0 <= (len(self.wsi.get_inside()[0]) - self.number_of_tiles):
            for i in range(self.marked_batches * self.batch_size):
                tile_list_final.append(self.wsi.tiles_inside[0][i])
        else:
            self.tile_number_check(self.wsi.tiles_inside[0])
            for i in range(self.marked_batches * self.batch_size):               
                tile_list_final.append(self.wsi.tiles_inside[0][i])
            if len(self.wsi.tiles_inside[0]) < self.batch_size:
                self.set_marked_batches(0)
                return None    
            """tile_list = self.wsi.tiles_inside[0]
            outside_list = self.wsi.tiles_outside[0]
            i = 0
            #check if enough tiles overall to satisfy the requested batch size
            if len(self.wsi.get_tiles_list()[idx]) < self.number_of_tiles:
                print('jbsjkablsknvkjafbvuwbrg')
                print(self.number_of_tiles)
                self.tile_number_check(self.wsi.get_tiles_list()[idx])
                print(self.number_of_tiles)
            #add tiles outside the marked region to all inside tiles
            while len(tile_list) != self.number_of_tiles:
                tile_list.append(outside_list[i])
                i += 1
            tile_list_final = tile_list"""
        #get numpy index list of overall tiles
        wsi_tile_list = self.wsi.get_tiles_list()[idx]
        wsi_tile_list = np.array(wsi_tile_list)
        tile_list_final = np.array(tile_list_final)
        index_list = []
        for ele in tile_list_final:
            index_list.append(np.where(wsi_tile_list == ele)[0][0])
        #sort tile list to get indexes
        index_list = np.array(index_list)
        rng.shuffle(index_list)
        return index_list
            
            

class AttentionSelection(Selection):
    """Marks tiles by assigned attention values
    
    Parameters
    ----------
    Selection : Selection
        Base class for all selection subtypes
    
    Attributes
    ------
    epoch : int
        contains the current epoch
    
    Functions
    -------
    tile_marking()
        Overwrites the base class function tile_marking()
    get_epoch()
        return the current epoch
    """      
    def __init__(self, wsi, setting, epoch, json_path=None) -> None:
        super().__init__(setting, wsi, json_path=json_path)
        self.epoch = epoch
    
    def get_epoch(self):
        return self.epoch
    
    def tile_marking(self, i):
        """overwrites the abstract method of the base class

        In the first epoch the tiles are marked at random
        After the first epoch the tiles are marked according to the curretn attention map of the wsi
        """    
        rng = np.random.default_rng()            
        if self.get_epoch() != 0:
            #iterate over tilelists
            tile_list = self.wsi.get_tiles_list()[i]
            if len(tile_list) < self.batch_size:
                self.set_marked_batches(0)
                return None
            temp_tile_list = sorted(tile_list, key= lambda tile: tile.get_attention_values()[self.epoch - 1])
            #check if number of tiles are fitted to tile list
            self.tile_number_check(temp_tile_list)
            #get all tiles from the list
            temp_tile_list = np.array(temp_tile_list)
            tile_list = np.array(tile_list)
            temp_tile_list = temp_tile_list[0:((self.marked_batches*self.number_of_tiles)-1)]
            index_tile_list = []
            for ele in temp_tile_list:
                index_tile_list.append(np.where(tile_list == ele)[0][0])
            index_tile_list = np.array(index_tile_list)
            rng.shuffle(index_tile_list)
            self.reset_to_initial_batch_count()
        else:
            #iterate tilelists of a wsi
            print('worked')
            tile_list = self.wsi.get_tiles_list()[i]
            if len(tile_list) < self.batch_size:
                self.set_marked_batches(0)
                return None
            self.tile_number_check(tile_list)
            indices_array = np.arange(0, len(tile_list)-1, 1, dtype=int)
            print(indices_array)
            rng.shuffle(indices_array)
            print(indices_array)
            index_tile_list = indices_array[0:self.number_of_tiles-1:1]
            self.reset_to_initial_batch_count()

        return index_tile_list


