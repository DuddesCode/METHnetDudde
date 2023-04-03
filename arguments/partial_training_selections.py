"""contains functions and a class that holds the selected settings for a train run"""

import random
from abc import ABC, abstractmethod

class Selection(ABC):
    
    def __init__(self, setting, wsi) -> None:
        self.marked_batches = 3
        self.batch_size = setting.get_batch_size()
        self.number_of_tiles = self.marked_batches * setting.get_batch_size()
        self.wsi = wsi

    @abstractmethod
    def tile_marking(self):
        pass

    def get_marked_batches(self):
        return self.marked_batches
    
    def tile_resetting(self):
        for tile_list in self.wsi.get_tiles_list():
            for tile in tile_list:
                tile.set_mark(False)

class RandomSelection(Selection):

    def __init__(self, setting, wsi) -> None:
        super().__init__(setting, wsi)

    def tile_marking(self):
        """changes the marked attribute of wsi Tiles of the given wsi until the number of batches is satisfied"""

        for tile_list in self.wsi.get_tiles_list():
            randomised_tile_list = random.sample(tile_list, self.number_of_tiles)
            for tile in randomised_tile_list:
                tile.set_mark(True)
        return randomised_tile_list  

class SolidSelection(Selection):

    def __init__(self, wsi, setting) -> None:
        super().__init__(setting, wsi)

    def tile_marking(self):
        """changes the first self.number_of_tiles of a wsi to marked"""
        for tilelist in self.wsi.get_tiles_list():
            for i, tile in enumerate(tilelist):
                if i is not self.number_of_tiles:
                    tile.set_mark(True)

class HandPickedSelection(Selection):
    
    def __init__(self, wsi, setting) -> None:
        super().__init__(setting, wsi)
        self.wsi.set_inside_outside()

    def tile_marking(self):
        max_num_marked = (len(self.wsi.tiles_inside[0]) - (len(self.wsi.tiles_inside[0]) % self.batch_size)) / self.batch_size
        max_num_marked = int(max_num_marked)
        if max_num_marked < self.marked_batches:
            self.marked_batches = max_num_marked
        for i in range(self.marked_batches * self.batch_size):
            self.wsi.tiles_inside[0][i].set_mark(True)

class AttentionSelection(Selection):

    def __init__(self, wsi, setting, epoch) -> None:
        super().__init__(setting, wsi)
        self.epoch = epoch
    
    def tile_marking(self):
        if self.epoch is not 0:
            print('Banane')
            #iterate tileprops
            for tile_list in self.wsi.get_tiles_list():
                temp_tile_list = sorted(tile_list, key= lambda tile: tile.get_attention_values()[self.epoch - 1])
                for i in range(self.marked_batches * self.batch_size):
                    temp_tile_list[i].set_mark(True)
        else:
            print('Doener')
            #iterate tileprops
            for tile_list in self.wsi.get_tiles_list():
                randomised_tile_list = random.sample(tile_list, self.number_of_tiles)
                for tile in randomised_tile_list:
                    tile.set_mark(True)
