import os

def create_folder(folder):
        
    subfolders = folder.split('/')
    current_path = ''

    for sf in subfolders:
        current_path += sf + '/'

        if not os.path.exists(current_path):
            os.mkdir(current_path)

"""contains functions used throughout the module"""
"""Function from the pytorch forum used to check if models have the same parameter values"""
import torch

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                #,'difference: ',key_item_1[1]-key_item_2[1]
                print('Mismatch found at','model 1: ',key_item_1[0],' model 2: ',key_item_2[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')