import csv
import enum
import string
import arguments.setting as setting
import datastructure.dataset as dataset
import os

from learning.partial_training_main_funct import partial_training, save_it_losses
from learning.testing_partial import test_partial
from learning.testing_parallelism import partial_training_parallel
from progress.bar import IncrementalBar
import learning.train
import learning.test
import getopt, sys

import numpy as np

from matplotlib import pyplot as plt
from timeit import default_timer as timer

def convert_results_to_csv(results):
    import csv

    header = ['Mode', 'num_batches', 'sensitivity', 'accuracy', 'specificity', 'iteration']
    with open('./results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with open('./results.csv', 'a', newline='') as f:
        
        writer = csv.writer(f)
        
        writer.writerows(results)

def dataset_saver(patients_test, patients_val, patients_train):
    complete_list = patients_test + patients_val + patients_train

    wsi_tile_dict = {}
    tile_count = 0
    for patients in complete_list:
        for p in patients:
            for wp in p.get_wsis():
                for wsi in wp:
                    for tilelist in wsi.get_tiles_list():
                        tile_count += len(tilelist)
                        wsi_tile_dict[wsi.get_identifier()] = len(tilelist)
    
    print(tile_count)
    wsi_tile_dict = dict(sorted(wsi_tile_dict.items(), key=lambda x: x[1]))
    last_element = list(wsi_tile_dict)[-1]
    print(last_element)
    last_value = wsi_tile_dict[last_element]
    print(last_value)
    print('max batches')
    print(last_value/128)
    logscale = np.logspace(2.0, 212.0, num=7, dtype=int)
    print(logscale)
    plt.bar(*zip(*wsi_tile_dict.items()))
    plt.show()
    

def run(data, setting, selection_mode, train=False, features_only=False, runs_start=0, runs=10, draw_map=False, json_path=None):
    """ Run model training and testing

    Parameters
    ----------
    data : Dataset
        Dataset to use for model training/testing
    setting : Setting
        Setting as specified by class
    train : bool
        True if want to train models
    features_only : bool
        True if only want to encode features
    runs_stat : int
        First run of Monte-Carlo cross-validation to run
    runs : int
        Last run of Monte-Carlo cross-validation to run
    draw_map : bool
        True if want to save attention maps
    """
    if runs_start >= runs:
        return 
    import numpy as np


    if features_only:
        return

    bar = IncrementalBar('Running Monte Carlo ', max=runs)

    balanced_accuracies = []
    sensitivities = []
    specificities = []
    runtimes = []
    # Iterate Monte-carlo
    for k in range(runs_start, runs):
        runtimer_start = timer()
        # Set split
        data.set_fold(k)

        #dataset_saver(data.get_test_set(), data.get_validation_set(), data.get_train_set())
        # Train model
        #if train:
        marked_batches = partial_training(train_patients = data.get_train_set(), validation_patients=data.get_validation_set(), setting = setting, fold = k, selection_mode = selection_mode, draw_map=draw_map, json_path = json_path)
        # Test model
        balanced_accuracy, sensitivity, specificity = test_partial(data.get_test_set(), k, setting, draw_map=draw_map, json_path=json_path)
        runtimer_stop = timer()
        runtimes.append(runtimer_stop-runtimer_start)
        print('SPECIFIC post test partial')
        print(specificity)
        balanced_accuracies.append(balanced_accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    runtimes = np.array(runtimes, dtype=np.float32)
    save_it_losses(runtimes, json_path, 'runtime')

    # Save results
    for patients in data.get_test_set():
        for p in patients:
            results_folder = setting.get_data_setting().get_results_folder()
            label = p.get_diagnosis().get_label()
            p.save_predicted_scores(f"{results_folder}_{label}")
            if draw_map:
                p.save_map()
            


def run_train(data_directories, csv_file, working_directory, selection_mode, draw_map, json_path=None):
    """ Set up setting and dataset and run training/testing
    MD added the json_path parameter for settings
    MD added draw_map parameter
    """
    s = setting.Setting(data_directories, csv_file, working_directory, json_path)

    data = dataset.Dataset(s)
    
    run(data, s, selection_mode=selection_mode, train=True, features_only=False, runs_start=0,runs=s.get_network_setting().get_runs(), draw_map=draw_map, json_path=json_path)

def run_train_eval_mode(selection_mode, draw_map,  setting, data, json_path=None):
    """script to use for training and evaluation"""

    run(data, setting, selection_mode=selection_mode, train=True, features_only=False, runs_start=0,runs=setting.get_network_setting().get_runs(), draw_map=draw_map, json_path=json_path)

def main(argv):
    try:
        #MD
        opts, args = getopt.getopt(argv, "hd:c:w:m:", ["data_directory=","csv_file=","working_directory=","selection_mode="])
    except getopt.GetoptError:
        print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -m <selection_mode>')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    if not('-d' in opts_vals or '--data_directory' in opts_vals):
        print('Specify -d or --data_directory')
        sys.exit(2)
    if not('-c' in opts_vals or '--csv_file' in opts_vals):
        print('Specify -c or --csv_file')
        sys.exit(2)
    if not('-w' in opts_vals or '--working_directory' in opts_vals):
        print('Specify -w or --working_directory')
        sys.exit(2)
        #MD
    if not('-m' in opts_vals or '--selection_mode' in opts_vals):
        print('Specify -m or wrong selection_mode:[random, hand_picked, solid, attention]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
             print('main.py -d <data_directory> -c <csv_file> -w <working_directory>')
        elif opt in ('-d', '--data_directory'):
            data_directory = arg.strip('[]').split(',')
        elif opt in ('-c', '--csv_file'):
            if type(arg) == str and arg.endswith('.csv'):
                csv_file = arg
            else:
                print("Wrong data type for -c or --csv_file should be path to .csv")
                sys.exit(2)
        elif opt in ('-w', '--working_directiory'):
            if type(arg) == str:
                working_directory = arg
            else:
                print("Wrong data type for -w or --working_directory should be string")
                sys.exit(2)
        elif opt in ('-m', '--selection_mode'):
            if type(arg) == str:
                selection_mode = arg
                if selection_mode not in ['random', 'hand_picked','solid', 'attention']:
                    print("Wrong data type for -m or wrong selection_mode:[random, hand_picked, solid, attention]")
                    sys.exit(2)
    draw_map = False
    run_train(data_directory, csv_file, working_directory, draw_map, selection_mode)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])