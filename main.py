import csv
import enum
import string
import arguments.setting as setting
import datastructure.dataset as dataset
import os

from learning.partial_training_main_funct import partial_training
from learning.testing_partial import test_partial
from progress.bar import IncrementalBar
import learning.train
import learning.test
import getopt, sys

def convert_results_to_csv(results):
    import csv

    header = ['Mode', 'num_batches', 'sensitivity', 'accuracy', 'specificity', 'iteration']
    with open('./results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with open('./results.csv', 'a', newline='') as f:
        
        writer = csv.writer(f)
        
        writer.writerows(results)
    

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
    # Iterate Monte-carlo
    for k in range(runs_start, runs):
        # Set split
        data.set_fold(k)
        # Train model
        if train:
            marked_batches = partial_training(patients = data.get_train_set(), patients_val=data.get_validation_set(), setting = setting, fold = k, selection_mode = selection_mode, json_path = json_path)
        # Test model
        balanced_accuracy, sensitivity, specificity = test_partial(data.get_test_set(), k, setting, draw_map=draw_map, json_path=json_path)
        print('SPECIFIC post test partial')
        print(specificity)
        balanced_accuracies.append(balanced_accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    result_list = []
    for idx, ele in enumerate(balanced_accuracies):
        temp_list = [selection_mode, marked_batches, sensitivities[idx], ele, specificities[idx], idx]
        result_list.append(temp_list)
    
    convert_results_to_csv(result_list)

    # Save results
    for patients in data.get_test_set():
        for p in patients:
            results_folder = setting.get_data_setting().get_results_folder()
            label = p.get_diagnosis().get_label()
            p.save_predicted_scores(f"{results_folder}_{label}")
            if draw_map:
                p.save_map()
            


def run_train(data_directories, csv_file, working_directory, selection_mode, json_path=None):
    """ Set up setting and dataset and run training/testing
    MD added the json_path parameter for settings
    """
    s = setting.Setting(data_directories, csv_file, working_directory, json_path)

    data = dataset.Dataset(s)
    
    run(data, s, selection_mode=selection_mode, train=True, features_only=False, runs_start=0,runs=s.get_network_setting().get_runs(), draw_map=True, json_path=json_path)



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
    run_train(data_directory, csv_file, working_directory, selection_mode)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])
