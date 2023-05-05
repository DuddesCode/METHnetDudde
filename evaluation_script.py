"""script to set the evaluations of the network"""

import json
import os
from utils.helper import create_folder
from main import run_train

def create_folder_path(batch_size, mode, num_batches, monte_carlo_runs, epochs, runs):
    return f"{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/"

def fill_json(batch_size, num_batches, top_level_folder, mode, monte_carlo_runs, epochs, runs):
    folder_path = create_folder_path(batch_size, mode, num_batches,monte_carlo_runs, epochs, runs)
    print(folder_path)
    temp_dict = {"batch_size": batch_size,"num_batches": num_batches,"mode": mode,"test_score_folder": top_level_folder+'/'+folder_path, "monte_carlo_runs": monte_carlo_runs, 'epochs': epochs, 'runs':runs}
    json_string = json.dumps(temp_dict)



    if not os.path.isdir(top_level_folder):
        create_folder(top_level_folder)
    
    create_folder(top_level_folder+'/'+folder_path)

    with open(top_level_folder+'/'+folder_path+'setup.json', 'w') as outfile:
        outfile.write(json_string)

    return top_level_folder+'/'+folder_path

def eval_loop():

    batch_size = 128
    mode_list = ['solid', 'random', 'hand_picked', 'attention']
    num_batches = 1
    monte_carlo_runs = 1
    epochs = 3
    runs = 1
    work_dir = os.path.join(os.getcwd(), 'data_testing')
    csv_file = work_dir + '/test_HP.csv'
    data_dir = work_dir + '/wsi_test'

    for ele in mode_list:
        folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', ele, monte_carlo_runs, epochs, runs)
        run_train([data_dir], csv_file, work_dir, ele, folder)
        print(folder)

eval_loop()