"""script to set the evaluations of the network"""

import json
import os
from utils.helper import create_folder
from main import run_train_eval_mode
import numpy as np
import matplotlib.pyplot as plt
import sys

from arguments import setting
from datastructure import dataset


def create_folder_path(batch_size, mode, num_batches, epochs, runs):
    """used to define a model folder to write to"""
    return f"{mode}_{batch_size}_{num_batches}_{epochs}_{runs}/"

def fill_json(batch_size, num_batches, top_level_folder, mode, epochs, runs):
    """used to create a json file that gets written into a directory"""
    #get folder path
    folder_path = create_folder_path(batch_size, mode, num_batches, epochs, runs)
    #create dict to convert to json
    temp_dict = {"batch_size": batch_size,"num_batches": num_batches,"mode": mode,"test_score_folder": top_level_folder+'/'+folder_path, 'epochs': epochs, 'runs':runs}
    #convert dict to json
    json_string = json.dumps(temp_dict)


    #check that the folder to write to exists
    if not os.path.isdir(top_level_folder):
        create_folder(top_level_folder)
    #create the specific folder
    create_folder(top_level_folder+'/'+folder_path)

    #write json to folder
    with open(top_level_folder+'/'+folder_path+'setup.json', 'w') as outfile:
        outfile.write(json_string)

    return top_level_folder+'/'+folder_path

def eval_loop():
    #'hand_picked', 'solid', 'attention'
    #defines batchsize
    batch_size = 50
    #defines modes to train with
    mode_list = ['attention']
    #defines initial number of batches 
    num_batches = 15
    #defines number of epochs
    epochs = 200
    #defines number of monte carlo runs
    runs = 5
    #contains the working directory
    work_dir = os.path.join(os.getcwd(), 'data')
    csv_file = work_dir + '/Patients.csv'
    data_dir = [work_dir + '/wsi_C']
    #contains the number of batches to be e2e learned
    num_batches_list = [0, 2, 15, 50]
    draw_map = False

    #create a folder with json for the initial modell
    folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', mode_list[0], epochs, runs)

    #create an initial setting
    s = setting.Setting(data_dir, csv_file, work_dir, folder)

    #create the dataset
    data = dataset.Dataset(s)

    #create modell per defined mode
    for ele in mode_list:
        #create model per defined e2e batches
        #
            #create new subfolder
        folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', ele, epochs, runs)
        #set fodler paths to new ones
        s.reset_folder_path(folder)
        s.get_data_setting().reset_folder_paths(s)
        s.get_network_setting().reset_folder_path(s)
        #run train in evaluation mode
        if ele == 'attention':
            draw_map = True
        else:
            draw_map = False
        run_train_eval_mode(ele, draw_map, s, data, json_path=folder)
        
    
    val_losses = []
    train_losses = []
    test_losses = []
    epoch_times = []
    runtimes= []
    for mode in mode_list:
        for num_batch in num_batches_list:

            #test losses
            temp_test = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/test_losses.npy')
            test_losses.append(temp_test)

            #train losses
            temp_train = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/train_losses.npy')
            train_losses.append(temp_train)

            #val losses
            
            temp_val = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/val_losses.npy')
            val_losses.append(temp_val)

            #runtime
            temp_runtime = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/runtime.npy')
            runtimes.append(temp_runtime)

            temp_epoch_time = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/epoch_time.npy')
            epoch_times.append(temp_epoch_time)

            temp_t_img = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/time_spent_img.npy')
            print(temp_t_img)

            temp_t_data = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batch}_{epochs}_{runs}/time_spent_dataset.npy')
            print(temp_t_data)
            print(np.sum(temp_t_data))

    print("delta")
    print(epoch_times)
    for i, ele in enumerate(epoch_times):
        plt.plot(ele, label=num_batches_list[i])
    xend = len(epoch_times[0])
    plt.axis([0,xend+1,0.0,2000])
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("time[s]")

    plt.show()

    for i, ele in enumerate(runtimes):
        plt.scatter(ele / 60, i, label=num_batches_list[i])
    xend = len(runtimes[0])
    plt.legend(loc='best')
    plt.xlabel("time[s]")
    plt.ylabel("runs")    

    plt.show()

    print("delta")
    print(train_losses)
    for i, ele in enumerate(train_losses):
        
        plt.plot(ele, label=num_batches_list[i]*50)
    xend = max([len(train_losses[i]) for i in range(len(train_losses))])#len(train_losses[0])
    plt.axis([0,xend+1,0.0,2.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()

    print('gamma')
    for i, ele in enumerate(val_losses):
        print(ele)
        plt.plot(ele, label=num_batches_list[i]*50)
    xend = max([len(val_losses[i]) for i in range(len(val_losses))])
    plt.axis([0,xend+1,0.0,100.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()

    print("epsilon")
    #print(test_losses)
    for i, ele in enumerate(test_losses):
        print(ele)
        
        plt.plot(ele, label=num_batches_list[i]*50)
    xend = len(test_losses[0])
    plt.axis([0,xend+1,0.0,100.0])
    plt.legend(loc='best')
    plt.xlabel("samples")
    plt.ylabel("loss")


    



    plt.show()
    sys.exit()    


eval_loop()