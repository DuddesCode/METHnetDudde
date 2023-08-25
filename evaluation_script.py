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
    return f"{mode}_{batch_size}_{num_batches}_{epochs}_{runs}/"

def fill_json(batch_size, num_batches, top_level_folder, mode, epochs, runs):
    folder_path = create_folder_path(batch_size, mode, num_batches, epochs, runs)
    print(folder_path)
    temp_dict = {"batch_size": batch_size,"num_batches": num_batches,"mode": mode,"test_score_folder": top_level_folder+'/'+folder_path, 'epochs': epochs, 'runs':runs}
    json_string = json.dumps(temp_dict)



    if not os.path.isdir(top_level_folder):
        create_folder(top_level_folder)
    
    create_folder(top_level_folder+'/'+folder_path)

    with open(top_level_folder+'/'+folder_path+'setup.json', 'w') as outfile:
        outfile.write(json_string)

    return top_level_folder+'/'+folder_path

def eval_loop():
    #'hand_picked', 'random', 'attention'
    batch_size = 50
    mode_list = ['random']
    num_batches = 0
    epochs = 70
    runs = 1
    work_dir = os.path.join(os.getcwd(), 'data')
    csv_file = work_dir + '/Patients.csv'
    data_dir = [work_dir + '/wsi_C']
    num_batches_list = [ 0, 2, 15, 75]
    draw_map = False

    np_losses = []
    folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', mode_list[0], epochs, runs)

    s = setting.Setting(data_dir, csv_file, work_dir, folder)

    data = dataset.Dataset(s)

    for ele in mode_list:
        for batch_num in num_batches_list:
            folder = fill_json(batch_size, batch_num, os.getcwd()+'/tests', ele, epochs, runs)
            s.reset_folder_path(folder)
            s.get_data_setting().reset_folder_paths(s)
            s.get_network_setting().reset_folder_path(s)
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