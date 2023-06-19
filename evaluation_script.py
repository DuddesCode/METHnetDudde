"""script to set the evaluations of the network"""

import json
import os
from utils.helper import create_folder
from main import run_train
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    #'hand_picked', 'random', 'attention'
    batch_size = 128
    mode_list = ['solid']
    num_batches = 1
    monte_carlo_runs = 1
    epochs = 2
    runs = 1
    work_dir = os.path.join(os.getcwd(), 'data')
    csv_file = work_dir + '/test_HP.csv'
    data_dir = work_dir + '/wsi_test'

    np_losses = []
    for ele in mode_list:
        folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', ele, monte_carlo_runs, epochs, runs)
        
        run_train([data_dir], csv_file, work_dir, ele, folder)
        
    
    val_losses = []
    train_losses = []
    test_losses = []
    epoch_times = []
    runtimes= []
    for mode in mode_list:

        #test losses
        temp_test = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/test_losses.npy')
        test_losses.append(temp_test)

        #train losses
        temp_train = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/train_losses.npy')
        train_losses.append(temp_train)

        #val losses
        for ep in range(epochs):
            print(ep)
            temp_val = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/val_losses.npy')
            val_losses.append(temp_val)

        #runtime
        temp_runtime = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/runtime.npy')
        runtimes.append(temp_runtime)

        temp_epoch_time = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/epoch_time.npy')
        epoch_times.append(temp_epoch_time)

    for i, ele in enumerate(epoch_times):
        plt.plot(ele, label=mode_list[i])
    xend = len(epoch_times[0])
    plt.axis([0,xend+1,0.0,1000.0])
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("time")

    plt.show()

    for i, ele in enumerate(runtimes):
        plt.scatter(ele, i, label=mode_list[i])
    xend = len(runtimes[0])
    plt.legend(loc='best')
    plt.xlabel("runs")
    plt.ylabel("time")    

    plt.show()


    for i, ele in enumerate(train_losses):
        print(ele)
        plt.subplot(2,1,1)
        plt.plot(ele, label=mode_list[i])
    xend = len(train_losses[0])
    plt.axis([0,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    
    for i, ele in enumerate(test_losses):
        print(ele)
        plt.subplot(2,1,2)
        plt.plot(ele, label='test')
    xend = len(test_losses[0])
    plt.axis([0,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")


    plt.show()
    
    for i, ele in enumerate(val_losses):
        print(ele)
        plt.subplot(2,1,2)
        plt.plot(ele, label=f'ep{i}')
    xend = len(val_losses[0])
    plt.axis([0,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")


    plt.show()
    sys.exit()    


eval_loop()