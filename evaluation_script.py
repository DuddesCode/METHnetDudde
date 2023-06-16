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
    epochs = 1
    runs = 1
    work_dir = os.path.join(os.getcwd(), 'data')
    csv_file = work_dir + '/test_HP.csv'
    data_dir = work_dir + '/wsi_test'

    np_losses = []
    for ele in mode_list:
        folder = fill_json(batch_size, num_batches, os.getcwd()+'/tests', ele, monte_carlo_runs, epochs, runs)
        
        run_train([data_dir], csv_file, work_dir, ele, folder)
        
        print(folder)
    
    val_losses_pos = []
    val_losses_neg = []
    train_losses_neg = []
    train_losses_pos = []
    test_losses_neg = []
    test_losses_pos = []
    epoch_times = []
    runtimes= []
    for mode in mode_list:

        #test losses
        temp_test_pos = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/test_losses_pos.npy')
        temp_test_neg = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/test_losses_neg.npy')
        test_losses_pos.append(temp_test_pos)
        test_losses_neg.append(temp_test_neg)

        #train losses
        temp_train_neg = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/train_losses_neg.npy')
        temp_train_pos = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/train_losses_pos.npy')
        train_losses_neg.append(temp_train_neg)
        train_losses_pos.append(temp_train_pos)

        #val losses
        for ep in range(epochs):
            print(ep)
            temp_pos = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/val_losses_pos_{ep}.npy')
            temp_neg = np.load(os.getcwd()+f'/tests/{mode}_{batch_size}_{num_batches}_{monte_carlo_runs}_{epochs}_{runs}/val_losses_neg_{ep}.npy')
            val_losses_pos.append(temp_pos)
            val_losses_neg.append(temp_neg)

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


    for i, ele in enumerate(train_losses_pos):
        print(ele)
        plt.subplot(6,1,1)
        plt.plot(ele, label=mode_list[i])
    xend = len(train_losses_pos[0])
    plt.axis([1,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    
    for i, ele in enumerate(train_losses_neg):
        print(ele)
        plt.subplot(6,1,2)
        plt.plot(ele, label=mode_list[i])
    xend = len(train_losses_neg[0])
    plt.axis([1,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")

    for i, ele in enumerate(test_losses_pos):
        print(ele)
        plt.subplot(6,1,3)
        plt.plot(ele, label=mode_list[i])
    xend = len(test_losses_pos[0])
    plt.axis([1,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")

    for i, ele in enumerate(train_losses_neg):
        print(ele)
        plt.subplot(6,1,4)
        plt.plot(ele, label=mode_list[i])
    xend = len(test_losses_neg[0])
    plt.axis([1,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    
    for i, ele in enumerate(val_losses_pos):
        print(ele)
        plt.subplot(6,1,5)
        plt.plot(ele, label=f'ep{i}')
    xend = len(val_losses_pos[0])
    plt.axis([1,xend+1,0.0,1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")

    for i, ele in enumerate(val_losses_neg):
        print('es')
        print(ele)
        plt.subplot(6,1,6)
        plt.plot(ele, label=f'ep{i}')
    xend = len(val_losses_neg[0])
    print(xend)
    plt.axis([0,xend, 0.0, 1.0])
    plt.legend(loc='best')
    plt.xlabel("iterations")
    plt.ylabel("loss")

    plt.show()
    sys.exit()    




eval_loop()