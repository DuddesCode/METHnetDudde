from learning.train import calculate_error
import torch
import numpy as np
from tqdm import tqdm

from learning.partial_training_main_funct import save_it_losses, Patient_Dataset, DataLoader, create_cleaned_dataset, CleanDataset, collate_features, SequentialSampler, test_decoder, test_encoder
from models.restnet_custom import resnet50_baseline
from models.model_Full import Partial_Net
from models.attention_model import Attention_MB

def test_partial(test_patients, fold, setting, draw_map = True, json_path= None):
    """Runs model prediction for a partially trained model"""

    network_setting = setting.get_network_setting()
    feature_setting = setting.get_feature_setting()
    n_classes = setting.get_class_setting().get_n_classes()

    p_encoder = resnet50_baseline()
    p_decoder = Attention_MB(setting)

    model = Partial_Net(p_encoder, p_decoder)

    patients_test = []

    for patients in test_patients:
        for patient in patients:
            patients_test.append(patient)

    model_folder = network_setting.get_model_folder()
    model_file = 's_{}_checkpoint.pt'.format(fold)
    state_dict_poll = torch.load(model_folder + model_file)
    from collections import OrderedDict
    state_dict_clean = OrderedDict()
    for k, v in state_dict_poll.items():
        name = k.replace('module.', '')
        state_dict_clean[name] = v

    model.load_state_dict(state_dict_clean)

    balanced_accuracy, sensitivity, specificity = test_partial_model(model, n_classes, patients_test, feature_setting, draw_map, json_path)

    return balanced_accuracy, sensitivity, specificity


def test_partial_model(model, n_classes, patients_test, feature_setting, draw_map, json_path):
    """tests the partial model"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.eval()
    model.to(device)
    # Values for sensitivity and specificity
    error_class_wise = np.zeros(n_classes)
    counter_class_wise = np.zeros(n_classes)
    test_losses = []
    for p in patients_test:
        # Iterate image properties
        label = p.get_diagnosis().get_label()
        if label == 0:
            label_for_error_classs_wise = 0
            label = torch.zeros(1)
            label = label.to(device)
        else:
            label_for_error_classs_wise = 1
            label = torch.ones(1)
            label = label.to(device)
        for wsi_prop in p.get_wsis():
            for wsi in wsi_prop:
                for i in range(len(wsi.get_tile_properties())):
                    if len(wsi.get_tiles_list()) != 0:
                        wsi.load_wsi()

                        data = Patient_Dataset(wsi, i)
                        data, times_placehold = create_cleaned_dataset(data, None)
                        data = CleanDataset(data)

                        loader = DataLoader(dataset= data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                        features = test_encoder(loader, model, device)
                        error, loss, Y_prob, Y_hat= test_decoder(model, features, label, device, draw_map)
                        test_losses.append(loss)
                        error_class_wise[label_for_error_classs_wise] += error
                        counter_class_wise[label_for_error_classs_wise] += 1
                        print(Y_prob.cpu()[0][1].numpy())
                        p.get_diagnosis().add_predicted_score(Y_prob.cpu()[0][label_for_error_classs_wise].numpy())
    test_losses = np.array(test_losses, dtype=np.float32)
    test_error_class_wise = np.array(error_class_wise, dtype=np.float32)
    test_counter_class_wise = np.array(counter_class_wise, dtype= np.float32)
    if json_path is not None:
        save_it_losses(test_losses, json_path, 'test_losses')
        save_it_losses(test_error_class_wise, json_path, 'test_error_class_wise')
        save_it_losses(test_counter_class_wise, json_path, 'test_counter_class_wise')
    sensitivity = (counter_class_wise[0] - error_class_wise[0]) / counter_class_wise[0]
    specificity = (counter_class_wise[1] - error_class_wise[1]) / counter_class_wise[1]

    balanced_accuracy = (sensitivity - specificity) / 2.

    return balanced_accuracy, sensitivity, specificity