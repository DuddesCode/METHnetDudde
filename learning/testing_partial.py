from learning.train import calculate_error
import torch
import numpy as np
from tqdm import tqdm

from learning.partial_training_main_funct import Patient_Dataset, DataLoader, create_cleaned_dataset, CleanDataset, collate_features, SequentialSampler, test_decoder, test_encoder
from models.restnet_custom import resnet50_baseline
from models.model_Full import Partial_Net
from models.attention_model import Attention_MB

def test_partial(test_patients, fold, setting, draw_map = True):
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

    model.load_state_dict(torch.load(model_folder + model_file))

    balanced_accuracy, sensitivity, specificity = test_partial_model(model, n_classes, patients_test, feature_setting, draw_map)

    return balanced_accuracy, sensitivity, specificity


def test_partial_model(model, n_classes, patients_test, feature_setting, draw_map):
    """tests the partial model"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.eval()
    model.to(device)
    # Values for sensitivity and specificity
    error_class_wise = np.zeros(n_classes)
    counter_class_wise = np.zeros(n_classes)

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
                        data, marked_tens, pos_list = create_cleaned_dataset(data)
                        data = CleanDataset(data)

                        loader = DataLoader(dataset= data, batch_size=feature_setting.get_batch_size(), collate_fn=collate_features, sampler=SequentialSampler(data))
                        features = test_encoder(loader, model, device)
                        error, Y_prop = test_decoder(model, features, label, device, draw_map)
                        error_class_wise[label_for_error_classs_wise] += error
                        counter_class_wise[label_for_error_classs_wise] += 1
                        p.get_diagnosis().add_predicted_score(Y_prop.cpu().item())

    sensitivity = (counter_class_wise[0] - error_class_wise[0]) / counter_class_wise[0]
    specificity = (counter_class_wise[1] - error_class_wise[1]) / counter_class_wise[1]

    balanced_accuracy = (sensitivity - specificity) / 2.

    return balanced_accuracy, sensitivity, specificity