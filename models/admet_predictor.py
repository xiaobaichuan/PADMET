#!/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from utils.my_dataloader import SequenceDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_path = os.getcwd() + '/models/new'
pattern_class = os.path.join(current_path, 'classification_checkpoint_epoch_2_fold_2.pt')
"""['regression_Bioavailability_checkpoint_epoch_1_fold_8.pt',
                'regression_VD_checkpoint_epoch_0_fold_8.pt',
                'regression_T1_2_checkpoint_epoch_0_fold_6.pt',
                'regression_CL_checkpoint_epoch_3_fold_8.pt']"""

pattern_adme = ['regression_Bioavailability_checkpoint_epoch_34_fold_0.pt',
                'regression_VD_checkpoint_epoch_3_fold_4.pt',
                'regression_T1_2_checkpoint_epoch_60_fold_1.pt',
                'regression_CL_checkpoint_epoch_18_fold_6.pt']
caco2_model_path = os.path.join(current_path, 'regression_CACO2_checkpoint_epoch_49_fold_0.pt')
pattern_adme = [os.path.join(current_path, adme_model) for adme_model in pattern_adme]


def get_forward_args(batch_data, device):
    species, route, x1, x2, x3, x4, x5 = batch_data[:7]
    args = {
        'species': species.to(device),
        'route': route.to(device),
        'smiles_ls': [x.to(device) for x in [x1, x2, x3, x4, x5]]
    }
    return args


def predict_classification(model, test_loader, device):
    model.eval()
    toxicity_flags = []
    with torch.no_grad():
        for batch_data in test_loader:
            args = get_forward_args(batch_data, device)
            _, classification_out = model(**args)
            toxicity_flags.append(classification_out.cpu().numpy().tolist())
    sigmoid_tensor = torch.sigmoid(torch.tensor(toxicity_flags[0]))
    toxicity_binary = (sigmoid_tensor > 0.5).float()
    return toxicity_binary


def predict_regression(model, test_loader, regression_checkpoint_paths, device):
    task_scores = []
    for checkpoint_path in regression_checkpoint_paths:
        regression_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(regression_checkpoint['model_state_dict'])
        model.eval()
        tmp = []
        with torch.no_grad():
            for batch_data in test_loader:
                args = get_forward_args(batch_data, device)
                regression_out, _ = model(**args)
                tmp.append(regression_out.cpu())
        real_adme = [torch.tensor(np.expm1(pred.numpy())) for pred in tmp]
        task_scores.append(real_adme)
    task_scores = list(map(list, zip(*task_scores)))
    return task_scores


def caco2_pred(species, route_tensor, smiles_ls_tensor, model, batch_size, device):
    checkpoint_path = caco2_model_path
    species_ = torch.zeros(len(species), 28, dtype=torch.float32, device=device)
    species_[:, 19] = 1
    dataset = SequenceDataset(
        species=species_,
        route=route_tensor,
        smiles_ls=smiles_ls_tensor
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    regression_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(regression_checkpoint['model_state_dict'])
    model.eval()
    tmp = []
    with torch.no_grad():
        for batch_data in test_loader:
            args = get_forward_args(batch_data, device)
            regression_out, _ = model(**args)
            tmp.append(regression_out.cpu())
    real_adme = [torch.tensor(np.expm1(pred.numpy())) for pred in tmp]
    return real_adme


def discriminator(species, route_tensor, smiles_ls_tensor, model, batch_size, device):
    dataset = SequenceDataset(
        species=species,
        route=route_tensor,
        smiles_ls=smiles_ls_tensor
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("加载分类模型...")
    classification_checkpoint = torch.load(pattern_class, map_location=device)
    model.load_state_dict(classification_checkpoint['model_state_dict'])
    toxicity_flags = predict_classification(model, test_loader, device)

    print("加载回归模型并进行预测...")
    adme_scores = predict_regression(model, test_loader, pattern_adme, device)
    # print(adme_scores)
    # adme_scores_tensor = torch.tensor(adme_scores, dtype=torch.float32)
    adme_scores_tensor = torch.cat(adme_scores[0], dim=1)
    print(f"adme_scores_tensor: {adme_scores_tensor}\ntoxicity_flags: {toxicity_flags}")

    toxicity_flags_tensor = torch.tensor(toxicity_flags, dtype=torch.float32)
    # print(adme_scores_tensor.size(), toxicity_flags_tensor.size())
    return adme_scores_tensor, toxicity_flags_tensor


if __name__ == '__main__':
    model_test = Scarlett(n_targets=1, num_classes=11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_test.to(device)
    # epoch = 50
    # batch_size = 4
    # checkpoint_path = []
