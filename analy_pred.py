#!/user/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json

import torch

from models.admet_predictor import discriminator, caco2_pred
from models.padmet_model import Padmet
from utils.param_config import Config
from utils.precess_txt import preprocess_txt, txt_df


class DataRunner:
    def __init__(self, config_path, model=None):
        self.config = Config(config_path)
        if model is None:
            self.model = Padmet(n_targets=1, num_classes=11)
        else:
            self.model = model
        self.device = getattr(self.config, 'device', 'cpu')
        self.batch_size = getattr(self.config, 'batch_size', 32)

    def run(self, input_path):
        data = preprocess_txt(input_path)
        dic_tensor = txt_df(data)
        smiles_tensor = dic_tensor['smiles_ls_tensor']
        species = dic_tensor['species_tensor']
        routes = dic_tensor['route_tensor']
        adme_scores_tensor, toxicity_flags_tensor = discriminator(species, routes, smiles_tensor, self.model, self.batch_size, self.device)
        caco_data_tensor = caco2_pred(species, routes, smiles_tensor, self.model, self.batch_size, self.device)[0]
        adme_scores_tensor = torch.cat([adme_scores_tensor, caco_data_tensor], dim=1)
        adme_scores = adme_scores_tensor.tolist()
        result = {
            "adme_scores": [
                [round(row[0] * 100, 2)] + [round(x, 2) for x in row[1:]]
                for row in adme_scores
            ],
            "toxicity_flags": [[str(bool(i)) for i in x] for x in toxicity_flags_tensor.tolist()]
        }
        print(json.dumps(result))


# 用法示例
if __name__ == '__main__':
    runner = DataRunner('./utils/config.yaml')
    parser = argparse.ArgumentParser(description='Process summary table file.')
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()
    runner.run(args.input_path)
