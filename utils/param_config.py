#!/user/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import torch
import yaml


class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.num_workers = config['training']['num_workers']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = Path(config['paths']['model_save_path'])
        self.output_save_path = Path(config['paths']['output_save_path'])
        self.train_reg_data_path = Path(config['paths']['train_reg_data_path'])
        self.train_class_data_path = Path(config['paths']['train_class_data_path'])
        self.test_reg_data_path = Path(config['paths']['test_reg_data_path'])
        self.test_class_data_path = Path(config['paths']['test_class_data_path'])
        # self.test_grad_cam_data_path = Path(config['paths']['test_grad_cam_data_path'])
