#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os.path
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import SubsetRandomSampler, DataLoader

from models.padmet_model import Padmet, MultiPropertyDeepLC
from utils.train_loss import Criterion, AsymmetricLoss, LogHuberLoss, AdaptiveHuber
from utils.data_precess import preprocess_data
from utils.early_stop import TaskAwareEarlyStopping
from utils.metrics import reg_metrics, calculate_multilabel_metrics, calculate_retain_count, calculate_toxibtl_metrics
from utils.my_dataloader import SequenceDataset
from utils.param_config import Config

torch.multiprocessing.set_sharing_strategy('file_system')


class ClassificationTrainer(object):

    def __init__(self, model, config):
        # super().__init__.py(config)
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        # self.class_criterion = AsymmetricLoss()
        # self.class_criterion = BCEWithLogitsLoss()
        self.reg_criterion = LogHuberLoss()
        # self.reg_criterion = AdaptiveHuber()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-5
        )
        self.cls_index = None
        # self.setup_logging()

    # def setup_logging(self):
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='%(asctime)s - %(levelname)s - %(message)s',
    #         handlers=[
    #             logging.FileHandler(f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
    #             logging.StreamHandler()
    #         ]
    #     )
    #     self.logger = logging.getLogger(__name__)
    #     # wandb.__init__.py.py(project="segmentation-training")

    def setup_dataloaders(self,
                          task_type='toxicity',
                          test_data='',
                          target=None,
                          data_paths=None):
        if task_type == 'toxicity':
            cache_file = f'./data/all_{test_data}data_toxicity_50.pt'
            # cache_file = f'./data/toxibtl_data/all_{test_data}data_toxicity_toxibtl.pt'
        elif task_type == 'adme':
            cache_file = f'./data/all_{test_data}data_{target}.pt'
        else:
            raise ValueError("task_type must be 'toxicity' or 'adme'.")

        if os.path.exists(cache_file):
            loaded_data = torch.load(cache_file)
        else:
            if test_data:
                data_path = data_paths['test_path']
            else:
                data_path = data_paths['train_path']
            if task_type == 'toxicity':
                loaded_data = preprocess_data(data_path, task_type='toxicity', test_data=test_data)
            else:
                loaded_data = preprocess_data(data_path, task_type='adme', test_data=test_data, target_column=target)

        species_tensor = loaded_data['species_tensor']
        route_tensor = loaded_data['route_tensor']
        smiles_ls_tensor = loaded_data['smiles_ls_tensor']
        if task_type == 'toxicity':
            targets = loaded_data['toxicity_targets']
            dataset = SequenceDataset(species_tensor, route_tensor, smiles_ls_tensor, None, targets)
        else:
            targets = loaded_data['regression_targets']
            dataset = SequenceDataset(species_tensor, route_tensor, smiles_ls_tensor, targets, None)

        return dataset

    def _get_forward_args(self, batch_data, task, phase='train'):
        species, route, x1, x2, x3, x4, x5 = batch_data[:7]
        args = {
            'species': species.to(self.device),
            'route': route.to(self.device),
            'smiles_ls': [x.to(self.device) for x in [x1, x2, x3, x4, x5]]
        }
        if isinstance(self.model, MultiPropertyDeepLC) and phase == 'train':
            if task == "regression":
                args['y_reg'] = batch_data[7].to(self.device)
            elif task == "classification":
                args['y_cls'] = batch_data[7].to(self.device)
        return args

    def train_epoch(self, train_loader: DataLoader, task: str, mask=None) -> float:
        self.model.train()
        total_samples, real_total_loss = 0, 0.0
        for batch_idx, (batch_data) in enumerate(train_loader):
            self.optimizer.zero_grad()
            if mask is not None:
                current_indices = train_loader.sampler.indices[batch_idx * self.config.batch_size:(batch_idx + 1) * self.config.batch_size]
                # loss = self.criterion(output, target)
                batch_mask = mask[current_indices].to(self.device)
                loss, _, _ = self._run_batch(batch_data, task, phase='train', mask=batch_mask)
            else:
                loss, _, _ = self._run_batch(batch_data, task, phase='train')
            loss.backward()
            # regression_out, _ = self.model(species_tensor, route_tensor, smiles_ls_tensor, y_reg=regression_targets)
            # _, classification_out = self.model(species_tensor, route_tensor, smiles_ls_tensor, y_cls=toxicity_targets)
            # total_loss = loss_regression + loss_classification
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            batch_size = batch_data[0].size(0)
            real_total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx}: {task.capitalize()} Loss = {loss.item():.6f}, LR = {current_lr:.6e}")
        return real_total_loss / total_samples

    @torch.no_grad()
    def validate(self, valid_loader: DataLoader, task: str, strategy: str):
        self.model.eval()
        total_mae, total_val_loss, total_samples = 0.0, 0.0, 0
        all_valid_predictions, all_valid_targets = [], []
        for batch_idx, batch_data in enumerate(valid_loader):
            loss, outputs, targets = self._run_batch(batch_data, task, phase='valid')
            batch_size = batch_data[0].size(0)
            total_val_loss += loss.item() * batch_size
            total_samples += batch_size
            all_valid_predictions.append(outputs.cpu())
            all_valid_targets.append(targets.cpu())

        if task == "regression":
            all_valid_predictions = [torch.tensor(np.expm1(pred.numpy())) for pred in all_valid_predictions]
            all_valid_targets = [torch.tensor(np.expm1(target.numpy())) for target in all_valid_targets]
            metrics = reg_metrics(all_valid_predictions, all_valid_targets)
            print(metrics)
            return total_val_loss / total_samples, metrics
        elif task == "classification":
            res_score = calculate_multilabel_metrics(all_valid_predictions, all_valid_targets, strategy=strategy)
            # res_score = calculate_toxibtl_metrics(all_valid_predictions, all_valid_targets)
            # f1_scores = [class_evaluate_metrics(pred, valid_data)['micro_f1'] for pred, valid_data in zip(all_valid_predictions, all_valid_targets)]
            # mcc_values = [metrics['MCC'] for metrics in res_score.values()]
            # return total_val_loss / total_samples, float(np.mean(f1_scores))
            # return total_val_loss / total_samples, sum(mcc_values) / len(mcc_values)
            return total_val_loss / total_samples, res_score

    def train_classification(self, adme_target=None):
        self.train_dataset(adme_target, task="classification")

    def k_fold_cross_validation(self, k_folds=10, strategy='mean'):
        loaded_data = self.setup_dataloaders(
            task_type='toxicity',
            test_data='',
            data_paths={
                'train_path': self.config.train_class_data_path,
            }
        )
        toxicity_targets = loaded_data.toxicity_targets
        targets = toxicity_targets.numpy() if isinstance(toxicity_targets, torch.Tensor) else np.array(toxicity_targets)
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        num_samples, num_classes = targets.shape
        toxic_class_indices = {}
        no_toxic_indices = np.where(np.all(targets == 0, axis=1))[0]
        np.random.shuffle(no_toxic_indices)
        for cls in range(num_classes):
            toxic_class_indices[cls] = np.where(targets[:, cls] == 1)[0]
        toxic_counts = [len(toxic_class_indices[cls]) for cls in range(num_classes)]
        retain_count = calculate_retain_count(len(no_toxic_indices), toxic_counts, strategy)
        no_toxic_indices = no_toxic_indices[:retain_count]
        for fold in range(k_folds):
            print(f"Fold {fold + 1}/{k_folds}")
            train_indices = set()
            valid_indices = set()
            no_toxic_train, no_toxic_valid = list(kfold.split(no_toxic_indices))[fold]
            no_toxic_train_indices = no_toxic_indices[no_toxic_train]
            no_toxic_valid_indices = no_toxic_indices[no_toxic_valid]
            train_indices.update(no_toxic_train_indices)
            valid_indices.update(no_toxic_valid_indices)

            for cls in range(num_classes):
                class_train, class_valid = list(kfold.split(toxic_class_indices[cls]))[fold]
                class_train_indices = toxic_class_indices[cls][class_train]
                class_valid_indices = toxic_class_indices[cls][class_valid]

                train_indices.update(class_train_indices)
                valid_indices.update(class_valid_indices)

            train_indices = list(train_indices - valid_indices)
            valid_indices = list(valid_indices)
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(
                loaded_data,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            valid_loader = DataLoader(
                loaded_data,
                batch_size=self.config.batch_size,
                sampler=valid_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            yield train_loader, valid_loader

    def k_fold_cross_validation_per_class(self, k_folds=10, strategy='mean'):
        loaded_data = self.setup_dataloaders(
            task_type='toxicity',
            test_data='',
            data_paths={
                'train_path': self.config.train_class_data_path,
            }
        )
        toxicity_targets = loaded_data.toxicity_targets
        targets = toxicity_targets.numpy() if isinstance(toxicity_targets, torch.Tensor) else np.array(toxicity_targets)
        num_samples, num_classes = targets.shape
        no_toxic_indices = np.where(np.all(targets == 0, axis=1))[0]
        np.random.shuffle(no_toxic_indices)
        toxic_class_indices = {cls: np.where(targets[:, cls] == 1)[0] for cls in range(num_classes)}
        toxic_counts = [len(toxic_class_indices[cls]) for cls in range(num_classes)]
        retain_count = calculate_retain_count(len(no_toxic_indices), toxic_counts, strategy)
        selected_no_toxic_indices = no_toxic_indices[:retain_count]

        # 每个类别单独做K折
        for cls in range(num_classes):
            early_stopping = TaskAwareEarlyStopping(
                task='classification',
                patience=6,
                min_delta=0.001
            )
            print(f"\n==== Class {cls} ====")
            pos_indices = toxic_class_indices[cls]
            all_indices = np.concatenate([pos_indices, selected_no_toxic_indices])
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            for fold, (train_idx, valid_idx) in enumerate(kfold.split(all_indices)):
                train_indices = all_indices[train_idx]
                valid_indices = all_indices[valid_idx]

                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(valid_indices)

                train_loader = DataLoader(
                    loaded_data,
                    batch_size=self.config.batch_size,
                    sampler=train_sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
                valid_loader = DataLoader(
                    loaded_data,
                    batch_size=self.config.batch_size,
                    sampler=valid_sampler,
                    num_workers=self.config.num_workers,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
                yield cls, early_stopping, fold, train_loader, valid_loader

    def k_fold_cross_validation_regression(self, adme_target, k_folds=10):
        loaded_data = self.setup_dataloaders(
            task_type='adme',
            test_data='',
            target=adme_target,
            data_paths={
                'train_path': self.config.train_reg_data_path,
            }
        )
        # loaded_data = self.setup_dataloaders(
        #     task_type='toxicity',
        #     test_data='',
        #     data_paths={
        #         'train_path': self.config.train_class_data_path,
        #     }
        # )
        num_samples = len(loaded_data)
        indices = np.arange(num_samples)
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(indices)):
            print(f"Fold {fold + 1}/{k_folds}")
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_loader = DataLoader(
                loaded_data,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            valid_loader = DataLoader(
                loaded_data,
                batch_size=self.config.batch_size,
                sampler=valid_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            yield train_loader, valid_loader

    def train_dataset(self, adme_target, task="classification"):
        all_metrics = []
        tmp = None
        # early_stopping = TaskAwareEarlyStopping(
        #     task=task,
        #     patience=6,
        #     min_delta=0.001
        # )
        if task == "classification":
            k_folds = self.k_fold_cross_validation(k_folds=9, strategy='mean')
            # k_folds = self.k_fold_cross_validation_regression(k_folds=9, adme_target=None)
            # k_folds = self.k_fold_cross_validation_per_class(k_folds=9, strategy='mean')
            mask = None
        elif task == "regression":
            k_folds = self.k_fold_cross_validation_regression(k_folds=9, adme_target=adme_target)
            mask = None
        else:
            raise ValueError("Invalid task type. Choose either 'regression' or 'classification'.")
        for fold, (train_loader, valid_loader) in enumerate(k_folds):
            early_stopping = TaskAwareEarlyStopping(
                task=task,
                patience=8,
                min_delta=0.001
            )
            # for cls_index, early_stopping, fold, train_loader, valid_loader in k_folds:
            #     self.cls_index = cls_index
            for epoch in range(self.config.epochs):
                train_loss = self.train_epoch(train_loader, task=task, mask=mask)
                val_loss, valid_scores = self.validate(valid_loader, task=task, strategy='mean')
                self.scheduler.step()
                if task == 'regression':
                    is_best, should_stop = early_stopping(task=task, valid_scores=valid_scores['MAE'])
                    print(f'Epoch {epoch + 1}/{self.config.epochs} | Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f} | Validation MAE: {valid_scores["MAE"]:.6f}')
                else:
                    mcc_values = [metrics['MCC'] for metrics in valid_scores.values()]
                    # mcc_values = valid_scores['MCC']
                    mcc_mean = sum(mcc_values) / len(mcc_values)
                    print(f'Epoch {epoch + 1}/{self.config.epochs} | Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f} | Validation MCC: {mcc_mean:.6f}')
                    # is_best, should_stop = early_stopping(task=task, valid_scores=mcc_values)
                    is_best, should_stop = early_stopping(task=task, valid_scores=mcc_mean)

                if is_best:
                    # self.save_checkpoint(epoch, val_loss)
                    self.save_checkpoint(epoch, fold, task=task, adme_target=adme_target)
                    tmp = valid_scores
                if should_stop:
                    print('Early stopping triggered', valid_scores)
                    all_metrics.append(tmp)
                    break
        if task == 'classification':
            # output_data = self.config.output_save_path / f'train_class{self.cls_index}_info.txt'
            output_data = self.config.output_save_path / f'train_class_info.txt'
            flattened_data = []
            for metrics in all_metrics:
                for class_name, values in metrics.items():
                    flattened_row = {'Class': class_name}
                    flattened_row.update(values)
                    flattened_data.append(flattened_row)
            df = pd.DataFrame(flattened_data)
            averages = df.groupby('Class').mean(numeric_only=True)
        else:
            output_data = self.config.output_save_path / f'train_reg_info_{adme_target}.txt'
            df = pd.DataFrame(all_metrics)
            averages = df.mean()
        print(adme_target, averages, all_metrics)
        # with open(output_data, "a") as f:
        #     f.write(f"ADME Target: {adme_target}\n")
        #     f.write("Averages:\n")
        #     f.write(averages.to_string())
        #     f.write("\nAll Metrics:\n")
        #     f.write(str(all_metrics))
        #     f.write('\n\n')

    def train_regression(self, adme_target_ls):
        # 冻结分类相关参数
        checkpoint_path = self.config.model_save_path / f'classification_checkpoint_epoch_2_fold_2.pt'
        self.load_checkpoint(checkpoint_path)
        self.freeze_model_parameters(["global_dense", "one_hot_conv", "amino_conv", "inter4", "diamino_conv"])
        self.optimizer = self.update_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-5
        )
        for adme_target in adme_target_ls:
            self.train_dataset(adme_target, task="regression")

    def save_checkpoint(self, epoch: int, fold: int, task: str, adme_target: str = ''):
        checkpoint = {
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if adme_target:
            save_path = self.config.model_save_path / f'{task}_{adme_target}_checkpoint_epoch_{epoch}_fold_{fold}.pt'
        else:
            # save_path = self.config.model_save_path / f'{task}_checkpoint_epoch_{epoch}_fold_{fold}_class{self.cls_index}.pt'
            save_path = self.config.model_save_path / f'{task}_checkpoint_epoch_{epoch}_fold_{fold}.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f'Saved checkpoint to {save_path}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')

    def update_optimizer(self):
        """
        Update optimizer to only optimize unfrozen parameters.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate
        )
        return optimizer

    def freeze_model_parameters(self, freeze_keys):
        """
        Freeze all model parameters except the regression head.
        """
        for name, param in self.model.named_parameters():
            if any(key in name for key in freeze_keys):
                param.requires_grad = False
            else:
                param.requires_grad = True

    @torch.no_grad()
    def predict_admet(self, test_loader, task, adme_target=None):
        self.model.eval()
        predictions = []
        species_ls, route_ls = [], []
        for batch_idx, batch_data in enumerate(test_loader):
            forward_args = self._get_forward_args(batch_data, task, phase='test')
            regression_out, classification_out = self.model(**forward_args)
            species_tensor = forward_args['species']
            route_tensor = forward_args['route']
            if task == "regression":
                regression_out_np = regression_out.cpu().numpy()
                predictions.extend(np.expm1(regression_out_np))

            elif task == "classification":
                pred = torch.sigmoid(classification_out).cpu().numpy()
                binary_pred = (pred > 0.5).astype(int)
                predictions.extend(binary_pred)
            species_ls.extend(species_tensor.cpu().numpy())
            route_ls.extend(route_tensor.cpu().numpy())
        admet_property = adme_target or task
        species_ls = [list(species) for species in species_ls]
        route_ls = [list(route) for route in route_ls]
        if task == 'classification':
            predictions = [list(pred) for pred in predictions]
        else:
            predictions = [list(pred)[0] for pred in predictions]
        results_df = pd.DataFrame({
            'Species_ont_hot': species_ls,
            'Route_ont_hot': route_ls,
            f'{admet_property}' if admet_property else 'one_hot': predictions
        })
        return results_df

    @staticmethod
    def regression_to_classification(predictions, targets, bins):

        pred_classes = [torch.bucketize(pred, torch.tensor(bins)) for pred in predictions]
        target_classes = [torch.bucketize(target, torch.tensor(bins)) for target in targets]
        return pred_classes, target_classes

    @torch.no_grad()
    def predict_admet2(self, test_loader, task, adme_target=None):
        self.model.eval()
        total_mae, total_val_loss, total_samples = 0.0, 0.0, 0
        all_test_predictions, all_test_targets = [], []
        for batch_idx, batch_data in enumerate(test_loader):
            loss, outputs, targets = self._run_batch(batch_data, task=task, phase='test')
            batch_size = batch_data[0].size(0)
            total_val_loss += loss.item() * batch_size
            total_samples += batch_size
            all_test_predictions.append(outputs.cpu())
            all_test_targets.append(targets.cpu())
            # pred = torch.sigmoid(classification_out) > 0.5
            # metrics = class_evaluate_metrics(pred, toxicity_targets)
        if task == "regression":
            all_test_predictions = [torch.tensor(np.expm1(pred.numpy())) for pred in all_test_predictions]
            all_test_targets = [torch.tensor(np.expm1(target.numpy())) for target in all_test_targets]
            # if adme_target == 'Bioavailability':
            #     bins = [0.2, 0.5, 0.8, 1]
            #     predictions = [torch.clamp(pred, min=0.0, max=1.0) for pred in all_test_predictions]
            # elif adme_target == 'VD':
            #     bins = [0.2, 0.7, 3, 10]
            #     predictions = [torch.clamp(pred, min=0.0) for pred in all_test_predictions]
            # elif adme_target == 'CL':
            #     bins = [0.1, 1, 10, 100]
            #     predictions = [torch.clamp(pred, min=0.0) for pred in all_test_predictions]
            # pred_classes, target_classes = self.regression_to_classification(predictions, all_test_targets, bins)
            # print(pred_classes, target_classes)
            # metrics = calculate_reg_class_metrics(pred_classes, target_classes)
            # mcc_values = [metrics['MCC'] for metrics in metrics.values()]
            # real_values = sum(mcc_values) / len(mcc_values)
            # print(real_values)
            metrics = reg_metrics(all_test_predictions, all_test_targets)
            print(metrics)
            predictions = torch.cat(all_test_predictions, dim=0).numpy()
            all_targets = torch.cat(all_test_targets, dim=0).numpy()
            df = pd.DataFrame({
                f"{adme_target}": predictions[:, 0],
                f"true_{adme_target}": all_targets[:, 0]
            })
            df.to_excel(f'./output/50_new_encoder/padmet_{adme_target}.xlsx', index=False)
            exit()
            return metrics, metrics['PCC']
            # return metrics, real_values
            # print(all_test_predictions)
        elif task == "classification":
            # pred_logits = torch.cat(all_test_predictions, dim=0)
            # pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
            # binary_preds = (pred_probs > 0.5).astype(int)
            # pred_binary = (binary_preds.sum(axis=1) > 0).astype(int)
            # target_logits = torch.cat(all_test_targets, dim=0)
            # target_binary = (target_logits.sum(axis=1) > 0).int().cpu().numpy()
            # print(pred_binary.shape, target_binary.shape)
            # columns = ['Neurotoxicity', 'Carcinogenicity', 'DART', 'Celiac-toxicity', 'Hemolytic toxicity', 'Respiratory Toxicity', 'Cardiotoxin',
            #            'Cytotoxicity', 'Toxicity-unclassified', 'Acute effect', 'H-HT / DILI']
            # df = pd.DataFrame(real_binary, columns=['label'])
            # df.to_excel('./output/50/padmet_toxibtl.xlsx', index=False)
            # exit()
            res_score = calculate_multilabel_metrics(all_test_predictions, all_test_targets)
            # res_score = calculate_toxibtl_metrics(all_test_predictions, all_test_targets)
            print(res_score)
            mcc_values = [metrics['MCC'] for metrics in res_score.values()]
            return res_score, sum(mcc_values) / len(mcc_values)
            # return res_score, res_score['MCC']

    def predict(self, task, adme_target=None):
        if task == 'classification':
            # pattern = f"{task}_checkpoint_epoch_*_fold_*.pt"
            pattern = f"{task}_checkpoint_epoch_2_fold_2.pt"
            # output_data = self.config.output_save_path / 'test_toxibtl_info.txt'
            output_data = self.config.output_save_path / 'test_class_info.txt'
            dataset = self.setup_dataloaders(
                'toxicity', test_data='test', data_paths={'test_path': self.config.test_class_data_path}
                # 'toxicity', test_data='toxibtl', data_paths={'test_path': './data/toxibtl.xlsx'}
            )
            # dataset = self.setup_classification_dataloaders(test_data='toxibtl')
        else:
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_*_fold_*.pt"
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_14_fold_0.pt"  # bio
            pattern = f"{task}_{adme_target}_checkpoint_epoch_9_fold_8.pt"  # bio
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_0_fold_2.pt"  # caco2  11 0
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_6_fold_2.pt"  # vd
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_5_fold_4.pt"  # cl
            # pattern = f"{task}_{adme_target}_checkpoint_epoch_6_fold_6.pt"  # t1/2
            output_data = self.config.output_save_path / f'test_reg_info_{adme_target}.txt'
            dataset = self.setup_dataloaders(
                task_type='adme',
                test_data='test',
                target=adme_target,
                data_paths={
                    'test_path': self.config.test_reg_data_path
                }
            )
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        # checkpoint_path = self.config.model_save_path / f'{task}_checkpoint_epoch_6.pt'
        checkpoint_path_ls = list(self.config.model_save_path.glob(pattern))
        tmp_d, tmp_metrics, tmp_pa = 0, None, None
        for checkpoint_path in checkpoint_path_ls:
            self.load_checkpoint(checkpoint_path)
            metrics_tmp, tmp_data = self.predict_admet2(test_loader, task=task, adme_target=adme_target)
            if tmp_data > tmp_d:
                tmp_d = tmp_data
                tmp_metrics = metrics_tmp
                tmp_pa = checkpoint_path
        print(tmp_pa, tmp_d, tmp_metrics)
        res_dic[tmp_pa].append(tmp_d)
        with open(output_data, "w", encoding='utf-8') as f:
            f.write(f"ADME Target: {adme_target}\n")
            f.write(f"Path:{str(tmp_pa)}\n")
            f.write(f"Average:{str(tmp_d)}\n")
            f.write("All Metrics:\n")
            f.write(str(tmp_metrics))
            f.write('\n\n')

    def _run_batch(self, batch_data, task, phase, mask=None):
        forward_args = self._get_forward_args(batch_data, task, phase)
        regression_out, classification_out = self.model(**forward_args)

        if task == "regression":
            regression_targets = batch_data[7].to(self.device)
            if regression_targets.dim() == 1:
                regression_targets = regression_targets.unsqueeze(-1)
            # loss = Criterion.regression_loss(regression_out, regression_targets)
            loss = self.reg_criterion(regression_out, regression_targets)
            return loss, regression_out, regression_targets

        elif task == "classification":
            toxicity_targets = batch_data[7].to(self.device)
            # toxicity_targets = batch_data[7][:, self.cls_index].to(self.device).unsqueeze(-1)
            loss = Criterion.toxicity_loss(classification_out, toxicity_targets, mask=mask)
            # loss = self.class_criterion(classification_out, toxicity_targets)
            return loss, classification_out, toxicity_targets

        else:
            raise ValueError("Invalid task type. Choose either 'regression' or 'classification'.")


def seed_everything(seed=42):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # a = {}
    # for k, v in a.items():
    # print(k, '\t', '\t'.join(map(str, v.values())))
    seed_everything()
    base_config = Config('./utils/config.yaml')
    base_model = Padmet(n_targets=1, num_classes=11)
    base_config.model_save_path = Path('./models/50_new_encoder')
    base_config.output_save_path = Path('./output/50_new_encoder')
    trainer = ClassificationTrainer(base_model, base_config)
    # trainer.train_classification()
    # exit()
    res_dic = defaultdict(list)
    # trainer.predict(task='classification')
    # exit()
    # for _ in range(50):
    #     trainer = ClassificationTrainer(base_model, base_config)
    #     trainer.predict(task='classification')
    # len_dic = {k: len(v) for k, v in res_dic.items()}
    # mean_dic = {k: sum(v) / len(v) for k, v in res_dic.items() if len(v) > 0}
    # print(mean_dic)
    # print(len_dic)
    # exit()
    # admet_targets = ['Bioavailability', 'VD', 'T1_2', 'CL', 'CACO2']
    admet_targets = ['Bioavailability']
    # trainer.train_regression(admet_targets)
    # exit()
    for target in admet_targets:
        trainer.predict(task='regression', adme_target=target)
        print(target)
