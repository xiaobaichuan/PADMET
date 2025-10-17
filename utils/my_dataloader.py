#!/user/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, species, route, smiles_ls):
        self.species = species
        self.route = route
        self.smiles_ls = smiles_ls

    def __len__(self):
        return len(self.species)

    def __getitem__(self, i):
        features = (self.species[i], self.route[i], self.smiles_ls[i][0], self.smiles_ls[i][1], self.smiles_ls[i][2], self.smiles_ls[i][3], self.smiles_ls[i][4])
        return features
