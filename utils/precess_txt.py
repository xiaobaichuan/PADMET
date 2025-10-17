#!/user/bin/env python3
# -*- coding: utf-8 -*-
import ast
import os
import re

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


from utils import parallel_apply
from utils.generate_input import is_valid_smiles, build_tensor2, count_hydrogens, get_max_length


def get_len_hydrogen(smiles_ls):
    hydrogen_counts = {}
    for smiles in smiles_ls:
        mol = Chem.MolFromSmiles(smiles)
        hydrogen_counts.update(count_hydrogens(mol))

    max_length = get_max_length(smiles_ls)
    return hydrogen_counts, max_length


def is_structure_file(s):
    return str(s).endswith('.pdb') or str(s).endswith('.sdf')


def convert_smiles_to_standard(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析 SMILES: {smiles}")
        return None
    Chem.RemoveStereochemistry(mol)
    standard_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return standard_smiles


def seq_to_smiles(seq):
    peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(seq))
    return peptide_smiles


def is_sequence(s):
    return bool(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', s, re.IGNORECASE))


def sdf_to_smiles(sdf_file):
    if not os.path.exists(sdf_file):
        return None
    supplier = Chem.SDMolSupplier(sdf_file)
    smiles_list = []
    for mol in supplier:
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)
    return smiles_list


def pdb_to_smiles(pdb_file):
    if not os.path.exists(pdb_file):
        return None
    mol = Chem.MolFromPDBFile(pdb_file)
    if mol:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    else:
        return None


def preprocess_txt(txt_file_path):
    keys = ['ID', 'SMILES', 'Species', 'Route']
    res = []
    user_dir = os.path.dirname(txt_file_path)
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'file_name' in line:
                continue
            line = line.strip()
            row_dict = dict(zip(keys, line.split()))
            row_dict['Species'], row_dict['Route'] = row_dict['Species'].lower(), row_dict['Route'].lower()
            if row_dict['SMILES'].endswith('.pdb'):
                smiles = pdb_to_smiles(os.path.join(user_dir, row_dict['SMILES']))
                smiles_std = [convert_smiles_to_standard(smiles)]
            elif row_dict['SMILES'].endswith('.sdf'):
                smiles = sdf_to_smiles(os.path.join(user_dir, row_dict['SMILES']))
                smiles_std = convert_smiles_to_standard(smiles)
            elif is_sequence(row_dict['SMILES']):
                smiles_std = [convert_smiles_to_standard(seq_to_smiles(row_dict['SMILES']))]
            else:
                smiles_std = [convert_smiles_to_standard(row_dict['SMILES'])]
            row_dict['SMILES'] = smiles_std
            res.append(row_dict)
    return res


def preprocess_txt_opt(txt_file_path):
    result = {}
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            key, value = line.split(': ')
            if 'Weights' in line:
                match = re.findall(r'([\d\.]+)\s*\(([\w\/]+)\)', line)
                for value, key in match:
                    result[key] = float(value)
            elif 'range' in line:
                range_values = re.findall(r'\(([^)]+)\)', line)[0]
                low, high = map(float, range_values.split(','))
                result[key] = (low, high)
            else:
                result[key.split(' ')[0]] = value
    real_config = {
        'naked_sequence': result['Sequence'],
        'Species': get_one_hot(result.get('Subject', 'ND'), species_categories),
        'Route': get_one_hot(result.get('Route', 'ND'), route_categories),
        'batch_size': 256,
        'structural': {
        },
        'toxicity_weights': result.get('Toxicity', 10) * 11,
        'penalty_folds': 5.0,
        'adme_weights': [result.get('BA', 1.0), result.get('VD', 1.0), result.get('T1/2', 1.0), result.get('CL', 1.0)],
        'learning_rate': 2e-4,
        'ADME_RANGES': [result.get('Bioavailability range', (0.2, 1.0)), result.get('VD range', (0.1, 1.0)), result.get('T1/2 range', (60.0, 100000.0)), result.get('Cl range', (0.0, 10.0))]
    }
    return real_config


def get_one_hot(lab, categories):
    one_hot = [0] * len(categories)
    if lab in categories:
        one_hot[categories.index(lab.capitalize())] = 1
        return one_hot

    return one_hot


species_categories = ["Mouse", "Dog", "Rat", "Monkey", "Rabbit", "Human", "Guinea-pig", "Cat", "Duck", "Quail", "Hamster", "Pig", "Sheep", "Chicken", "Pigeon", "Squirrel", "Turkey", "Bird",
                      "Frog", "In-vitro", "Rodent", "Mammalian", "Invitro", "Yeast", "Bovine", "In-vivo", "Tilapia", "Horse"]
route_categories = ["Intraperitoneal", "Intravenous", "Intramuscular", "Oral", "Subcutaneous", "Intestinal", "Intracrebral", "Intraarterial", "Skin", "Intraspinal", "Intratracheal", "Rectal",
                    "Inhalation"]


def txt_df(res):
    df = pd.DataFrame(res)
    df['Species_one_hot'] = df['Species'].apply(lambda x: get_one_hot(x, species_categories))
    df['Route_one_hot'] = df['Route'].apply(lambda x: get_one_hot(x, route_categories))
    df = df.explode('SMILES')
    df["SMILES"] = df['SMILES'].apply(convert_smiles_to_standard)
    df = df[df['SMILES'].astype(bool)]
    smiles_ls = df['SMILES'].tolist()
    valid_mask = parallel_apply(df['SMILES'], is_valid_smiles, n_jobs=3, desc='Validating SMILES')
    df = df[np.array(valid_mask)]
    hydrogen_counts, _ = get_len_hydrogen(smiles_ls)
    max_length = 50
    print('max_length.......', max_length)

    df['smiles_tensor'] = parallel_apply(
        df['SMILES'],
        build_tensor2,
        n_jobs=4,
        desc=f'building smiles tensor, max_length={max_length}',
        hydrogen_counts=hydrogen_counts,
        max_length=max_length
    )
    species_tensor = torch.tensor(df['Species_one_hot'].tolist(), dtype=torch.float)
    route_tensor = torch.tensor(df['Route_one_hot'].tolist(), dtype=torch.float)
    df['smiles_tensor'] = df['smiles_tensor'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    smiles_ls_tensor = df['smiles_tensor'].tolist()
    return {
        'species_tensor': species_tensor,
        'route_tensor': route_tensor,
        'smiles_ls_tensor': smiles_ls_tensor,
    }


if __name__ == '__main__':
    txt_file_path = '../user_data/20251012145238/summary_table.txt'
    # data = preprocess_txt(txt_file_path)
    # txt_df(data)
    config = preprocess_txt_opt(r'..\user_data\20251012203553\summary_table.txt')
    print(config)
