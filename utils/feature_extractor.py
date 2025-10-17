#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
from rdkit import Chem

from utils.get_smiles import iupac_to_smiles
from utils.global_config import *
from utils.mod_iupac import add_modifications
from utils.generate_input import count_hydrogens, get_max_length, is_valid_smiles, build_tensor2


def get_len_hydrogen(smiles_ls):
    hydrogen_counts = {}
    for smiles in smiles_ls:
        mol = Chem.MolFromSmiles(smiles)
        hydrogen_counts.update(count_hydrogens(mol))

    max_length = get_max_length(smiles_ls)
    return hydrogen_counts, max_length


def convert_smiles_to_standard(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析 SMILES: {smiles}")
        return None
    Chem.RemoveStereochemistry(mol)
    standard_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

    return standard_smiles


def get_tensor(in_sequence, in_modifications):
    iupac_result = add_modifications(in_sequence, in_modifications)
    # print(iupac_result)
    smiles = iupac_to_smiles(iupac_result)
    if smiles:
        stand_smiles = convert_smiles_to_standard(smiles)
        if stand_smiles:
            is_valid = is_valid_smiles(stand_smiles)
            if is_valid:
                hydrogen_counts, max_length = get_len_hydrogen([stand_smiles])
                max_length = 50
                real_tensor = build_tensor2(stand_smiles, hydrogen_counts, max_length)
                return real_tensor
            else:
                print('Invalid SMILES', smiles)
                return None
        else:
            print('Unable to convert standard smiles', smiles)
            return None
    else:
        print('iupac to smiles error', iupac_result)
        return None


def feat_extract(modified_feats, seq_idx):
    # try:
    batch_size, seq_len, feature_dim = modified_feats.shape
    assert feature_dim == len(AMINO_ACIDS) + len(MOD_VOCAB), "特征维度不匹配，可能未正确初始化"

    sequences = []
    modifications = []

    for b in range(batch_size):
        aa_sequence = []
        mod_info = {idx: "" for idx in range(seq_len)}
        for i in range(seq_len):
            aa_idx = torch.argmax(modified_feats[b, i, :len(AMINO_ACIDS)]).item()
            aa = IDX_TO_AA.get(aa_idx, 'X')
            aa_sequence.append(aa)

            mod_vec = modified_feats[b, i, len(AMINO_ACIDS):]
            if mod_vec.sum() > 0:
                mod_idx = torch.argmax(mod_vec).item()
                if mod_idx < len(MOD_VOCAB) and mod_idx in seq_idx[aa]:
                    mod_info[i] = MOD_REVERSE.get(mod_idx, '')

        n_term_mod_idx = torch.argmax(modified_feats[b, 0, len(AMINO_ACIDS):]).item()
        if n_term_mod_idx in seq_idx['N-terminal']:
            mod_info[0] = MOD_REVERSE.get(n_term_mod_idx, '')

        c_term_mod_idx = torch.argmax(modified_feats[b, -1, len(AMINO_ACIDS):]).item()
        if c_term_mod_idx in seq_idx['C-terminal']:
            mod_info[seq_len - 1] = MOD_REVERSE.get(c_term_mod_idx, '')

        sequences.append(''.join(aa_sequence))
        modifications.append(mod_info)
    # smiles_list, masks = [], []
    smiles_list, iupac_list, masks = [], [], []

    for seq, mods in zip(sequences, modifications):
        # print(seq, mods)
        iupac = add_modifications(seq, mods)
        seq_tensor = get_tensor(seq, mods)
        if not seq_tensor:
            masks.append(False)
            continue
        masks.append(True)
        smiles_list.append(seq_tensor)
        iupac_list.append(iupac)
    # return smiles_list, torch.tensor(masks)
    return smiles_list, iupac_list, torch.tensor(masks)
    # return smiles_list, torch.tensor(masks)


if __name__ == '__main__':
    # test_sequence = "ACDE"
    # test_modifications = {idx: "" for idx in range(len(test_sequence) + 1)}
    # test_modifications[0] = 'Boc'
    # test_modifications[2] = 'Phospho'
    # test_modifications[5] = '4-Fluorobenzoyl'
    # result = add_modifications(test_sequence, test_modifications)
    print(feat_extract('ACDE'))
