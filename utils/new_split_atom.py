import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolops, AllChem, Draw, Descriptors

# import matplotlib.pyplot as plt
from collections import defaultdict
import copy


# 抑制RDKit的警告日志
RDLogger.DisableLog('rdApp.*')


def split_smiles2(smiles):
    """
    将多肽SMILES拆分为独立的氨基酸残基
    
    Args:
        smiles (str): 多肽的SMILES字符串
    
    Returns:
        dict: 包含拆分结果和统计信息的字典
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return {'cleaned_residues': [], 'amide_bond_count': 0, 'alpha_carbon_count': 0}

    # 记录原始结构
    original_mol = copy.deepcopy(mol)

    # 1. 找到所有酰胺键（主链连接点）和α碳数量
    amide_bonds, alpha_carbon_count = find_amide_bonds(mol)
    if not amide_bonds:
        print("未找到酰胺键，可能不是多肽或是单一氨基酸")
        return {'cleaned_residues': [smiles], 'amide_bond_count': 0, 'alpha_carbon_count': 0}

    print(f"找到 {len(amide_bonds)} 个酰胺键和 {alpha_carbon_count} 个α碳原子")

    # 2. 找到需要断开的环连接
    cycle_bonds = find_cycle_bonds_to_break(mol, amide_bonds)
    print(f"找到 {len(cycle_bonds)} 个需要断开的环连接")

    # 3. 找到二硫键等侧链间连接
    sidechain_bonds = find_sidechain_bonds(mol)
    print(f"找到 {len(sidechain_bonds)} 个侧链间连接")

    # 4. 合并所有需要断开的键
    bonds_to_break = amide_bonds + cycle_bonds + sidechain_bonds

    # 如果没有找到需要断开的键，返回原始SMILES
    if not bonds_to_break:
        print("未找到需要断开的键")
        return {'cleaned_residues': [smiles], 'amide_bond_count': 0, 'alpha_carbon_count': 0}

    # 5. 断开所有标记的键
    fragments = rdmolops.FragmentOnBonds(mol, bonds_to_break, addDummies=False)

    # 6. 获取片段的SMILES
    fragment_smiles = Chem.MolToSmiles(fragments, isomericSmiles=True).split('.')

    # 7. 清理片段SMILES，去除冗余信息
    cleaned_residues = clean_fragments(fragment_smiles)

    # 8. 可视化原始结构和拆分结果
    # visualize_split(original_mol, fragments)

    return {
        'cleaned_residues': cleaned_residues,
        'amide_bond_count': len(amide_bonds),
        'alpha_carbon_count': alpha_carbon_count
    }


def split_smiles(smiles):
    """
    将多肽SMILES拆分为独立的氨基酸残基

    Args:
        smiles (str): 多肽的SMILES字符串

    Returns:
        dict: 包含拆分结果和统计信息的字典
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return []

    # 记录原始结构
    original_mol = copy.deepcopy(mol)

    # 1. 找到所有酰胺键（主链连接点）和α碳数量
    amide_bonds, alpha_carbon_count = find_amide_bonds(mol)
    if not amide_bonds:
        print("未找到酰胺键，可能不是多肽或是单一氨基酸")
        return [smiles]

    print(f"找到 {len(amide_bonds)} 个酰胺键和 {alpha_carbon_count} 个α碳原子")

    # 2. 找到需要断开的环连接
    cycle_bonds = find_cycle_bonds_to_break(mol, amide_bonds)
    print(f"找到 {len(cycle_bonds)} 个需要断开的环连接")

    # 3. 找到二硫键等侧链间连接
    sidechain_bonds = find_sidechain_bonds(mol)
    print(f"找到 {len(sidechain_bonds)} 个侧链间连接")

    # 4. 合并所有需要断开的键
    bonds_to_break = amide_bonds + cycle_bonds + sidechain_bonds

    # 如果没有找到需要断开的键，返回原始SMILES
    if not bonds_to_break:
        print("未找到需要断开的键")
        return [smiles]

    # 5. 断开所有标记的键
    fragments = rdmolops.FragmentOnBonds(mol, bonds_to_break, addDummies=False)

    # 6. 获取片段的SMILES
    fragment_smiles = Chem.MolToSmiles(fragments, isomericSmiles=True).split('.')

    # 7. 清理片段SMILES，去除冗余信息
    cleaned_residues = clean_fragments(fragment_smiles)

    # 8. 可视化原始结构和拆分结果
    # visualize_split(original_mol, fragments)

    return cleaned_residues


def find_amide_bonds(mol):
    """找到所有酰胺键及其变体，返回键索引列表和α碳数量
    
    包括以下类型：
    1. 标准酰胺键 [CX3](=[OX1])[NX3]
    2. 季铵化类似酰胺键 [CX3](=[OX1])[N+X4]
    3. 质子化羟胺基团 [CX3](=[OH+])[NX3,N+X4]
    4. 碳正离子类似结构 [C+]([OX1])([NX3])
    5. 碳负离子类似结构 [C-]([OX1])([NX3])
    
    返回:
        tuple: (amide_bonds列表, alpha_carbon_count)
    """
    amide_bonds = []
    alpha_carbons = set()  # 使用集合避免重复计数

    # 定义所有酰胺键变体的SMARTS模式
    patterns = [
        # 标准酰胺键
        ("standard", "[CX3](=[OX1])[NX3]"),
        # 季铵化类似酰胺键
        ("quaternary", "[CX3](=[OX1])[N+X4]"),
        # 质子化羟胺基团
        ("protonated_hydroxylamine", "[CX3](=[OH1+])[NX3,N+X4]"),
        # 碳正离子类似结构
        ("carbocation", "[C+1]([OX1])[NX3]"),
        # 碳负离子类似结构
        ("carbanion", "[C-1]([OX1])[NX3]")
    ]

    for name, smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if not pattern:
            continue

        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # 根据模式不同，原子索引位置可能不同
            if name in ["standard", "quaternary", "protonated_hydroxylamine"]:
                c_idx, o_idx, n_idx = match
            else:  # 对于碳离子结构，顺序是 C, O, N
                c_idx, o_idx, n_idx = match

            # 找到C-N键
            bond = mol.GetBondBetweenAtoms(c_idx, n_idx)
            if bond is None:
                continue

            # 检查这个酰胺键是否在分子末端
            n_atom = mol.GetAtomWithIdx(n_idx)
            is_terminal = True

            # 检查氮原子是否连接了其他非氢原子（除了当前碳原子）
            for neighbor in n_atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx != c_idx and neighbor.GetSymbol() != 'H':
                    is_terminal = False
                    break

            # 如果不是末端酰胺键，则添加到拆分列表
            if not is_terminal and bond.GetIdx() not in amide_bonds:
                amide_bonds.append(bond.GetIdx())

                # 识别α碳（连接到N但不是酰胺键C的碳原子）
                for neighbor in n_atom.GetNeighbors():
                    if (neighbor.GetSymbol() == 'C' and
                            neighbor.GetIdx() != c_idx and
                            not neighbor.GetIsAromatic()):  # 排除芳香碳
                        alpha_carbons.add(neighbor.GetIdx())

    return amide_bonds, len(alpha_carbons)


# [保持其他辅助函数不变：find_cycle_bonds_to_break, is_sidechain_backbone_connection,
# identify_backbone_atoms, find_sidechain_bonds, clean_fragments, is_valid_residue, 
# visualize_split]

def find_cycle_bonds_to_break(mol, amide_bonds):
    """
    找到需要断开的环连接键
    - 断开主链环（头尾环化）
    - 断开侧链与主链形成的环
    - 保留残基内部的环（如芳香环）
    """
    cycle_bonds = []

    # 获取所有环
    rings = Chem.GetSSSR(mol)

    # 识别主链原子
    backbone_atoms = identify_backbone_atoms(mol, amide_bonds)

    # 检查每个环
    for ring in rings:
        ring_bonds = mol.GetRingInfo().AtomRings()
        # ring_bonds = [bond.GetIdx() for bond in ring]

        # 如果环中包含酰胺键，说明这可能是一个主链环或侧链-主链环
        if any(bond_idx in amide_bonds for bond_idx in ring_bonds):
            # 找出环中不是酰胺键的键
            for bond_idx in ring_bonds:
                if bond_idx not in amide_bonds and bond_idx not in cycle_bonds:
                    bond = mol.GetBondWithIdx(bond_idx)
                    begin_atom_idx = bond.GetBeginAtomIdx()
                    end_atom_idx = bond.GetEndAtomIdx()

                    # 检查这个键是否连接了两个主链原子
                    if (begin_atom_idx in backbone_atoms and
                            end_atom_idx in backbone_atoms):
                        # 这可能是头尾环化的键
                        cycle_bonds.append(bond_idx)

                    # 检查这个键是否连接了一个主链原子和一个非主链原子
                    elif (begin_atom_idx in backbone_atoms or
                          end_atom_idx in backbone_atoms):
                        # 这可能是侧链-主链连接
                        # 但我们需要确保这不是残基内部的环（如脯氨酸的环）

                        # 简单检查：如果这个键是C-C键或C-N键，且不是酰胺键的一部分，
                        # 那么它可能是侧链-主链连接
                        begin_atom = bond.GetBeginAtom()
                        end_atom = bond.GetEndAtom()

                        if is_sidechain_backbone_connection(bond, backbone_atoms):
                            cycle_bonds.append(bond_idx)

    return cycle_bonds


def is_sidechain_backbone_connection(bond, backbone_atoms):
    """判断一个键是否是侧链-主链连接而不是残基内部的环"""
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_idx = begin_atom.GetIdx()
    end_idx = end_atom.GetIdx()

    # 如果一端在主链上，另一端不在
    if ((begin_idx in backbone_atoms and end_idx not in backbone_atoms) or
            (begin_idx not in backbone_atoms and end_idx in backbone_atoms)):

        # 排除可能是残基内部环的情况
        # 例如，脯氨酸的五元环含有主链N和侧链C的连接

        # 如果是C-N键，需要进一步检查
        if ((begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N') or
                (begin_atom.GetSymbol() == 'N' and end_atom.GetSymbol() == 'C')):

            # 检查这个键是否是酰胺键的一部分
            if begin_atom.GetSymbol() == 'C':
                c_atom = begin_atom
            else:
                c_atom = end_atom

            # 检查碳原子是否连接双键氧（C=O）
            has_carbonyl = False
            for neighbor in c_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    bond = c_atom.GetOwningMol().GetBondBetweenAtoms(
                        c_atom.GetIdx(), neighbor.GetIdx())
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        has_carbonyl = True
                        break

            # 如果碳原子连接羰基氧，这可能是酰胺键的一部分，不是侧链-主链连接
            if has_carbonyl:
                return False

        # 其他类型的键可能是侧链-主链连接
        return True

    return False


def identify_backbone_atoms(mol, amide_bonds):
    """识别多肽主链上的原子"""
    backbone_atoms = set()

    # 从酰胺键开始识别主链
    for bond_idx in amide_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        c_atom_idx = bond.GetBeginAtomIdx()
        n_atom_idx = bond.GetEndAtomIdx()

        # 确保C是碳原子，N是氮原子
        c_atom = mol.GetAtomWithIdx(c_atom_idx)
        n_atom = mol.GetAtomWithIdx(n_atom_idx)

        if c_atom.GetSymbol() != 'C':
            c_atom_idx, n_atom_idx = n_atom_idx, c_atom_idx
            c_atom, n_atom = n_atom, c_atom

        # 添加酰胺键的C和N原子到主链
        backbone_atoms.add(c_atom_idx)
        backbone_atoms.add(n_atom_idx)

        # 添加羰基氧（C=O）
        for neighbor in c_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(c_atom_idx, neighbor.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    backbone_atoms.add(neighbor.GetIdx())

        # 添加氨基氢（N-H）
        for neighbor in n_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                backbone_atoms.add(neighbor.GetIdx())

        # 尝试找到α碳（连接到N但不是酰胺键C的碳原子）
        for neighbor in n_atom.GetNeighbors():
            if (neighbor.GetSymbol() == 'C' and
                    neighbor.GetIdx() != c_atom_idx):
                backbone_atoms.add(neighbor.GetIdx())

                # 添加α碳上的氢原子
                for h_neighbor in neighbor.GetNeighbors():
                    if h_neighbor.GetSymbol() == 'H':
                        backbone_atoms.add(h_neighbor.GetIdx())

    return backbone_atoms


from rdkit import Chem


def find_sidechain_bonds(mol):
    """找到需要断开的侧链间连接，返回键索引列表
    
    包括以下类型：
    1. 二硫键 (S-S)
    2. 配位键 (金属与杂原子之间的键)
    3. 酯键 (O=C-O-C，排除羧酸)
    4. 磷酸酯键 (P-O-C)
    """
    sidechain_bonds = []
    bond_indices = set()  # 用于去重

    # 预定义的金属原子序数列表
    METAL_ATOMIC_NUMBERS = {
        3, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
        69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
        87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103
    }

    # 定义所有需要断开的侧链连接模式
    patterns = [
        # 格式: (类型名称, SMARTS模式, 需要记录的键的原子对)
        ("disulfide", "[SX2]-[SX2]", [(0, 1)]),  # 精确匹配S-S单键
        ("ester", "[#6](=[OX1])-[OX2]-[#6]", [(1, 2)]),  # 只匹配C(=O)-O-C部分
        ("phosphate_ester", "[PX4](-[OX1])(-[OX1])(-[OX2]-[#6])", [(2, 3)]),  # 磷酸酯键 P-O-C
        ("coordination", "[#6,#7,#8][#6,#7,#8]", []),  # 配位键(由后续处理)
    ]

    for name, smarts, bond_pairs in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if not pattern:
            continue

        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # 处理预定义的键对
            for i, j in bond_pairs:
                if i >= len(match) or j >= len(match):
                    continue

                bond = mol.GetBondBetweenAtoms(match[i], match[j])
                if bond and bond.GetIdx() not in bond_indices:
                    # 对酯键进行额外验证
                    if name == "ester":
                        # 确保不是羧酸 (O连接的C不能有=O或O-H)
                        o_atom = mol.GetAtomWithIdx(match[1])
                        c_atom = mol.GetAtomWithIdx(match[2])

                        # 检查O原子是否连接H (羧酸情况)
                        has_h = any(n.GetSymbol() == 'H' for n in o_atom.GetNeighbors())

                        # 检查C原子是否连接其他O (酸酐情况)
                        has_other_o = sum(1 for n in c_atom.GetNeighbors()
                                          if n.GetSymbol() == 'O' and n.GetIdx() != match[1])

                        if not has_h and not has_other_o:
                            sidechain_bonds.append(bond.GetIdx())
                            bond_indices.add(bond.GetIdx())
                    else:
                        sidechain_bonds.append(bond.GetIdx())
                        bond_indices.add(bond.GetIdx())

            # 特殊处理配位键
            if name == "coordination":
                for i, atom_idx in enumerate(match):
                    atom = mol.GetAtomWithIdx(atom_idx)
                    # 检查是否是金属原子
                    if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS:
                        # 查找金属原子的所有键
                        for bond in atom.GetBonds():
                            neighbor = bond.GetOtherAtom(atom)
                            # 检查配位键特征 (金属与N,O,S等杂原子)
                            if neighbor.GetAtomicNum() in (7, 8, 16):
                                if bond.GetIdx() not in bond_indices:
                                    sidechain_bonds.append(bond.GetIdx())
                                    bond_indices.add(bond.GetIdx())

    return sidechain_bonds


# def find_sidechain_bonds(mol):
# """找到需要断开的侧链间连接，如二硫键"""
# sidechain_bonds = []

# # 寻找二硫键（S-S）
# ss_pattern = Chem.MolFromSmarts("S-S")
# if ss_pattern:
# matches = mol.GetSubstructMatches(ss_pattern)
# for match in matches:
# s1_idx, s2_idx = match
# # 找到S-S键
# for bond in mol.GetBonds():
# if ((bond.GetBeginAtomIdx() == s1_idx and bond.GetEndAtomIdx() == s2_idx) or
# (bond.GetBeginAtomIdx() == s2_idx and bond.GetEndAtomIdx() == s1_idx)):
# sidechain_bonds.append(bond.GetIdx())

# 寻找配位键
# 寻找酯键（O=C-O）
# ester_pattern = Chem.MolFromSmarts("O=C-O")
# if ester_pattern:
# matches = mol.GetSubstructMatches(ester_pattern)
# for match in matches:
# c_idx, o1_idx, o2_idx = match  # 酯键中的原子索引
# # 找到酯键
# for bond in mol.GetBonds():
# if ((bond.GetBeginAtomIdx() == c_idx and bond.GetEndAtomIdx() == o1_idx) or
# (bond.GetBeginAtomIdx() == o1_idx and bond.GetEndAtomIdx() == c_idx) or
# (bond.GetBeginAtomIdx() == o1_idx and bond.GetEndAtomIdx() == o2_idx) or
# (bond.GetBeginAtomIdx() == o2_idx and bond.GetEndAtomIdx() == o1_idx)):
# sidechain_bonds.append(bond.GetIdx())

# 这里可以添加其他类型的侧链间连接检测
# 例如，其他类型的交联

# return sidechain_bonds

def clean_fragments(fragment_smiles):
    """清理片段SMILES，去除冗余信息"""
    cleaned_residues = []

    for frag in fragment_smiles:
        if not frag.strip():
            continue

        # 尝试重新解析片段以确保有效
        try:
            frag_mol = Chem.MolFromSmiles(frag)
            if frag_mol:
                # 规范化SMILES表示
                cleaned_smiles = Chem.MolToSmiles(frag_mol, isomericSmiles=True)
                cleaned_residues.append(cleaned_smiles)

                # # 检查是否是有效的氨基酸残基
                # if is_valid_residue(frag_mol):
                # cleaned_residues.append(cleaned_smiles)
                # print(f"✅ 有效残基: {cleaned_smiles}")
                # else:
                # print(f"⚠️ 跳过无效残基: {cleaned_smiles}")
        except Exception as e:
            print(f"⚠️ 片段处理错误: {str(e)}")

    return cleaned_residues


def is_valid_residue(mol):
    """检查分子是否可能是有效的氨基酸残基"""
    # 简单检查：分子应该至少包含一个氮原子和一个碳原子
    has_nitrogen = False
    has_carbon = False

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            has_nitrogen = True
        elif atom.GetSymbol() == 'C':
            has_carbon = True

        if has_nitrogen and has_carbon:
            return True

    return False


def visualize_split(original_mol, fragments):
    """可视化原始结构和拆分结果"""
    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        img1 = Draw.MolToImage(original_mol, size=(300, 300))
        plt.imshow(img1)
        plt.title("原始多肽结构")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        img2 = Draw.MolToImage(fragments, size=(300, 300))
        plt.imshow(img2)
        plt.title("拆分后的残基")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"可视化出错: {str(e)}")


def test_split_peptide():
    """测试拆分函数"""
    test_peptides = [
        'NC(=O)CCC1NC(=O)C(Cc2ccccc2)NC(=O)C(Cc2ccc(O)cc2)NC(=O)CCSSCC(C(=O)N2CCCC2C(=O)NC(CCCN=C(N)N)C(=O)NCC(N)=O)NC(=O)C(CC(N)=O)NC1=O'
        # "N[C@@H](CC1=CNC=N1)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(O)=O)C(=O)NCC(=O)N[C@@H]([C@H](O)C)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]([C@H](O)C)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](CCC(O)=O)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N"
    ]

    for i, smiles in enumerate(test_peptides, 1):
        print(f"{'=' * 50}")
        print(f"测试 {i}: {smiles}")
        print(f"{'=' * 50}")

        result = split_smiles2(smiles)
        print(f"\n拆分结果 ({len(result['cleaned_residues'])} 个残基):")
        for j, residue in enumerate(result['cleaned_residues'], 1):
            print(f"  {j}. {residue}")

        print(f"酰胺键数量: {result['amide_bond_count']}")
        print(f"α碳原子数量: {result['alpha_carbon_count']}\n")


def convert_smiles_to_standard(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析 SMILES: {smiles}")
        return None
    Chem.RemoveStereochemistry(mol)
    standard_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

    return standard_smiles


def molecular_weight(smiles):
    try:
        # 将 SMILES 转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无效的 SMILES 表示")

        # 计算分子量
        mol_weight = Descriptors.MolWt(mol)
        return mol_weight
    except Exception as e:
        print(f"错误: {e}")
        return False


if __name__ == "__main__":
    """
    {
        'cleaned_residues': cleaned_residues,
        'amide_bond_count': len(amide_bonds),
        'alpha_carbon_count': alpha_carbon_count
    }
    """
    from utils import parallel_apply
    pa_ls = ['test_adme', 'train_adme']
    for pa in pa_ls:
        df = pd.read_excel(f'./data/{pa}.xlsx')
        df["SMILES"] = df['SMILES'].apply(convert_smiles_to_standard)
        df = df[df['SMILES'].astype(bool)]
        res = parallel_apply(df['SMILES'], split_smiles2, desc='split')
        # print(res)
        df_multi = pd.DataFrame(res)
        df = pd.concat([df, df_multi], axis=1)
        # df[['cleaned_residues', 'amide_bond_count', 'alpha_carbon_count']] = parallel_apply(df['SMILES'], split_smiles, desc='split')
        df['len'] = df['cleaned_residues'].apply(len)
        df['MV'] = df['SMILES'].apply(molecular_weight)
        df.to_excel(f'./{pa}_test.xlsx', index=False)
    # test_split_peptide()
