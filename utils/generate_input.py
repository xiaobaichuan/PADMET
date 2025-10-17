from rdkit.Chem import rdmolops
from collections import defaultdict, deque
import hashlib
from rdkit import Chem
from Bio import SeqIO
import torch

from utils.new_split_atom import split_smiles


def count_carbons_in_path(mol, path):
    """计算路径中碳原子的数量"""
    return sum(1 for idx in path if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')


# def find_all_paths(mol, start, visited=None, path=None):
#     """递归查找所有可能的路径"""
#     if visited is None:
#         visited = set()
#     if path is None:
#         path = []
#
#     current_path = path + [start]
#     visited.add(start)
#     paths = [current_path]
#
#     # 获取当前原子的所有相邻原子
#     current_atom = mol.GetAtomWithIdx(start)
#     for neighbor in current_atom.GetNeighbors():
#         neighbor_idx = neighbor.GetIdx()
#         if neighbor_idx not in visited:
#             new_paths = find_all_paths(mol, neighbor_idx, visited.copy(), current_path)
#             paths.extend(new_paths)
#
#     return paths
def find_all_paths(mol, start, max_depth=30):
    """迭代实现路径搜索，添加化学智能剪枝"""
    stack = [(start, {start}, [start])]  # (当前原子, 已访问集合, 当前路径)
    all_paths = []
    best_carbon_count = 0  # 跟踪当前最优碳原子数

    while stack:
        current_idx, visited, path = stack.pop()

        # 剪枝策略1：当前路径已不可能超越最优解
        current_carbons = count_carbons_in_path(mol, path)
        remaining_depth = max_depth - len(path)
        if (best_carbon_count - current_carbons) > remaining_depth:
            continue

        # 剪枝策略2：优先扩展高碳路径
        path_candidates = []
        current_atom = mol.GetAtomWithIdx(current_idx)

        for neighbor in current_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited:
                # 计算邻接原子的优先级（碳原子优先）
                priority = 1 if neighbor.GetSymbol() == 'C' else 0
                path_candidates.append((priority, n_idx))

        # 按优先级排序（碳原子优先扩展）
        path_candidates.sort(reverse=True, key=lambda x: x[0])

        for _, n_idx in path_candidates:
            new_visited = visited.copy()
            new_visited.add(n_idx)
            new_path = path + [n_idx]

            # 更新最优解
            new_carbons = current_carbons + (1 if mol.GetAtomWithIdx(n_idx).GetSymbol() == 'C' else 0)
            if new_carbons > best_carbon_count:
                best_carbon_count = new_carbons

            if len(new_path) <= max_depth:
                stack.append((n_idx, new_visited, new_path))

        all_paths.append(path)

    return all_paths


# def find_longest_carbon_chain(smiles):
#     """找到包含最多碳原子的链"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError("Invalid SMILES string")
#
#     # 获取所有原子
#     all_atoms = range(mol.GetNumAtoms())
#
#     # 存储所有可能的路径
#     all_paths = []
#     max_carbons = 0
#     best_path = None
#
#     # 从每个原子开始寻找路径
#     for start in all_atoms:
#         paths = find_all_paths(mol, start)
#         for path in paths:
#             carbon_count = count_carbons_in_path(mol, path)
#             # 如果发现包含更多碳原子的路径，或相同碳原子数但更短的路径
#             if carbon_count > max_carbons or (
#                     carbon_count == max_carbons and (best_path is None or len(path) < len(best_path))):
#                 max_carbons = carbon_count
#                 best_path = path
#
#     if best_path:
#         main_chain_smiles = Chem.MolFragmentToSmiles(
#             mol,
#             atomsToUse=best_path,
#             kekuleSmiles=True,
#             canonical=True
#         )
#         return mol, best_path, main_chain_smiles, max_carbons
#
#     return mol, [], "", 0
def find_longest_carbon_chain(smiles):
    """找到包含最多碳原子的链"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # 获取所有原子
    all_atoms = range(mol.GetNumAtoms())

    # 存储所有可能的路径
    all_paths = []
    max_carbons = 0
    best_path = None

    # 从每个原子开始寻找路径
    for start in all_atoms:
        paths = find_all_paths(mol, start, max_depth=30)
        for path in paths:
            carbon_count = count_carbons_in_path(mol, path)
            # 如果发现包含更多碳原子的路径，或相同碳原子数但更短的路径
            if carbon_count > max_carbons or (
                    carbon_count == max_carbons and (best_path is None or len(path) < len(best_path))):
                max_carbons = carbon_count
                best_path = path

    if best_path:
        main_chain_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=best_path,
            kekuleSmiles=True,
            canonical=True
        )
        return mol, best_path, main_chain_smiles, max_carbons

    return mol, [], "", 0


def extract_side_chains(original_mol, main_chain):
    side_chains = {}
    fragments = []
    terminal_carbons = set()  # 存储需要从主链中移除的端点碳原子

    # 获取主链上的碳原子索引
    carbon_indices = [idx for idx in main_chain if original_mol.GetAtomWithIdx(idx).GetSymbol() == 'C']

    # 检查 carbon_indices 是否为空
    if not carbon_indices:
        print("Warning: No carbon atoms found in the main chain.")
        return side_chains, fragments, terminal_carbons  # 返回空结果，避免 IndexError

    # 为每个碳原子初始化空侧链
    for carbon_idx in carbon_indices:
        side_chains[carbon_idx] = ""

    # 首先处理端点碳原子
    start_carbon = carbon_indices[0]
    end_carbon = carbon_indices[-1]

    # 检查端点碳原子
    for terminal_idx in (start_carbon, end_carbon):
        current_atom = original_mol.GetAtomWithIdx(terminal_idx)

        # 检查端点碳原子的邻接原子
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            # 如果邻接原子是主链上的碳原子
            if (neighbor_idx in main_chain and
                    neighbor.GetSymbol() == 'C' and
                    neighbor_idx not in (start_carbon, end_carbon)):
                # 将端点碳原子作为这个邻接碳原子的侧链
                terminal_carbons.add(terminal_idx)

                # 获取端点碳原子的完整侧链
                terminal_side_chain = []
                queue = deque([terminal_idx])
                visited = {terminal_idx}

                while queue:
                    current = queue.popleft()
                    terminal_side_chain.append(current)
                    atom = original_mol.GetAtomWithIdx(current)

                    for next_neighbor in atom.GetNeighbors():
                        next_idx = next_neighbor.GetIdx()
                        if (next_idx not in visited and
                                next_idx not in main_chain):
                            visited.add(next_idx)
                            queue.append(next_idx)

                # 生成端点及其侧链的SMILES
                if terminal_side_chain:
                    bonds_to_use = []
                    for atom_idx in terminal_side_chain:
                        atom = original_mol.GetAtomWithIdx(atom_idx)
                        for bond in atom.GetBonds():
                            begin_idx = bond.GetBeginAtomIdx()
                            end_idx = bond.GetEndAtomIdx()
                            if (begin_idx in terminal_side_chain and
                                    end_idx in terminal_side_chain):
                                bonds_to_use.append(bond.GetIdx())

                    try:
                        terminal_smiles = Chem.MolFragmentToSmiles(
                            original_mol,
                            atomsToUse=terminal_side_chain,
                            bondsToUse=bonds_to_use,
                            isomericSmiles=True,
                            kekuleSmiles=True,
                            allBondsExplicit=False,
                            allHsExplicit=False
                        )
                        if terminal_smiles and terminal_smiles != "":
                            # 将端点侧链添加到邻接碳原子的侧链中
                            existing_side_chain = side_chains.get(neighbor_idx, "")
                            if existing_side_chain:
                                side_chains[neighbor_idx] = f"{existing_side_chain}.{terminal_smiles}"
                            else:
                                side_chains[neighbor_idx] = terminal_smiles
                            fragments.append(terminal_smiles)
                    except Exception as e:
                        print(f"Warning: Failed to generate SMILES for terminal carbon {terminal_idx}: {str(e)}")

    # 处理剩余的非端点碳原子的侧链
    remaining_carbons = [idx for idx in carbon_indices if idx not in terminal_carbons]
    for carbon_idx in remaining_carbons:
        current_atom = original_mol.GetAtomWithIdx(carbon_idx)
        neighbor_atoms = []

        # 收集所有非主链的邻接原子
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in main_chain:
                neighbor_atoms.append(neighbor_idx)

        if neighbor_atoms:
            try:
                # 通过BFS获取完整的侧链原子
                side_chain_atoms = set(neighbor_atoms)
                queue = deque(neighbor_atoms)
                visited = set(neighbor_atoms)
                visited.add(carbon_idx)

                while queue:
                    current = queue.popleft()
                    atom = original_mol.GetAtomWithIdx(current)

                    for neighbor in atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if (neighbor_idx not in visited and
                                neighbor_idx not in main_chain):
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)
                            side_chain_atoms.add(neighbor_idx)

                fragment_atoms = list(side_chain_atoms)

                if fragment_atoms:
                    bonds_to_use = []
                    for atom_idx in fragment_atoms:
                        atom = original_mol.GetAtomWithIdx(atom_idx)
                        for bond in atom.GetBonds():
                            begin_idx = bond.GetBeginAtomIdx()
                            end_idx = bond.GetEndAtomIdx()
                            if (begin_idx in fragment_atoms and
                                    end_idx in fragment_atoms):
                                bonds_to_use.append(bond.GetIdx())

                    fragment_smiles = Chem.MolFragmentToSmiles(
                        original_mol,
                        atomsToUse=fragment_atoms,
                        bondsToUse=bonds_to_use,
                        isomericSmiles=True,
                        kekuleSmiles=True,
                        allBondsExplicit=False,
                        allHsExplicit=False
                    )

                    if fragment_smiles and fragment_smiles != "":
                        side_chains[carbon_idx] = fragment_smiles
                        fragments.append(fragment_smiles)

            except Exception as e:
                print(f"Warning: Failed to generate SMILES for carbon {carbon_idx}: {str(e)}")

    return side_chains, fragments, terminal_carbons


def get_carbon_with_sidechain_smiles(original_mol, carbon_idx, side_chain):
    """生成碳原子和它的侧链组合的SMILES"""
    if not side_chain:
        return "C"  # 如果没有侧链，就只返回C

    # 如果有侧链，将侧链连接到碳原子上
    combined_smiles = f"C({side_chain})"
    return combined_smiles


def find_rings_in_mainchain(mol, carbon_indices):
    """识别主链中的环结构"""
    rings = []
    ring_info = mol.GetRingInfo()

    # 获取所有环
    for ring in ring_info.AtomRings():
        # 检查环中的原子是否在主链上
        main_chain_ring_atoms = [atom for atom in ring if atom in carbon_indices]
        if len(main_chain_ring_atoms) >= 2:  # 如果环中至少有2个原子在主链上
            rings.append(sorted(ring))  # 保存整个环的原子索引

    return rings


def get_ring_smiles_with_sidechains(mol, ring_atoms, side_chains):
    """获取环结构的SMILES，包括侧链"""
    # 创建一个新分子来构建完整的环+侧链结构
    ring_mol = Chem.RWMol()

    # 复制环上的原子和它们之间的键
    atom_map = {}  # 原始分子中的原子索引到新分子中的原子索引的映射

    # 首先添加环上的所有原子
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_atom = ring_mol.AddAtom(atom)
        atom_map[atom_idx] = new_atom

    # 添加环内的键
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in ring_atoms and end_idx in ring_atoms:
            ring_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())

    # 为每个环上的碳原子添加侧链
    for atom_idx in ring_atoms:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C' and atom_idx in side_chains:
            sidechain = side_chains[atom_idx]
            if sidechain:
                # 将侧链SMILES转换为分子片段并添加到环结构上
                sidechain_mol = Chem.MolFromSmiles(sidechain)
                if sidechain_mol:
                    # 将侧链连接到环上对应的碳原子
                    combined_mol = Chem.CombineMols(ring_mol, sidechain_mol)
                    ring_mol = Chem.RWMol(combined_mol)

    # 生成最终的SMILES
    try:
        Chem.SanitizeMol(ring_mol)
        ring_smiles = Chem.MolToSmiles(ring_mol, isomericSmiles=True)
    except:
        # 如果无法生成完整SMILES，返回基本环的SMILES
        ring_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=ring_atoms,
            bondsToUse=[bond.GetIdx() for bond in mol.GetBonds()
                        if bond.GetBeginAtomIdx() in ring_atoms
                        and bond.GetEndAtomIdx() in ring_atoms],
            isomericSmiles=True,
            allHsExplicit=False
        )

    # 收集环上碳原子的侧链信息用于显示
    ring_sidechains = {}
    for atom_idx in ring_atoms:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C':
            if atom_idx in side_chains and side_chains[atom_idx]:
                ring_sidechains[atom_idx] = side_chains[atom_idx]

    return ring_smiles, ring_sidechains


def assign_heteroatoms_to_nearest_carbon(mol, main_chain, side_chains, rings):
    """将主链上的杂原子分配给最近的碳原子，对氮原子有特殊规则"""
    updated_side_chains = side_chains.copy()
    main_chain_atoms = list(main_chain)

    # 收集所有可用的碳原子（包括环上的）
    available_carbons = []
    for idx in main_chain_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C':
            available_carbons.append(idx)

    def has_oxygen_double_bond(mol, carbon_idx, side_chain):
        """检查碳原子是否有=O侧链"""
        if not side_chain:
            return False
        # 检查侧链中是否包含=O
        return '=O' in side_chain

    # 找出所有主链上的非碳原子
    for i, idx in enumerate(main_chain_atoms):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            # 找到当前杂原子在主链中的位置
            current_pos = i

            # 找左侧最近的碳原子
            left_carbon_idx = None
            left_distance = float('inf')
            for j in range(current_pos - 1, -1, -1):
                if j >= 0 and mol.GetAtomWithIdx(main_chain_atoms[j]).GetSymbol() == 'C':
                    left_carbon_idx = main_chain_atoms[j]
                    left_distance = current_pos - j
                    break

            # 找右侧最近的碳原子
            right_carbon_idx = None
            right_distance = float('inf')
            for j in range(current_pos + 1, len(main_chain_atoms)):
                if mol.GetAtomWithIdx(main_chain_atoms[j]).GetSymbol() == 'C':
                    right_carbon_idx = main_chain_atoms[j]
                    right_distance = j - current_pos
                    break

            # 获取杂原子的SMILES
            hetero_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=[idx],
                bondsToUse=[],
                isomericSmiles=True,
                allHsExplicit=False
            )

            # 决定将杂原子分配给哪个碳原子
            target_carbon_idx = None

            # 对氮原子的特殊处理
            if atom.GetSymbol() == 'N' and left_carbon_idx is not None and right_carbon_idx is not None:
                left_has_oxygen = has_oxygen_double_bond(mol, left_carbon_idx,
                                                         updated_side_chains.get(left_carbon_idx, ""))
                right_has_oxygen = has_oxygen_double_bond(mol, right_carbon_idx,
                                                          updated_side_chains.get(right_carbon_idx, ""))

                # 如果一个有=O侧链而另一个没有，选择没有=O侧链的
                if left_has_oxygen and not right_has_oxygen:
                    target_carbon_idx = right_carbon_idx
                elif not left_has_oxygen and right_has_oxygen:
                    target_carbon_idx = left_carbon_idx
                else:
                    # 如果都有或都没有=O侧链，选择左边的碳原子
                    target_carbon_idx = left_carbon_idx
            else:
                # 其他杂原子的处理逻辑保持不变
                if left_carbon_idx is not None and right_carbon_idx is not None:
                    # 检查两边碳原子的侧链情况
                    left_has_sidechain = left_carbon_idx in updated_side_chains and updated_side_chains[left_carbon_idx]
                    right_has_sidechain = right_carbon_idx in updated_side_chains and updated_side_chains[
                        right_carbon_idx]

                    # 优先选择没有侧链的碳原子
                    if not left_has_sidechain and right_has_sidechain:
                        target_carbon_idx = left_carbon_idx
                    elif left_has_sidechain and not right_has_sidechain:
                        target_carbon_idx = right_carbon_idx
                    else:
                        # 如果两边都有或都没有侧链，选择距离最近的
                        if left_distance <= right_distance:
                            target_carbon_idx = left_carbon_idx
                        else:
                            target_carbon_idx = right_carbon_idx
                elif left_carbon_idx is not None:
                    target_carbon_idx = left_carbon_idx
                elif right_carbon_idx is not None:
                    target_carbon_idx = right_carbon_idx

            # 将杂原子添加到目标碳原子的侧链中
            if target_carbon_idx is not None:
                if target_carbon_idx in updated_side_chains:
                    updated_side_chains[target_carbon_idx] += hetero_smiles
                else:
                    updated_side_chains[target_carbon_idx] = hetero_smiles

    return updated_side_chains


def sha256_hash(smiles):
    """将 SMILES 字符串转化为 SHA-256 哈希值并返回十六进制表示"""
    return hashlib.sha256(smiles.encode('utf-8')).hexdigest()


def count_hydrogens(mol):
    """计算每个非氢原子实际带有的氢原子个数"""
    hydrogen_counts = {}

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            # 计算氢原子数量
            # Degree: 连接的原子数
            # ImplicitValence: 隐式价态
            # NumExplicitHs: 显式氢原子数
            hydrogen_count = atom.GetImplicitValence() + atom.GetDegree() - atom.GetNumExplicitHs()

            # 确保氢原子数量不为负值
            hydrogen_count = max(hydrogen_count, 0)

            hydrogen_counts[atom.GetIdx()] = hydrogen_count

    return hydrogen_counts


# def split_smiles(smiles):
#     """拆分SMILES，返回分割后的原子团SMILES列表"""
#     original_mol = Chem.MolFromSmiles(smiles)
#     if original_mol is None:
#         # print("无法解析SMILES字符串")
#         return []
#
#     # 找到主链
#     mol, main_chain, main_chain_smiles, carbon_count = find_longest_carbon_chain(smiles)
#     # print(f"主链SMILES (包含 {carbon_count} 个碳原子):", main_chain_smiles)
#
#     if not main_chain:
#         return []
#
#     # 找到所有主链碳原子
#     carbon_indices = [idx for idx in main_chain if original_mol.GetAtomWithIdx(idx).GetSymbol() == 'C']
#     # print("主链碳原子索引:", carbon_indices)
#
#     # 识别主链中的环结构
#     rings = find_rings_in_mainchain(original_mol, carbon_indices)
#
#     # 在原始分子上提取侧链
#     side_chains, fragments, terminal_carbons = extract_side_chains(original_mol, main_chain)
#
#     # 将主链上的杂原子分配给最近的没有侧链的碳原子（包括环上的碳原子）
#     side_chains = assign_heteroatoms_to_nearest_carbon(original_mol, main_chain, side_chains, rings)
#
#     # 存储分割后的SMILES
#     split_smiles_list = []
#
#     # 创建一个集合来存储已经处理过的环中的原子
#     processed_atoms = set()
#     processed_atoms.update(terminal_carbons)  # 添加端点碳原子到已处理集合
#
#     # 处理环结构
#     if rings:
#         # print("\n环结构信息:")
#         for i, ring in enumerate(rings, 1):
#             processed_atoms.update(ring)
#             ring_smiles, ring_sidechains = get_ring_smiles_with_sidechains(original_mol, ring, side_chains)
#             split_smiles_list.append((ring_smiles, min(ring)))  # 添加环SMILES以及环中最小的碳原子索引
#             # print(f"\n环 {i}:")
#             # print(f"  原子索引: {ring}")
#             # print(f"  环SMILES: {ring_smiles}")
#
#             # 输出环 SMILES 的 SHA-256 哈希值
#             ring_hash = sha256_hash(ring_smiles)
#             # print("  环 SMILES 的 SHA-256 哈希:", ring_hash)
#
#             if ring_sidechains:
#                 # print("  环上碳原子的侧链:")
#                 for atom_idx, sidechain in ring_sidechains.items():
#                     print(f"    碳原子 {atom_idx}: {sidechain}")
#
#                     # 处理环上碳原子的侧链
#                     sidechain_smiles = get_carbon_with_sidechain_smiles(original_mol, atom_idx, sidechain)
#                     split_smiles_list.append((sidechain_smiles, atom_idx))  # 将侧链SMILES与对应的碳原子索引一起存储
#
#     # 处理非环结构的碳原子（排除端点碳原子）
#     # print("\n非环结构的碳原子信息:")
#     linear_carbons = [idx for idx in carbon_indices if idx not in processed_atoms and idx not in terminal_carbons]
#     if linear_carbons:
#         for carbon_idx in linear_carbons:
#             atom = original_mol.GetAtomWithIdx(carbon_idx)
#             if atom.GetSymbol() == 'C':
#                 sidechain = side_chains.get(carbon_idx, "")
#                 carbon_smiles = get_carbon_with_sidechain_smiles(original_mol, carbon_idx, sidechain)
#                 # print(f"\n碳原子 {carbon_idx}:")
#                 # print(f"  SMILES: {carbon_smiles}")
#                 # if sidechain:
#                 #     print(f"  侧链: {sidechain}")
#
#                 # 输出碳原子及其侧链的 SMILES 的 SHA-256 哈希值
#                 carbon_hash = sha256_hash(carbon_smiles)
#                 # print("  碳原子及其侧链 SMILES 的 SHA-256 哈希:", carbon_hash)
#                 split_smiles_list.append((carbon_smiles, carbon_idx))  # 将非环碳原子SMILES与其索引存储
#                 # print(len(split_smiles_list))
#     else:
#         # print("  没有非环结构的碳原子")
#         pass
#
#         # 对分割后的 SMILES 按照碳原子索引的最小值排序
#     sorted_smiles_list = [smiles for smiles, _ in sorted(split_smiles_list, key=lambda x: x[1])]
#
#     return sorted_smiles_list


def get_max_length(smiles_list):
    """计算最大长度"""
    max_length = max(len(split_smiles(smiles)) for smiles in smiles_list)
    print(max_length)
    return max_length + 1 if max_length % 2 != 0 else max_length


def build_tensor(smiles_list, hydrogen_counts, max_length):
    # 为每个SMILES创建单独的tensor列表
    all_first = []
    all_second = []
    all_third = []
    all_fourth = []
    all_fifth = []
    one_hots = []

    for smiles, one_hot in smiles_list.items():
        one_hots.append(eval(one_hot))
        mol = Chem.MolFromSmiles(smiles)
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        h_count = sum(hydrogen_counts.get(atom.GetIdx(), 0) for atom in mol.GetAtoms() if atom.GetSymbol() != 'H')
        o_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        n_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        p_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P')
        s_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        group_count = len(split_smiles(smiles))

        first_interface = [c_count, h_count, o_count, n_count, p_count, s_count, group_count]
        all_first.append(first_interface)

        # 第二个接口
        second_interface = torch.zeros((max_length, 6), dtype=torch.int)
        groups = split_smiles(smiles)

        for j, group in enumerate(groups):
            if j < max_length:
                mol_group = Chem.MolFromSmiles(group)
                if mol_group:
                    counts = [0] * 6
                    for atom in mol_group.GetAtoms():
                        symbol = atom.GetSymbol()
                        idx = atom.GetIdx()
                        if symbol == 'C':
                            counts[0] += 1
                        elif symbol == 'H':
                            counts[1] += hydrogen_counts.get(idx, 0)
                        elif symbol == 'O':
                            counts[2] += 1
                        elif symbol == 'N':
                            counts[3] += 1
                        elif symbol == 'P':
                            counts[4] += 1
                        elif symbol == 'S':
                            counts[5] += 1
                    second_interface[j] = torch.tensor(counts)
        all_second.append(second_interface)

        # 第三个接口
        third_interface = torch.zeros((max_length // 2 + max_length % 2, 6), dtype=torch.int)
        for j in range(0, len(groups), 2):
            counts = [0] * 6
            for k in range(2):
                if j + k < len(groups):
                    group = groups[j + k]
                    mol_group = Chem.MolFromSmiles(group)
                    if mol_group:
                        for atom in mol_group.GetAtoms():
                            symbol = atom.GetSymbol()
                            if symbol == 'C':
                                counts[0] += 1
                            elif symbol == 'H':
                                counts[1] += hydrogen_counts.get(atom.GetIdx(), 0)
                            elif symbol == 'O':
                                counts[2] += 1
                            elif symbol == 'N':
                                counts[3] += 1
                            elif symbol == 'P':
                                counts[4] += 1
                            elif symbol == 'S':
                                counts[5] += 1
            third_interface[j // 2] = torch.tensor(counts)
        all_third.append(third_interface)

        # 第四个接口
        # 初始化 max_length 行，256 列的矩阵，数据类型为 int64
        fourth_interface = torch.zeros((max_length, 256), dtype=torch.int64)

        # 调用 split_smiles(smiles) 分割 SMILES
        groups = split_smiles(smiles)
        print(groups)

        # 遍历分割后的 SMILES 片段
        for j, group in enumerate(groups):
            if j < max_length:
                # 计算当前片段的 SHA-256 哈希值
                hash_value = sha256_hash(group)

                # 将哈希值转换为二进制字符串（256 位）
                binary_hash = bin(int(hash_value, 16))[2:].zfill(256)

                # 将 256 位二进制字符串逐位存储到矩阵的第 j 行
                for k in range(256):
                    fourth_interface[j][k] = int(binary_hash[k])

        # 如果分割片段不足 max_length，则用 256 个 '0' 填充
        for j in range(len(groups), max_length):
            # 填充 256 个 '0'
            for k in range(256):
                fourth_interface[j][k] = 0

        # 将矩阵添加到 all_fourth 列表中
        all_fourth.append(fourth_interface)

        # 第五个接口
        fifth_interface = torch.zeros((8, 6), dtype=torch.int)
        groups = split_smiles(smiles)
        group_info = []

        for group in groups:
            mol_group = Chem.MolFromSmiles(group)
            if mol_group:
                counts = [0] * 6
                for atom in mol_group.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol == 'C':
                        counts[0] += 1
                    elif symbol == 'H':
                        counts[1] += hydrogen_counts.get(atom.GetIdx(), 0)
                    elif symbol == 'O':
                        counts[2] += 1
                    elif symbol == 'N':
                        counts[3] += 1
                    elif symbol == 'P':
                        counts[4] += 1
                    elif symbol == 'S':
                        counts[5] += 1
                group_info.append(counts)

        # 填充前4个和后4个组
        for i in range(min(4, len(group_info))):
            fifth_interface[i] = torch.tensor(group_info[i])
        for i in range(min(4, len(group_info))):
            if i < len(group_info):
                fifth_interface[4 + i] = torch.tensor(group_info[-(i + 1)])
        all_fifth.append(fifth_interface)

    # 将所有tensor堆叠在一起
    first_tensor = torch.tensor(all_first, dtype=torch.float32)
    second_tensor = torch.stack(all_second).to(torch.float32)
    third_tensor = torch.stack(all_third).to(torch.float32)
    fourth_tensor = torch.stack(all_fourth).to(torch.float32)
    fifth_tensor = torch.stack(all_fifth).to(torch.float32)
    one_hot_tensor = torch.tensor(one_hots, dtype=torch.float32)

    return first_tensor, second_tensor, third_tensor, fourth_tensor, fifth_tensor, one_hot_tensor


def build_tensor2(smiles, hydrogen_counts, max_length, fourth_interface_type='sha256'):
    res = []
    mol = Chem.MolFromSmiles(smiles)
    c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    # h_count = sum(hydrogen_counts.get(atom.GetIdx(), 0) for atom in mol.GetAtoms() if atom.GetSymbol() != 'H')
    h_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    o_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    n_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    p_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P')
    s_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
    groups = split_smiles(smiles)
    group_count = len(groups)

    first_interface = [c_count, h_count, o_count, n_count, p_count, s_count, group_count]
    res.append(torch.tensor(first_interface, dtype=torch.float32))

    # 第二个接口
    second_interface = torch.zeros((max_length, 6), dtype=torch.float32)

    for j, group in enumerate(groups):
        if j < max_length:
            mol_group = Chem.MolFromSmiles(group)
            if mol_group:
                counts = [0] * 6
                for atom in mol_group.GetAtoms():
                    symbol = atom.GetSymbol()
                    counts[1] += atom.GetTotalNumHs()
                    # idx = atom.GetIdx()
                    if symbol == 'C':
                        counts[0] += 1
                    # elif symbol == 'H':
                    #     counts[1] += hydrogen_counts.get(idx, 0)
                    elif symbol == 'O':
                        counts[2] += 1
                    elif symbol == 'N':
                        counts[3] += 1
                    elif symbol == 'P':
                        counts[4] += 1
                    elif symbol == 'S':
                        counts[5] += 1
                second_interface[j] = torch.tensor(counts)
    res.append(second_interface)

    # 第三个接口
    third_interface = torch.zeros((max_length // 2 + max_length % 2, 6), dtype=torch.float32)
    for j in range(0, min(max_length, len(groups)), 2):
        counts = [0] * 6
        for k in range(2):
            if j + k < min(len(groups), max_length):
                group = groups[j + k]
                mol_group = Chem.MolFromSmiles(group)
                if mol_group:
                    for atom in mol_group.GetAtoms():
                        symbol = atom.GetSymbol()
                        counts[1] += atom.GetTotalNumHs()
                        if symbol == 'C':
                            counts[0] += 1
                        # elif symbol == 'H':
                        #     counts[1] += hydrogen_counts.get(atom.GetIdx(), 0)
                        elif symbol == 'O':
                            counts[2] += 1
                        elif symbol == 'N':
                            counts[3] += 1
                        elif symbol == 'P':
                            counts[4] += 1
                        elif symbol == 'S':
                            counts[5] += 1
        third_interface[j // 2] = torch.tensor(counts)
    res.append(third_interface)

    # 第四个接口
    # 初始化 max_length 行，256 列的矩阵，数据类型为 int64
    if fourth_interface_type == 'sha256':
        fourth_interface = torch.zeros((max_length, 256), dtype=torch.float32)
        for j, group in enumerate(groups):
            if j < max_length:
                hash_value = sha256_hash(group)
                binary_hash = bin(int(hash_value, 16))[2:].zfill(256)
                for k in range(256):
                    fourth_interface[j][k] = int(binary_hash[k])
    elif fourth_interface_type == 'ascii':
        # ASCII 编码版本
        fourth_interface = torch.zeros((max_length, 25), dtype=torch.float32)
        for j, group in enumerate(groups):
            if j < max_length:
                truncated_group = group[:25]
                ascii_values = []
                for char in truncated_group:
                    if ord(char) > 127:
                        ascii_values.append(0)
                    else:
                        ascii_values.append(ord(char))
                padding_length = 25 - len(ascii_values)
                ascii_values += [0] * padding_length
                fourth_interface[j] = torch.tensor(ascii_values, dtype=torch.int64)
    else:
        fourth_interface = torch.zeros((max_length, 256), dtype=torch.float32)
        for j, group in enumerate(groups):
            if j < max_length:
                hash_value = sha256_hash(group)
                binary_hash = bin(int(hash_value, 16))[2:].zfill(256)
                for k in range(256):
                    fourth_interface[j][k] = int(binary_hash[k])
    # 将矩阵添加到 all_fourth 列表中
    res.append(fourth_interface)

    # 第五个接口
    fifth_interface = torch.zeros((8, 6), dtype=torch.float32)
    group_info = []

    for group in groups:
        mol_group = Chem.MolFromSmiles(group)
        if mol_group:
            counts = [0] * 6
            for atom in mol_group.GetAtoms():
                symbol = atom.GetSymbol()
                counts[1] += atom.GetTotalNumHs()
                if symbol == 'C':
                    counts[0] += 1
                # elif symbol == 'H':
                #     counts[1] += hydrogen_counts.get(atom.GetIdx(), 0)
                elif symbol == 'O':
                    counts[2] += 1
                elif symbol == 'N':
                    counts[3] += 1
                elif symbol == 'P':
                    counts[4] += 1
                elif symbol == 'S':
                    counts[5] += 1
            group_info.append(counts)

    # 填充前4个和后4个组
    for i in range(min(4, len(group_info))):
        fifth_interface[i] = torch.tensor(group_info[i])
    for i in range(min(4, len(group_info))):
        if i < len(group_info):
            fifth_interface[4 + i] = torch.tensor(group_info[-(i + 1)])
    res.append(fifth_interface)

    return res


def generate(smiles_list):
    hydrogen_counts = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        hydrogen_counts.update(count_hydrogens(mol))

    # 计算最大长度
    max_length = get_max_length(smiles_list)

    # 构建Tensor
    first_tensor, second_tensor, third_tensor, fourth_tensor, fifth_tensor, one_hot_tensor = build_tensor(smiles_list, hydrogen_counts,
                                                                                                          max_length)
    # 保存Tensor
    # torch.save({
    #     'first': first_tensor,
    #     'second': second_tensor,
    #     'third': third_tensor,
    #     'fourth': fourth_tensor,
    #     'fifth': fifth_tensor
    # }, 'C:/Users/黄浩东hhd/Desktop/project/iupac/output_tensor.pt')
    # print(first_tensor)
    # print(second_tensor)
    # print(third_tensor)
    # print(fourth_tensor)
    # print(fifth_tensor)

    return first_tensor, second_tensor, third_tensor, fourth_tensor, fifth_tensor, one_hot_tensor


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        Chem.Kekulize(mol, clearAromaticFlags=True)

        # 尝试调用 find_longest_carbon_chain
        result = find_longest_carbon_chain(smiles)
        if result is None:
            return False
        return True
    except Chem.KekulizeException:
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":

    # 设置输入和输出文件路径
    fasta_file = "C:/Users/黄浩东hhd/Desktop/project/iupac/test1.fasta"
    invalid_smiles_file = "C:/Users/黄浩东hhd/Desktop/project/iupac/invalid_smiles.txt"

    smiles_list = []
    invalid_smiles = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        smiles = str(record.seq)
        if is_valid_smiles(smiles):
            smiles_list.append(smiles)
        else:
            invalid_smiles.append(smiles)

    with open(invalid_smiles_file, "w") as f:
        for smiles in invalid_smiles:
            f.write(smiles + "\n")

    # main(smiles_list)
    generate(smiles_list)
