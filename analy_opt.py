import argparse
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List
import logging
import warnings

from models.admet_predictor import discriminator
from models.real_scarlett import Scarlett
from utils.feature_extractor import feat_extract, add_modifications
from utils.global_config import *
from utils.precess_txt import preprocess_txt_opt

# ============= 配置日志 =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
batch_size = 256


# ============= ADME标准化函数 =============
def normalize_adme(adme_scores: torch.Tensor) -> torch.Tensor:
    """统一使用log(1+x)变换处理ADME指标"""
    return torch.log(1 + adme_scores)


def denormalize_adme(normalized_scores: torch.Tensor) -> torch.Tensor:
    """反变换回原始值用于打印"""
    return torch.exp(normalized_scores) - 1


# ============= 约束掩码生成器实现 =============
class ConstraintMasker:
    def __init__(self, constraints_config: dict):
        """
        初始化约束掩码生成器
        
        Args:
            constraints_config: 约束配置字典
            示例:
            constraints_config = {
                'residue': {'S': ['Phospho'], 'K': ['Ubiquitin']},
                'position': {
                    'N-terminal': ['Acetylation'],
                    'C-terminal': ['Amidation']
                }
            }
        """
        self.constraints_config = constraints_config
        self.residue_constraints = {}
        for aa, mods in constraints_config.get('residue', {}).items():
            self.residue_constraints[aa] = [MOD_VOCAB[m] for m in mods]
            # self.seq_idx[aa] = [for m in mods]
        self.position_constraints = {}
        for pos, mods in constraints_config.get('position', {}).items():
            self.position_constraints[pos] = [MOD_VOCAB[m] for m in mods]

    def get_mask(self, aa_seq: torch.Tensor) -> torch.Tensor:
        """
        生成约束掩码
        
        Args:
            aa_seq: [batch_size, seq_len] 氨基酸序列索引张量
            
        Returns:
            mask: [batch_size, seq_len, mod_types] 布尔掩码张量
        """
        batch_size, seq_len = aa_seq.shape
        mask = torch.zeros(batch_size, seq_len, len(MOD_VOCAB),
                           dtype=torch.bool, device=aa_seq.device)

        # 处理残基级别约束
        for b in range(batch_size):
            # print(b)
            for i in range(seq_len):
                # print(i)
                aa_idx = aa_seq[b, i].item()
                aa = IDX_TO_AA.get(aa_idx, 'X')
                flag = list(set(self.residue_constraints[aa]) - set(self.position_constraints['N-terminal']))
                flag = list(set(flag) - set(self.position_constraints['C-terminal']))
                mask[b, i, flag] = True
                # if aa in self.residue_constraints:
                # # print(self.residue_constraints[aa])

        # 处理N端约束
        if 'N-terminal' in self.position_constraints:
            mask[:, 0, self.position_constraints['N-terminal']] = True

        # # 处理C端约束
        if 'C-terminal' in self.position_constraints:
            flag = list(set(self.position_constraints['C-terminal']) & set(self.residue_constraints[aa]))
            mask[:, -1, flag] = True

        # 考虑特殊情况
        mask[:, :, MOD_VOCAB['No-mod']] = True
        # mask[:, :, MOD_VOCAB['O4-sulfo']] = False
        # mask[:, :, MOD_VOCAB['O4-ethyl']] = False
        # mask[:, :, MOD_VOCAB["N'-Carbamoylation"]] = False

        # seq_idx = self.residue_constraints | self.position_constraints
        seq_idx = self.residue_constraints.copy()
        seq_idx.update(self.position_constraints)
        # print(mask)
        return mask, seq_idx


# ============= 修饰应用函数 =============
def apply_modifications(aa_sequences: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
    """
    将修饰应用到氨基酸序列并返回特征张量
    
    Args:
        aa_sequences: [batch_size, seq_len] 氨基酸索引张量
        actions: [batch_size, seq_len] 修饰类型索引张量
    
    Returns:
        modified_feats: [batch_size, seq_len, feature_dim] 特征张量
    """
    batch_size, seq_len = aa_sequences.shape
    feature_dim = len(AMINO_ACIDS) + len(MOD_VOCAB)
    modified_feats = torch.zeros(
        (batch_size, seq_len, feature_dim),
        device=aa_sequences.device
    )

    for b in range(batch_size):
        for pos in range(seq_len):
            aa_idx = aa_sequences[b, pos].item()
            mod_idx = actions[b, pos].item()

            # 设置氨基酸one-hot编码
            if 0 <= aa_idx < len(AMINO_ACIDS):
                modified_feats[b, pos, aa_idx] = 1.0

                # 添加修饰特征
                if mod_idx < len(MOD_VOCAB):
                    modified_feats[b, pos, len(AMINO_ACIDS) + mod_idx] = 1.0
            else:
                # 对于无效的氨基酸索引，使用特殊标记
                modified_feats[b, pos, -1] = 1.0

    return modified_feats


class ModGenerator(nn.Module):
    def __init__(self, seq_len: int):
        """
        初始化修饰生成器
        
        Args:
            seq_len: 序列长度
        """
        super().__init__()
        self.seq_len = seq_len

        # 氨基酸嵌入层（新增）
        self.aa_embedding = nn.Embedding(len(AMINO_ACIDS), 128)

        # 序列编码层
        self.gru = nn.GRU(128, 256, num_layers=2,
                          bidirectional=True, batch_first=True)

        # 多尺度特征融合
        self.conv_branch = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=5, padding=2)
        )

        # 修饰预测头
        self.mod_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, len(MOD_VOCAB))
        )

        # 价值预估头（修改为输出1维标量）
        self.value_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, aa_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            aa_indices: [batch_size, seq_len] 氨基酸序列索引张量
            
        Returns:
            mod_logits: [batch_size, seq_len, mod_types] 修饰概率对数
            value_pred: [batch_size] 价值预测（标量）
        """
        # 氨基酸序列嵌入
        embedded = self.aa_embedding(aa_indices)  # [batch_size, seq_len, 128]

        # 序列编码
        pos_out, _ = self.gru(embedded)  # [batch_size, seq_len, 512]

        # 特征融合
        conv_input = pos_out.transpose(1, 2)  # [batch_size, 512, seq_len]
        conv_feat = self.conv_branch(conv_input)  # [batch_size, 128, seq_len]
        conv_feat = conv_feat.transpose(1, 2)  # [batch_size, seq_len, 128]

        # 生成修饰概率
        mod_logits = self.mod_head(conv_feat)  # [batch_size, seq_len, mod_types]

        # 价值预测（全局平均池化 -> 标量）
        global_rep = pos_out.mean(dim=1)  # [batch_size, 512]
        value_pred = self.value_head(global_rep).squeeze(-1)  # [batch_size]

        return mod_logits, value_pred


# ============= 训练系统 =============
class ADMETTrainer:
    def __init__(self, config: dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            示例:
            config = {
                'seq_len': 15,
                'structural': {
                    'residue': {'S': ['Phospho'], 'K': ['Ubiquitin']},
                    'position': {
                        'N-terminal': ['Acetylation'],
                        'C-terminal': ['Amidation']
                    }
                },
                'toxicity_weights': [1.0, 0.8, 1.2, 0.5] + [1.0]*12,
                'learning_rate': 2e-4
            }
        """
        self.generator = ModGenerator(config['seq_len'])
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.get('learning_rate', 2e-4)
        )

        # 初始化约束掩码生成器
        self.masker = ConstraintMasker(config.get('structural', {}))

        # ADMET相关参数
        self.tox_weights = torch.tensor(
            config.get('toxicity_weights', [1.0] * TOXICITY_DIM)
        )
        self.adme_weights = torch.tensor(config.get('adme_weights', [0.2, 0.3, 0.8, 0.2]))
        # [0.2, 0.3, 0.2, 0.2]
        # 训练状态
        self.train_steps = 0

    def apply_constraints(self, seq: torch.Tensor,
                          logits: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """应用约束"""
        mask, seq_idx = self.masker.get_mask(seq)
        return logits.masked_fill(~mask, float('-inf')), seq_idx

    def calc_toxicity_penalty(self, toxicity_flags: torch.Tensor) -> torch.Tensor:
        """计算复合毒性惩罚"""
        weighted_tox = toxicity_flags * self.tox_weights.to(toxicity_flags.device)
        return torch.sum(weighted_tox, dim=1) * config.get('penalty_folds', 5.0)

    def get_optimal(self, iupac_list, adme_tensor, tox_tensor):
        # ADME理想范围
        # ADME_RANGES = [
        #     (0.2, 1.0),  # bioavailability: >20%
        #     (0.1, 1),  # VD: 0.1–1 L/kg
        #     (60, 100000),  # half-life: >60 min
        #     (0, 10)  # CL: <10 ml/min/kg
        # ]
        ADME_RANGES = config['ADME_RANGES']

        with open(os.path.join(os.path.dirname(args.txt_file), 'result.tsv'), 'a', encoding='utf-8') as f:
        # with open('result.tsv', 'a', encoding='utf-8') as f:
            for iupac, adme, tox in zip(iupac_list, adme_tensor, tox_tensor):
                print(iupac, adme, tox)
                adme_ok = (
                        (adme[0] > ADME_RANGES[0][0]) and
                        (ADME_RANGES[1][0] <= adme[1] <= ADME_RANGES[1][1]) and
                        (adme[2] > ADME_RANGES[2][0]) and
                        (ADME_RANGES[3][0] <= adme[3] < ADME_RANGES[3][1])
                )
                tox_ok = tox.sum() == 0
                if adme_ok and tox_ok:
                    adme_str = '\t'.join([str(x) for x in adme])
                    f.write(f"{iupac}\t{adme_str}\n")

    def train_step(self, aa_sequences: torch.Tensor) -> dict:
        """
        执行一步训练
        
        Args:
            aa_sequences: [batch_size, seq_len] 氨基酸序列索引张量
            
        Returns:
            metrics: 训练指标字典
        """

        # 验证输入
        if not isinstance(aa_sequences, torch.Tensor):
            raise TypeError("aa_sequences must be a torch.Tensor")
        if aa_sequences.dim() != 2:
            raise ValueError("aa_sequences must be 2-dimensional")
        if aa_sequences.size(1) != self.generator.seq_len:
            raise ValueError(f"Sequence length must be {self.generator.seq_len}")

        # 生成修饰方案
        mod_logits, value_pred = self.generator(aa_sequences)

        # 带约束的采样
        constrained_logits, seq_idx = self.apply_constraints(aa_sequences, mod_logits)
        dist = Categorical(logits=constrained_logits)
        actions = dist.sample()

        modified_feats = apply_modifications(aa_sequences, actions)
        with torch.no_grad():
            features, iupacs, masks = feat_extract(modified_feats, seq_idx)
            if features:
                size = len(features)
                species_tensor = torch.tensor(config['Species'], dtype=torch.float32).unsqueeze(0).repeat(size, 1)
                route_tensor = torch.tensor(config['Route'], dtype=torch.float32).unsqueeze(0).repeat(size, 1)
                adme_scores, toxicity_flags = discriminator(species_tensor, route_tensor, features, discriminant_model, batch_size, device)
                self.get_optimal(iupacs, adme_scores, toxicity_flags)
            else:
                return {'loss': 99999,
                        'reward': -99999,
                        'steps': self.train_steps}

        # ADME标准化
        adme_normalized = normalize_adme(adme_scores)

        # 计算奖励
        adme_reward = torch.einsum(
            'bd,d->b', adme_normalized,
            self.adme_weights.to(adme_normalized.device)
        )

        toxicity_penalty = self.calc_toxicity_penalty(toxicity_flags.float())

        # 计算总奖励（标准化后）
        total_reward = adme_reward - toxicity_penalty

        # 策略梯度更新
        loss = self._compute_loss(
            constrained_logits, actions,
            total_reward, value_pred, masks
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1

        # 打印时使用原始ADME值
        adme_original = denormalize_adme(adme_normalized)

        metrics = {
            'loss': loss.item(),
            'adme': adme_original.mean(dim=0).cpu().numpy(),  # 使用原始值
            'toxicity': toxicity_flags.float().mean(dim=1).cpu().numpy(),
            'reward': total_reward.mean().item(),
            'steps': self.train_steps
        }

        if self.train_steps % 100 == 0:
            logger.info(
                f"Step {self.train_steps}: "
                f"Loss={metrics['loss']:.3f}, "
                f"Reward={metrics['reward']:.2f}, "
                f"ADME原始值={adme_original.mean(0)}"
            )

        return metrics

    def _compute_loss(self, logits: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      value_pred: torch.Tensor,
                      masks: torch.Tensor) -> torch.Tensor:
        """
        计算损失函数
        
        Args:
            logits: 动作概率对数
            actions: 采样的动作
            rewards: 奖励值（total_reward）
            value_pred: 价值预测（标量）
            masks: 损失掩码
            
        Returns:
            loss: 总损失
        """
        # 按掩码取值
        logits, actions, value_pred = logits[masks], actions[masks], value_pred[masks]

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions).mean(dim=1)

        # 价值损失：预测值与实际平均奖励的MSE
        mean_reward = rewards.mean()  # 计算batch的平均奖励
        value_loss = F.mse_loss(value_pred, mean_reward.expand_as(value_pred))

        # 策略梯度：计算优势值
        advantages = rewards - value_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs * advantages).mean()

        # 熵正则化
        entropy = dist.entropy().mean()

        return policy_loss + 0.5 * value_loss - 0.1 * entropy

    def predict(self, aa_sequence: torch.Tensor) -> List[List[dict]]:
        self.generator.eval()
        with torch.no_grad():
            # 扩展为批次维度
            aa_sequence = aa_sequence.unsqueeze(0)

            # 生成修饰概率
            mod_logits, _ = self.generator(aa_sequence)
            constrained_logits, _ = self.apply_constraints(
                aa_sequence, mod_logits
            )
            probs = F.softmax(constrained_logits, dim=-1)

            # 转换为可读格式
            predictions = []
            for b in range(probs.size(0)):
                for pos in range(probs.size(1)):
                    modifiers = []
                    for mod_idx in range(probs.size(2)):
                        mod_prob = probs[b, pos, mod_idx].item()
                        if mod_prob > 0:  # 只考虑概率值不为零的结果
                            mod_name = MOD_REVERSE.get(mod_idx, '')
                            modifiers.append({
                                'position': pos + 1,
                                'type': mod_name,
                                'probability': mod_prob
                            })

                    # 对 modifiers 按照概率值从高到低排序
                    modifiers = sorted(modifiers, key=lambda x: x['probability'], reverse=True)
                    predictions.append(modifiers)

        return predictions


def make_input(config):
    in_tensor = []
    for b in range(config['batch_size']):
        seq_idx = []
        for l in range(config['seq_len']):
            seq_idx.append(AA_TO_IDX[config['naked_sequence'][l]])
        in_tensor.append(seq_idx)
    return torch.tensor(in_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process summary table file.')
    parser.add_argument('--txt_file', type=str, required=True, help='Path to the summary_table.txt file')
    args = parser.parse_args()
    config = preprocess_txt_opt(args.txt_file)
    os.chmod(os.path.join(os.path.dirname(args.txt_file), 'result.tsv'), 0o777)
    print(config)
    # config = {
    #     'naked_sequence': 'AFASYNLKPA',
    #     'seq_len': 15,
    # 'batch_size': batch_size,
    # 'structural': {
    # },
    # 'toxicity_weights': [10.0] * 11,
    # 'penalty_folds': 5.0,
    # 'adme_weights': [1.0, 1.0, 1.0, 1.0],
    # 'learning_rate': 2e-4,
    # 'ADME_RANGES': []
    # }
    config['seq_len'] = len(config['naked_sequence'])
    config['structural'].update(constraints_config)
    discriminant_model = Scarlett(n_targets=1, num_classes=11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    discriminant_model.to(device)
    trainer = ADMETTrainer(config)
    in_tensor = make_input(config)
    logger.info("=== 训练阶段 ===")
    for epoch in range(1):
        metrics = trainer.train_step(in_tensor)
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Loss={metrics['loss']:.3f}, "
            f"Reward={metrics['reward']:.2f}"
        )
    # 测试示例
    logger.info("\n=== 测试阶段 ===")
    predictions = trainer.predict(in_tensor[0])

    # 输出结果
    # aa_seq_str = "".join([IDX_TO_AA.get(x.item(), 'X') for x in test_sequence])
    logger.info(f"原始序列: {config['naked_sequence']}")
    logger.info("推荐修饰方案：")
    for pos, mods in enumerate(predictions):
        logger.info(f"位置 {pos + 1}:")
        for mod_info in mods:
            logger.info(
                f"  {mod_info['type']}: "
                f"{mod_info['probability']:.2%}"
            )
