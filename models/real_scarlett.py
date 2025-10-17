import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def calculate_padding(kernel_size, dilation=1):
    return ((kernel_size - 1) * dilation) // 2


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ConvBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ConvBlock3(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvBlock3, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=calculate_padding(kernel_size))
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.norm1 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.conv2(x))
        return x


class DenseNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, last_combine=None, units=16):
        super(DenseNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, units))
            else:
                self.layers.append(nn.Linear(units, units))

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.last_combine = last_combine
        if self.last_combine:
            self.last_layer = nn.Linear(units, 11)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.last_combine:
            x = self.last_layer(x)
        return x


class Scarlett(nn.Module):

    def __init__(self, n_targets, num_classes):
        super(Scarlett, self).__init__()
        self.model_type = "base"
        self.global_dense = DenseNetwork(input_dim=7, num_layers=3, units=16)

        self.one_hot_conv = ConvBlock(6, 120, kernel_size=2)

        self.amino_conv1 = ConvBlock2(6, 256, kernel_size=8)
        self.amino_conv2 = ConvBlock2(256, 128, kernel_size=8)
        self.amino_conv3 = ConvBlock3(128, 64, kernel_size=8)

        self.inter4 = ConvBlock2(256, 6, kernel_size=3)

        self.diamino_conv1 = ConvBlock2(6, 128, kernel_size=2)
        self.diamino_conv2 = ConvBlock3(128, 64, kernel_size=2)

        self.combined_dense = DenseNetwork(input_dim=838, num_layers=4, units=128)

        self.species = DenseNetwork(input_dim=28, num_layers=1, units=128)
        self.route = DenseNetwork(input_dim=13, num_layers=1, units=128)

        self.regression_output = nn.Linear(128, n_targets)

        self.classification_output = nn.Linear(128, num_classes)

    def forward(self, species, route, smiles_ls):
        # x1: (batch_size, 7)
        # x2: (batch_size, 130, 6)
        # x3: (batch_size, 65, 6)
        # x4: (batch_size, 130, 256)
        # x5: (batch_size, 8, 6)
        smiles1, smiles2, smiles3, smiles4, smiles5 = smiles_ls
        # print(smiles1.size(), smiles2.size(), smiles3.size(), smiles4.size(), smiles5.size())
        # exit()
        smiles1_out = self.global_dense(smiles1)
        smiles1_out = smiles1_out.view(smiles1_out.size(0), -1)

        one_hot = smiles2.permute(0, 2, 1)
        one_hot_out = self.one_hot_conv(one_hot)
        smiles2_out = one_hot_out.view(one_hot_out.size(0), -1)

        amino_acid = smiles3.permute(0, 2, 1)
        amino_out = self.amino_conv1(amino_acid)
        amino_out = self.amino_conv2(amino_out)
        amino_out = self.amino_conv3(amino_out)
        smiles3_out = amino_out.view(amino_acid.size(0), -1)

        x4_out = smiles4.permute(0, 2, 1)
        x4_out = self.inter4(x4_out)
        smiles4_out = x4_out.view(x4_out.size(0), -1)

        diamino_acid = smiles5.permute(0, 2, 1)
        diamino_out = self.diamino_conv1(diamino_acid)
        diamino_out = self.diamino_conv2(diamino_out)
        smiles5_out = diamino_out.view(diamino_acid.size(0), -1)

        out_concat = torch.cat([smiles1_out, smiles2_out, smiles3_out, smiles4_out, smiles5_out], dim=1)
        # print(out_concat.size())
        out = self.combined_dense(out_concat)
        x6_out, x7_out = self.species(species), self.route(route)
        out = 0.8 * out + 0.1 * x6_out + 0.1 * x7_out
        regression_out = self.regression_output(out)  # 形状为 (batch_size, 4)
        classification_out = self.classification_output(out)  # 形状为 (batch_size, num_classes)
        return regression_out, classification_out

    # def get_embedding(self, species, route, smiles_ls):
    #     smiles1, smiles2, smiles3, smiles4, smiles5 = smiles_ls
    #     smiles1_out = self.global_dense(smiles1)
    #     smiles1_out = smiles1_out.view(smiles1_out.size(0), -1)
    #
    #     one_hot = smiles2.permute(0, 2, 1)
    #     one_hot_out = self.one_hot_conv(one_hot)
    #     smiles2_out = one_hot_out.view(one_hot_out.size(0), -1)
    #
    #     amino_acid = smiles3.permute(0, 2, 1)
    #     amino_out = self.amino_conv1(amino_acid)
    #     amino_out = self.amino_conv2(amino_out)
    #     amino_out = self.amino_conv3(amino_out)
    #     smiles3_out = amino_out.view(amino_acid.size(0), -1)
    #
    #     x4_out = smiles4.permute(0, 2, 1)
    #     x4_out = self.inter4(x4_out)
    #     smiles4_out = x4_out.view(x4_out.size(0), -1)
    #
    #     diamino_acid = smiles5.permute(0, 2, 1)
    #     diamino_out = self.diamino_conv1(diamino_acid)
    #     diamino_out = self.diamino_conv2(diamino_out)
    #     smiles5_out = diamino_out.view(diamino_acid.size(0), -1)
    #
    #     out_concat = torch.cat([smiles1_out, smiles2_out, smiles3_out, smiles4_out, smiles5_out], dim=1)
    #     # print(out_concat.size())
    #     out = self.combined_dense(out_concat)
    #     return out


class PropertySpecificResidual(nn.Module):
    """属性专用残差模块"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 基础特征+物种+给药方式
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(output_dim)
        )
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        residual = self.output_layer(x)
        out = self.residual_net(x)
        return out + residual


class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Softmax(dim=1)
        )

    def forward(self, base_feat, species_feat, route_feat):
        combined = torch.cat([base_feat, species_feat, route_feat], dim=1)
        weights = self.attention(combined)
        return base_feat * weights[:, 0:1] + species_feat * weights[:, 1:2] + route_feat * weights[:, 2:3]


class MultiPropertyScarlett(nn.Module):
    def __init__(self, base_model, n_targets, num_classes):
        super().__init__()
        self.base_model = base_model
        # self.freeze_smiles_layers()
        self.gate_species = nn.Sequential(
            nn.Linear(28, 28),  # 维度需与特征维度匹配
            nn.LayerNorm(28),
            nn.Sigmoid()
        )

        # 给药途径特征修正门控
        self.gate_route = nn.Sequential(
            nn.Linear(13, 13),
            nn.LayerNorm(13),
            nn.Sigmoid()
        )

        self.species_mapper = nn.ModuleDict({
            'reg': nn.Sequential(
                nn.Linear(n_targets, 28),
                nn.Tanh()
            ),
            'cls': nn.Sequential(
                nn.Linear(num_classes, 28),
            )
        })
        self.route_mapper = nn.ModuleDict({
            'reg': nn.Sequential(
                nn.Linear(n_targets, 13),
                nn.Tanh()
            ),
            'cls': nn.Sequential(
                nn.Linear(num_classes, 13),
            )
        })
        self._initialize_weights()

    def _initialize_weights(self):
        """ Xavier初始化门控层参数 """
        for module in [self.gate_species, self.gate_route]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init.xavier_normal_(layer.weight)
                    init.constant_(layer.bias, 0.1)

    def freeze_smiles_layers(self):
        for name, param in self.base_model.named_parameters():
            if any(key in name for key in ["global_dense", "one_hot_conv",
                                           "amino_conv", "inter4", "diamino_conv"]):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def update_predictions(self, species, route, smiles_ls):
        return self.base_model(species, route, smiles_ls)

    def forward(self, species, route, smiles_ls, y_reg=None, y_cls=None):
        # smiles_feat = self.base_model.get_embedding(species, route, smiles_ls)
        # processed_feat = smiles_feat.clone()

        reg_pre, cls_pre = self.update_predictions(species, route, smiles_ls)
        if y_reg is None and y_cls is None:
            return reg_pre, cls_pre
        species_orig = species.clone()
        route_orig = route.clone()
        for _ in range(1):
            if y_reg is not None:
                residual = y_reg - reg_pre
                species_res = self.species_mapper['reg'](residual)
                route_res = self.route_mapper['reg'](residual)
            elif y_cls is not None:
                residual = y_cls - torch.sigmoid(cls_pre)
                species_res = self.species_mapper['cls'](residual)
                route_res = self.route_mapper['cls'](residual)
            else:
                raise ValueError("Must provide either regression or classification labels")

            # species = torch.sigmoid(species_res)
            # route = torch.sigmoid(route_res)
            species = species_orig + self.gate_species(species_res) * species_res
            route = route_orig + self.gate_route(route_res) * route_res
            new_reg, new_cls = self.update_predictions(species, route, smiles_ls)

            reg_pre = reg_pre + new_reg
            cls_pre = cls_pre + new_cls
        # 双任务输出
        # reg_output = self.reg_head(final_feat)
        # cls_output = self.cls_head(final_feat)

        return reg_pre, cls_pre


if __name__ == "__main__":
    model = Scarlett(n_targets=1, num_classes=11)
    batch_size = 4

    x1 = torch.randn(batch_size, 7).to(torch.float32)
    x2 = torch.randn(batch_size, 122, 6).to(torch.float32)
    x3 = torch.randn(batch_size, 61, 6).to(torch.float32)
    x4 = torch.randn(batch_size, 122, 256).to(torch.float32)
    x5 = torch.randn(batch_size, 8, 6).to(torch.float32)
    random_index = torch.randint(0, 28, (batch_size,))
    one_hot1 = F.one_hot(random_index, num_classes=28).float()
    random_index2 = torch.randint(0, 13, (batch_size,))
    one_hot2 = F.one_hot(random_index2, num_classes=13).float()
    y1, y2 = model(one_hot1, one_hot2, [x1, x2, x3, x4, x5])

    # admet = MultiPropertyScarlett(model, 1, 15)
    # y1, y2 = admet(one_hot, one_hot, [x1, x2, x3, x4, x5])
    #
    # print("Output shape:", y1.shape, y2.shape)
