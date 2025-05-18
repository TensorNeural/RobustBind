import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointbert import misc  # assumes misc.fps is defined


### KNN UTILITIES ###
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    dists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


### MODULES ###
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        B, N, _ = xyz.shape
        center = misc.fps(xyz, self.num_group)  # B G 3
        idx = knn_point(self.group_size, xyz, center)  # B G M

        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.view(B * N, -1)[idx, :]
        neighborhood = neighborhood.view(B, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, point_groups):
        B, G, N, _ = point_groups.shape
        x = point_groups.reshape(B * G, N, 3)
        x = self.first_conv(x.transpose(2, 1))
        x_max = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.cat([x_max.expand(-1, -1, N), x], dim=1)
        x = self.second_conv(x)
        x = torch.max(x, dim=2)[0]
        return x.view(B, G, -1)


class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = num_fine // 4

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size).view(1, -1).repeat(self.grid_size, 1).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size).view(-1, 1).repeat(1, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, -1)

    def forward(self, feature_global):
        B, G, C = feature_global.shape
        x = feature_global.reshape(B * G, C)
        coarse = self.mlp(x).view(B * G, self.num_coarse, 3)

        seed = self.folding_seed.to(x.device).expand(B * G, -1, self.num_fine)
        center = coarse.unsqueeze(2).expand(-1, -1, self.num_fine // self.num_coarse, -1).reshape(B * G, self.num_fine, 3)
        center = center.transpose(1, 2)

        x_exp = x.unsqueeze(2).expand(-1, -1, self.num_fine)
        feat = torch.cat([x_exp, seed, center], dim=1)

        fine = self.final_conv(feat) + center
        fine = fine.view(B, G, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.view(B, G, self.num_coarse, 3)
        return coarse, fine


class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                    nn.GroupNorm(4, 1024),
                                    nn.LeakyReLU(0.2))
        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, 1, bias=False),
                                    nn.GroupNorm(4, output_channel),
                                    nn.LeakyReLU(0.2))

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k, k=4):
        B, C, N = x_q.shape
        _, _, M = x_k.shape

        idx = knn_point(k, coor_k.transpose(1, 2), coor_q.transpose(1, 2))  # [B, N, k]
        idx_base = torch.arange(B, device=x_q.device).view(-1, 1, 1) * M
        idx = idx + idx_base
        idx = idx.view(-1)

        x_k = x_k.transpose(2, 1).contiguous()  # [B, M, C]
        neighbors = x_k.view(B * M, -1)[idx, :].view(B, N, k, C).permute(0, 3, 1, 2)
        x_q = x_q.unsqueeze(-1).expand(-1, -1, -1, k)
        return torch.cat((neighbors - x_q, x_q), dim=1)

    def forward(self, f, coor):
        feat_list = []
        coor = coor.transpose(1, 2)
        f = self.input_trans(f.transpose(1, 2))

        f1 = self.get_graph_feature(coor, f, coor, f)
        f1 = self.layer1(f1).max(dim=-1)[0]
        feat_list.append(f1)

        f2 = self.get_graph_feature(coor, f1, coor, f1)
        f2 = self.layer2(f2).max(dim=-1)[0]
        feat_list.append(f2)

        f3 = self.get_graph_feature(coor, f2, coor, f2)
        f3 = self.layer3(f3).max(dim=-1)[0]
        feat_list.append(f3)

        f4 = self.get_graph_feature(coor, f3, coor, f3)
        f4 = self.layer4(f4).max(dim=-1)[0]
        feat_list.append(f4)

        x = torch.cat(feat_list, dim=1)
        x = self.layer5(x)
        return x.transpose(1, 2)


class DiscreteVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims
        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        self.group_divider = Group(self.num_group, self.group_size)
        self.encoder = Encoder(self.encoder_dims)
        self.dgcnn_1 = DGCNN(self.encoder_dims, self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))
        self.dgcnn_2 = DGCNN(self.tokens_dims, self.decoder_dims)
        self.decoder = Decoder(self.decoder_dims, self.group_size)

    def forward(self, inp, temperature=1., hard=False):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)
        logits = self.dgcnn_1(logits, center)
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=-1, hard=hard)
        sampled = torch.einsum('bgn,nc->bgc', soft_one_hot, self.codebook)
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        with torch.no_grad():
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.shape[0], -1, 3)
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.shape[0], -1, 3)

        return whole_coarse, whole_fine, coarse, fine, neighborhood, logits
