import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os.path as osp
import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Patcher(nn.Module):
    def __init__(self, patch_size, stride, in_chan, out_dim):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        linear_input_dim = int(in_chan * patch_size[0] * patch_size[1])
        self.to_out = nn.Sequential(
            nn.Linear(linear_input_dim, out_dim),
            nn.GELU()
        )
        self.transpose_for_linear = Rearrange("b features_dim num_patches -> b num_patches features_dim")

    def forward(self, x):
        # print(f"[Patcher forward] Input x shape: {x.shape}")
        x_unfolded = self.unfold(x)
        # print(f"[Patcher forward] x_unfolded shape: {x_unfolded.shape}")
        x_transposed = self.transpose_for_linear(x_unfolded)
        # print(f"[Patcher forward] x_transposed shape (for Linear): {x_transposed.shape}")
        x_out = self.to_out(x_transposed)
        # print(f"[Patcher forward] Output x_out shape: {x_out.shape}")
        return x_out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x_normed = self.norm(x)
        qkv = self.to_qkv(x_normed).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn_block, ff_block in self.layers:
            x_attended = attn_block(x)
            x = x_attended + x
            x_ff = ff_block(x)
            x = x_ff + x
        return self.norm_final(x)


class PatchFormer(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel_tuple, pool_kernel_w=4):
        # print(f"[temporal_learner] Conv kernel: {kernel_tuple}, Pool kernel: (1, {pool_kernel_w})")
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_tuple, stride=(1, 1), padding=self.get_padding(kernel_tuple[-1])),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_chan),
            nn.AvgPool2d((1, pool_kernel_w), (1, pool_kernel_w))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T, patch_time, patch_step, dim_head, depth, heads,
                 dropout_rate, idx_graph):
        super(PatchFormer, self).__init__()

        # print(f"[PatchFormer __init__] input_size: {input_size}, sampling_rate: {sampling_rate}, num_T (CNN out_chan): {num_T}")
        # print(f"[PatchFormer __init__] patch_time (Patcher kernel_W): {patch_time}, patch_step (Patcher stride_W): {patch_step}")

        self.idx = idx_graph
        self.window_duration_factor = 0.5
        self.eeg_channels_count = input_size[1]
        self.brain_area_count = len(self.idx)

        temporal_cnn_kernel_w = int(self.window_duration_factor * sampling_rate + 1)
        self.temporal_cnn = self.temporal_learner(input_size[0], num_T,
                                                  (1, temporal_cnn_kernel_w), pool_kernel_w=4)

        one_x_one_pool_kernel_w = 1
        # print(f"[PatchFormer __init__] OneXOneConv Pool kernel: (1, {one_x_one_pool_kernel_w})")
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_T),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, one_x_one_pool_kernel_w), (1, one_x_one_pool_kernel_w)))

        self.global_cnn = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(self.eeg_channels_count, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_T),
            nn.LeakyReLU()
        )

        _L_original_input = input_size[2]

        padding_tuple_conv1 = self.get_padding(temporal_cnn_kernel_w)
        _pad_conv1_w = padding_tuple_conv1[1] 

        _L_out_conv1 = (_L_original_input + 2 * _pad_conv1_w - temporal_cnn_kernel_w) // 1 + 1 

        _L_after_pool1_in_tcnn = _L_out_conv1 // 4
        _L_final_temporal_after_1x1 = _L_after_pool1_in_tcnn // one_x_one_pool_kernel_w

        if _L_final_temporal_after_1x1 < patch_time:
            raise ValueError(f"Calculated L_final_temporal ({_L_final_temporal_after_1x1}) is less than Patcher's patch_time ({patch_time}). "
                             "Adjust pooling in temporal_cnn/OneXOneConv or reduce patch_time argument in main script.")

        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.eeg_channels_count, num_T * _L_final_temporal_after_1x1),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.eeg_channels_count, 1), dtype=torch.float32),
                                              requires_grad=True)

        self.aggregate = Aggregator(self.idx)

        patcher_kernel_h = 1
        self.to_patch = Patcher(patch_size=(patcher_kernel_h, patch_time),
                                stride=(1, patch_step),
                                in_chan=num_T,
                                out_dim=dim_head)

        self.transformer_mlp_dim = dim_head * 4
        self.transformer = Transformer(
            dim=dim_head, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=self.transformer_mlp_dim, dropout=dropout_rate,
        )

        _H_input_to_Patcher = self.brain_area_count + 1
        _W_input_to_Patcher = _L_final_temporal_after_1x1

        _num_patches_W_by_Patcher = (_W_input_to_Patcher - patch_time) // patch_step + 1
        _num_patches_H_by_Patcher = (_H_input_to_Patcher - patcher_kernel_h) // 1 + 1

        num_total_patches_for_transformer = _num_patches_H_by_Patcher * _num_patches_W_by_Patcher

        if _num_patches_W_by_Patcher <= 0 or _num_patches_H_by_Patcher <=0:
             raise ValueError(f"Calculated num_patches_W ({_num_patches_W_by_Patcher}) or num_patches_H ({_num_patches_H_by_Patcher}) is not positive. "
                              "Input dimensions to Patcher or Patcher's kernel/stride are incompatible.")

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(num_total_patches_for_transformer * dim_head), num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)

        out_temporal_cnn = self.temporal_cnn(x)
        # print(f"[PF forward] After temporal_cnn: {out_temporal_cnn.shape}")
        out_one_x_one = self.OneXOneConv(out_temporal_cnn)
        # print(f"[PF forward] After OneXOneConv: {out_one_x_one.shape}")
        out_global_branch = out_one_x_one
        b, k_num_T, c_eeg, l_final = out_global_branch.size()
        out_global = self.global_cnn(out_global_branch)
        # print(f"[PF forward] out_global: {out_global.shape}")
        out_for_local_filter = rearrange(out_one_x_one, 'b k c l -> b c (k l)')
        # print(f"[PF forward] out_for_local_filter (for local_filter_fun): {out_for_local_filter.shape}")
        out_filtered_local = self.local_filter_fun(out_for_local_filter, self.local_filter_weight)
        # print(f"[PF forward] out_filtered_local: {out_filtered_local.shape}")
        out_aggregated_local = self.aggregate.forward(out_filtered_local)
        # print(f"[PF forward] out_aggregated_local: {out_aggregated_local.shape}")
        out_local_reshaped = rearrange(out_aggregated_local, 'b g (k l) -> b k g l', k=k_num_T, l=l_final)
        # print(f"[PF forward] out_local_reshaped: {out_local_reshaped.shape}")
        out_to_patcher_input = torch.cat((out_global, out_local_reshaped), dim=-2)
        # print(f"[PF forward] out_to_patcher_input (Input to Patcher): {out_to_patcher_input.shape}")
        out_patched = self.to_patch(out_to_patcher_input)
        # print(f"[PF forward] out_patched (Output of Patcher): {out_patched.shape}")
        out_transformed = self.transformer(out_patched)
        # print(f"[PF forward] out_transformed: {out_transformed.shape}")
        out_flattened = out_transformed.reshape(out_transformed.size(0), -1)
        # print(f"[PF forward] out_flattened: {out_flattened.shape}")
        out_final = self.fc(out_flattened)
        # print(f"[PF forward] out_final: {out_final.shape}")
        return out_final

    def get_size_temporal(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        with torch.no_grad():
            out = self.temporal_cnn(data)
            out = self.OneXOneConv(out)
        return out.shape

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_padding(self, kernel_width_dim):
        return (0, int(0.5 * (kernel_width_dim - 1)))


class Aggregator():
    def __init__(self, idx_area_list_of_channel_counts):
        self.idx_boundaries = self.get_idx_boundaries(idx_area_list_of_channel_counts)
        self.num_areas = len(idx_area_list_of_channel_counts)

    def forward(self, x):
        data_aggregated_per_area = []
        for i in range(self.num_areas):
            start_channel_idx = self.idx_boundaries[i]
            end_channel_idx = self.idx_boundaries[i+1]
            channels_for_current_area = x[:, start_channel_idx:end_channel_idx, :]
            data_aggregated_per_area.append(self.aggr_fun(channels_for_current_area, dim=1))
        return torch.stack(data_aggregated_per_area, dim=1)

    def get_idx_boundaries(self, chan_counts_in_each_area):
        boundaries = [0]
        current_sum = 0
        for count in chan_counts_in_each_area:
            current_sum += count
            boundaries.append(current_sum)
        return boundaries

    def aggr_fun(self, x, dim):
        return torch.mean(x, dim=dim)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    bs = 2
    num_classes_example = 3
    num_eeg_channels = 30
    L_original_eeg = 300
    sampling_rate_eeg = 200
    num_T_cnn_out_channels = 14
    patch_time_for_patcher = 40
    patch_step_for_patcher = 20
    dim_head_transformer = 64
    depth_transformer = 3
    heads_transformer = 4
    dropout_rate_model = 0.5
    idx_graph_example = [num_eeg_channels]
    input_size_example = (1, 28, 800)

    print("--- Example Instantiation ---")
    example_data = torch.randn(bs, num_eeg_channels, L_original_eeg)

    net = PatchFormer(
        num_classes=num_classes_example,
        input_size=input_size_example,
        sampling_rate=sampling_rate_eeg,
        num_T=num_T_cnn_out_channels,
        patch_time=patch_time_for_patcher,
        patch_step=patch_step_for_patcher,
        dim_head=dim_head_transformer,
        depth=depth_transformer,
        heads=heads_transformer,
        dropout_rate=dropout_rate_model,
        idx_graph=idx_graph_example
    )

    print(f"\nTotal trainable parameters: {count_parameters(net)}")

    print("\n--- Example Forward Pass ---")
    try:
        with torch.no_grad():
            out = net(example_data)
        print(f"Output shape of forward pass: {out.shape}")
        assert out.shape == (bs, num_classes_example)
        print("Example forward pass successful and output shape is correct.")
    except Exception as e:
        print(f"Error during example forward pass: {e}")
        import traceback
        traceback.print_exc()