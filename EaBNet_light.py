"""Lightweight EaBNet with Fourier Convolutional Attention Encoder.

This module implements the method described in
"A Lightweight Fourier Convolutional Attention Encoder for Multi-Channel
Speech Enhancement".  The public interface follows the original EaBNet.py:
input STFT shape is (B, T, F, M, 2), output shape is (B, 2, T, F).
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


def _match_tf(x: Tensor, ref: Tensor) -> Tensor:
    """Crop/pad time-frequency axes of x to match ref."""
    target_t, target_f = ref.shape[-2], ref.shape[-1]
    x = x[..., :target_t, :target_f]
    pad_t = target_t - x.shape[-2]
    pad_f = target_f - x.shape[-1]
    if pad_t > 0 or pad_f > 0:
        x = nn.functional.pad(x, (0, max(0, pad_f), 0, max(0, pad_t)))
    return x


class Chomp_T(nn.Module):
    def __init__(self, t: int):
        super().__init__()
        self.t = int(t)

    def forward(self, x: Tensor) -> Tensor:
        if self.t <= 0:
            return x
        return x[:, :, :-self.t, :]


class GateConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t - 1, 0), value=0.0),
                nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride),
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        outputs, gate = self.conv(inputs).chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size, stride),
                Chomp_T(k_t - 1),
            )
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels * 2, kernel_size, stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        outputs, gate = self.conv(inputs).chunk(2, dim=1)
        return outputs * gate.sigmoid()


class CausalConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t - 1, 0), value=0.0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class CausalConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        k_t = kernel_size[0]
        if k_t > 1:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                Chomp_T(k_t - 1),
            )
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_grad=False)

    def forward(self, inpt: Tensor) -> Tensor:
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)
        entry_cnt = np.arange(channel, channel * (seq_len + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type()).view(1, -1).expand(b_size, -1)
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_power_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        x = (inpt - cum_mean.unsqueeze(1)) / cum_std.unsqueeze(1)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm2d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt: Tensor) -> Tensor:
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1, 3], keepdim=True)
        step_pow_sum = inpt.pow(2).sum([1, 3], keepdim=True)
        cum_sum = torch.cumsum(step_sum, dim=-2)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)
        entry_cnt = np.arange(channel * freq_num, channel * freq_num * (seq_len + 1), channel * freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type()).view(1, 1, seq_len, 1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class NormSwitch(nn.Module):
    def __init__(self, norm_type: str, dim_size: str, c: int):
        super().__init__()
        assert norm_type in ["BN", "IN", "cLN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            self.norm = nn.BatchNorm1d(c) if dim_size == "1D" else nn.BatchNorm2d(c)
        elif norm_type == "IN":
            self.norm = nn.InstanceNorm1d(c, affine=True) if dim_size == "1D" else nn.InstanceNorm2d(c, affine=True)
        else:
            self.norm = CumulativeLayerNorm1d(c, affine=True) if dim_size == "1D" else CumulativeLayerNorm2d(c, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.avg_conv = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=False)
        self.max_conv = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        att = self.avg_conv(torch.relu(avg_map)) + self.max_conv(torch.relu(max_map))
        return x * torch.sigmoid(att)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = torch.mean(x, dim=(-2, -1), keepdim=True)
        max_pool = torch.amax(x, dim=(-2, -1), keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * att


class SkipAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        return self.sa(self.ca(x))


class FourierAttentionBlock(nn.Module):
    """FCAE/FCAD core after the first convolution step."""

    def __init__(self, channels: int, norm_type: str):
        super().__init__()
        self.sa_q = SpatialAttention()
        self.fft_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            NormSwitch(norm_type, "2D", channels * 2),
            nn.ReLU(inplace=True),
        )
        self.sa_p = SpatialAttention()
        self.ca = ChannelAttention(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, q: Tensor) -> Tensor:
        freq_len = q.shape[-1]
        h = torch.fft.rfft(self.sa_q(q).float(), n=freq_len, dim=-1)
        h_ri = torch.cat([h.real, h.imag], dim=1).to(dtype=q.dtype)
        p = self.fft_conv(h_ri)
        z = self.sa_p(p)
        zr, zi = z.chunk(2, dim=1)
        k = torch.fft.irfft(torch.complex(zr.float(), zi.float()), n=freq_len, dim=-1).to(dtype=q.dtype)
        return self.out_conv(k + self.ca(q))


class FCAE(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: tuple, stride: tuple, norm_type: str):
        super().__init__()
        self.in_conv = nn.Sequential(
            CausalConv2d(cin, cout, kernel_size, stride),
            NormSwitch(norm_type, "2D", cout),
            nn.PReLU(cout),
        )
        self.fourier_attention = FourierAttentionBlock(cout, norm_type)

    def forward(self, x: Tensor) -> Tensor:
        q = self.in_conv(x)
        return self.fourier_attention(q)


class FCAD(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: tuple, stride: tuple, norm_type: str):
        super().__init__()
        self.in_deconv = nn.Sequential(
            CausalConvTranspose2d(cin, cout, kernel_size, stride),
            NormSwitch(norm_type, "2D", cout),
            nn.PReLU(cout),
        )
        self.fourier_attention = FourierAttentionBlock(cout, norm_type)

    def forward(self, x: Tensor, target: Tensor = None) -> Tensor:
        q = self.in_deconv(x)
        if target is not None:
            q = _match_tf(q, target)
        return self.fourier_attention(q)


class SharedDFSMN(nn.Module):
    """A compact temporal DFSMN-style memory block shared across repeats."""

    def __init__(self, channels: int = 64, hidden_units: int = 64, memory_size: int = 5, norm_type: str = "BN", is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
        self.memory_size = int(memory_size)
        self.in_conv = nn.Conv1d(channels, hidden_units, kernel_size=1, bias=False)
        self.norm1 = NormSwitch(norm_type, "1D", hidden_units)
        self.prelu = nn.PReLU(hidden_units)
        self.memory_conv = nn.Conv1d(
            hidden_units,
            hidden_units,
            kernel_size=memory_size,
            groups=hidden_units,
            bias=False,
        )
        self.norm2 = NormSwitch(norm_type, "1D", hidden_units)
        self.out_conv = nn.Conv1d(hidden_units, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b_size, channels, seq_len, freq_len = x.shape
        y = x.permute(0, 3, 1, 2).reshape(b_size * freq_len, channels, seq_len)
        residual = y
        y = self.prelu(self.norm1(self.in_conv(y)))
        if self.is_causal:
            y = nn.functional.pad(y, (self.memory_size - 1, 0))
        else:
            left = (self.memory_size - 1) // 2
            right = self.memory_size - 1 - left
            y = nn.functional.pad(y, (left, right))
        y = self.memory_conv(y)
        y = self.norm2(y)
        y = self.out_conv(y) + residual
        return y.reshape(b_size, freq_len, channels, seq_len).permute(0, 2, 3, 1).contiguous()


class CRED(nn.Module):
    def __init__(
        self,
        cin: int,
        channels: int = 64,
        embed_dim: int = 64,
        norm_type: str = "BN",
        is_causal: bool = True,
        dfsmn_layers: int = 3,
        dfsmn_hidden: int = 64,
    ):
        super().__init__()
        enc_kernels = [(2, 5), (2, 3), (2, 3), (2, 3), (2, 3)]
        strides = [(1, 2)] * 5
        self.encoder = nn.ModuleList()
        for idx, kernel in enumerate(enc_kernels):
            self.encoder.append(FCAE(cin if idx == 0 else channels, channels, kernel, strides[idx], norm_type))

        self.dfsmn = SharedDFSMN(channels, dfsmn_hidden, memory_size=5, norm_type=norm_type, is_causal=is_causal)
        self.dfsmn_layers = int(dfsmn_layers)
        self.skip_attention = nn.ModuleList([SkipAttention(channels) for _ in range(5)])

        self.decoder = nn.ModuleList()
        for idx, kernel in enumerate(reversed(enc_kernels)):
            self.decoder.append(FCAD(channels * 2, channels, kernel, (1, 2), norm_type))
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
            NormSwitch(norm_type, "2D", embed_dim),
            nn.PReLU(embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        skips: List[Tensor] = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        for _ in range(self.dfsmn_layers):
            x = self.dfsmn(x)

        for idx, dec in enumerate(self.decoder):
            skip = self.skip_attention[-(idx + 1)](skips[-(idx + 1)])
            x = _match_tf(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, target=skip)
        return self.out_conv(x)


class LSTM_BF(nn.Module):
    def __init__(self, embed_dim: int, M: int, hid_node: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.rnn1 = nn.LSTM(input_size=embed_dim, hidden_size=hid_node, batch_first=True, bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hid_node, hidden_size=hid_node, batch_first=True, bidirectional=False)
        self.w_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node),
            nn.ReLU(True),
            nn.Linear(hid_node, 2 * M),
        )
        self.norm = nn.LayerNorm([embed_dim])

    def forward(self, embed_x: Tensor) -> Tensor:
        b_size, _, seq_len, freq_len = embed_x.shape
        x = self.norm(embed_x.permute(0, 3, 2, 1).contiguous())
        x = x.view(b_size * freq_len, seq_len, -1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.view(b_size, freq_len, seq_len, -1).transpose(1, 2).contiguous()
        return self.w_dnn(x).view(b_size, seq_len, freq_len, self.M, 2)


class EaBNet(nn.Module):
    def __init__(
        self,
        k1: tuple = (2, 3),
        k2: tuple = (1, 3),
        c: int = 64,
        M: int = 8,
        embed_dim: int = 64,
        kd1: int = 5,
        cd1: int = 64,
        d_feat: int = 256,
        p: int = 6,
        q: int = 3,
        is_causal: bool = True,
        is_u2: bool = True,
        bf_type: str = "lstm",
        topo_type: str = "mimo",
        intra_connect: str = "cat",
        norm_type: str = "BN",
        dfsmn_layers: int = 3,
    ):
        super().__init__()
        self.M = M
        self.embed_dim = embed_dim
        self.bf_type = bf_type
        self.topo_type = topo_type
        self.norm_type = norm_type
        self.cred = CRED(
            cin=M * 2,
            channels=c,
            embed_dim=embed_dim,
            norm_type=norm_type,
            is_causal=is_causal,
            dfsmn_layers=dfsmn_layers,
            dfsmn_hidden=cd1,
        )

        if topo_type == "mimo":
            if bf_type == "lstm":
                self.bf_map = LSTM_BF(embed_dim, M)
            elif bf_type == "cnn":
                self.bf_map = nn.Conv2d(embed_dim, M * 2, (1, 1), (1, 1))
            else:
                raise ValueError(f"unsupported bf_type: {bf_type}")
        elif topo_type == "miso":
            self.bf_map = nn.Conv2d(embed_dim, 2, (1, 1), (1, 1))
        else:
            raise ValueError(f"unsupported topo_type: {topo_type}")

    def forward(self, inpt: Tensor) -> Tensor:
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=-2)
        b_size, seq_len, freq_len, mic_num, _ = inpt.shape
        if mic_num != self.M:
            raise ValueError(f"expected {self.M} microphones, got {mic_num}")

        x = inpt.transpose(-2, -1).contiguous()
        x = x.view(b_size, seq_len, freq_len, -1).permute(0, 3, 1, 2)
        x = self.cred(x)
        x = _match_tf(x, torch.empty(b_size, 1, seq_len, freq_len, device=x.device, dtype=x.dtype))

        if self.topo_type == "mimo":
            if self.bf_type == "lstm":
                bf_w = self.bf_map(x)
            else:
                bf_w = self.bf_map(x)
                bf_w = bf_w.view(b_size, self.M, 2, seq_len, freq_len).permute(0, 3, 4, 1, 2)
            bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., 1]
            esti_x_r = (bf_w_r * inpt[..., 0] - bf_w_i * inpt[..., 1]).sum(dim=-1)
            esti_x_i = (bf_w_r * inpt[..., 1] + bf_w_i * inpt[..., 0]).sum(dim=-1)
            return torch.stack((esti_x_r, esti_x_i), dim=1)

        bf_w = self.bf_map(x).permute(0, 2, 3, 1)
        bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., 1]
        esti_x_r = bf_w_r * inpt[..., 0, 0] - bf_w_i * inpt[..., 0, 1]
        esti_x_i = bf_w_r * inpt[..., 0, 1] + bf_w_i * inpt[..., 0, 0]
        return torch.stack((esti_x_r, esti_x_i), dim=1)


def com_mag_mse_loss(esti: Tensor, label: Tensor, frame_list: Sequence[int]) -> Tensor:
    mask_for_loss = []
    utt_num = esti.size(0)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size(-1)), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        mask_for_loss = mask_for_loss[:, : esti.size(-2), :]
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((mag_esti - mag_label) ** 2.0) * mask_for_loss).sum() / mask_for_loss.sum()
    loss2 = (((esti - label) ** 2.0) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


def numParams(net: nn.Module) -> int:
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def zmain(args, net):
    batch_size = args.batch_size
    mics = args.mics
    sr = args.sr
    wav_len = int(args.wav_len * sr)
    win_size = int(args.win_size * sr)
    win_shift = int(args.win_shift * sr)
    fft_num = args.fft_num
    noisy_list, target_list, frame_list = [], [], []
    for _ in range(batch_size):
        noisy_list.append(torch.rand(wav_len, mics))
        target_list.append(torch.rand(wav_len))
        frame_list.append(wav_len // win_shift + 1)
    noisy_wav = nn.utils.rnn.pad_sequence(noisy_list, batch_first=True)
    target_wav = nn.utils.rnn.pad_sequence(target_list, batch_first=True)
    noisy_wav = noisy_wav.transpose(-2, -1).contiguous().view(batch_size * mics, wav_len)

    window = torch.hann_window(win_size).to(noisy_wav.device)
    noisy_stft = torch.stft(noisy_wav, fft_num, win_shift, win_size, window, return_complex=False)
    target_stft = torch.stft(target_wav, fft_num, win_shift, win_size, window, return_complex=False)
    _, freq_num, seq_len, _ = noisy_stft.shape
    noisy_stft = noisy_stft.view(batch_size, mics, freq_num, seq_len, -1).permute(0, 3, 2, 1, 4).to(next(net.parameters()).device)
    target_stft = target_stft.permute(0, 3, 2, 1).to(next(net.parameters()).device)
    noisy_mag = torch.norm(noisy_stft, dim=-1).pow(0.5)
    noisy_phase = torch.atan2(noisy_stft[..., 1], noisy_stft[..., 0])
    target_mag = torch.norm(target_stft, dim=1).pow(0.5)
    target_phase = torch.atan2(target_stft[:, 1, ...], target_stft[:, 0, ...])
    noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=-1)
    target_stft = torch.stack((target_mag * torch.cos(target_phase), target_mag * torch.sin(target_phase)), dim=1)

    esti_stft = net(noisy_stft)
    print(f"input size:{noisy_stft.shape} -> output size:{esti_stft.shape}, label size:{target_stft.shape}")
    loss = com_mag_mse_loss(esti_stft, target_stft, frame_list)
    print(f"Calculated loss value:{loss.item()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Lightweight Fourier convolutional attention EaBNet smoke test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mics", type=int, default=8)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--wav_len", type=float, default=1.0)
    parser.add_argument("--win_size", type=float, default=0.020)
    parser.add_argument("--win_shift", type=float, default=0.010)
    parser.add_argument("--fft_num", type=int, default=512)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--cd1", type=int, default=64)
    parser.add_argument("--norm_type", type=str, default="BN", choices=["BN", "IN", "cLN"])
    parser.add_argument("--is_causal", action="store_true", default=True)
    parser.add_argument("--bf_type", type=str, default="lstm", choices=["lstm", "cnn"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = EaBNet(c=args.channels, M=args.mics, embed_dim=args.embed_dim, cd1=args.cd1, norm_type=args.norm_type, bf_type=args.bf_type).to(device)
    net.eval()
    print(f"The number of trainable parameters is:{numParams(net)}")
    with torch.no_grad():
        zmain(args, net)
