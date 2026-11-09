"""Past-frame attention beside the existing spectral GRU; quality and CUDA cost are unverified."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.causal_decoder_monaural import CausalDecoderMonaural


class PastFrameAttention(nn.Module):
    width = 128
    heads = 4
    context = 128

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(816, 3 * self.width)
        self.output = nn.Linear(self.width, 816)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def process(self, value, state=None):
        batch, frames, features = value.shape
        if features != 816 or frames < 1:
            raise ValueError('Use nonempty current-frame spectral features')
        cache = state is not None
        width = self.width // self.heads
        q, k, v = self.qkv(value.float()).reshape(batch, frames, 3, self.heads, width).unbind(2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        previous = 0
        valid = torch.ones(frames, dtype=torch.bool, device=value.device)
        if cache:
            if not isinstance(state, dict) or (state and set(state) != {'k', 'v', 'valid'}):
                raise ValueError('Use complete independently owned attention state')
            previous = self.context - 1
            expected = (batch, self.heads, previous, width)
            if state:
                for name in ('k', 'v'):
                    tensor = state[name]
                    if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected
                            or tensor.dtype != torch.float32 or tensor.device != value.device):
                        raise ValueError('Attention cache differs from this stream')
                mask = state['valid']
                if (not isinstance(mask, torch.Tensor) or mask.shape != (previous,)
                        or mask.dtype != torch.bool or mask.device != value.device):
                    raise ValueError('Attention validity cache differs from this stream')
                before_k, before_v = state['k'], state['v']
            else:
                before_k = value.new_zeros(expected)
                before_v = value.new_zeros(expected)
                mask = torch.zeros(previous, dtype=torch.bool, device=value.device)
            k, v = torch.cat((before_k, k), 2), torch.cat((before_v, v), 2)
            valid = torch.cat((mask, valid))
        queries = previous + torch.arange(frames, device=value.device)
        keys = torch.arange(previous + frames, device=value.device)
        allowed = ((keys[None] <= queries[:, None])
                   & (keys[None] > queries[:, None] - self.context) & valid[None])
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        result = self.output(result.transpose(1, 2).reshape(batch, frames, self.width))
        updated = dict(k=k[:, :, -previous:].detach().clone(), v=v[:, :, -previous:].detach().clone(),
                       valid=valid[-previous:].detach().clone()) if cache else None
        return result, updated


class CausalTemporalMonaural(CausalDecoderMonaural):
    def __init__(self, wide_source):
        super().__init__(wide_source)
        with torch.random.fork_rng(devices=[]):
            self.temporal_memory = PastFrameAttention()

    @torch.no_grad()
    def load_warm(self, state):
        target = self.state_dict()
        extra = {name for name in target if name.startswith('temporal_memory.')}
        if set(state) != set(target) - extra:
            raise ValueError('Use the complete original monaural/spatial state')
        if any(value.shape != target[name].shape or not torch.isfinite(value).all() for name, value in state.items()):
            raise ValueError('Warm weights must have the original finite shapes')
        missing = self.load_state_dict(state, strict=False)
        assert set(missing.missing_keys) == extra and not missing.unexpected_keys
        nn.init.zeros_(self.temporal_memory.output.weight)
        nn.init.zeros_(self.temporal_memory.output.bias)

    def process(self, inpt, state=None):
        if inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1:
            raise ValueError('Expected compressed STFT(B,T,257,8,2)')
        cache = state is not None
        keys = {'base', 'monaural_hidden', 'temporal_memory'}
        if cache and (not isinstance(state, dict) or (state and set(state) != keys)):
            raise ValueError('Use complete independently owned temporal monaural state')
        memories = {} if state is None else state
        batch, frames = inpt.shape[:2]
        hidden = memories.get('monaural_hidden')
        if memories and (not isinstance(memories['base'], dict)
                or not isinstance(memories['temporal_memory'], dict)
                or not isinstance(hidden, torch.Tensor) or hidden.shape != (1, batch, 128)
                or hidden.dtype != torch.float32 or hidden.device != inpt.device):
            raise ValueError('Temporal monaural state differs from this stream')
        estimate, base_state = self.base.process(inpt, memories.get('base', {}) if cache else None)
        if estimate.shape != (batch, 2, frames, 257):
            raise ValueError('Causal source returned an incompatible spectrum')
        with torch.autocast(inpt.device.type, enabled=False):
            reference = inpt[..., 0, :].float()
            scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).clamp_min(1e-4)
            features = (torch.cat((reference, estimate.float().permute(0, 2, 3, 1)), -1)
                        / scale[..., None]).permute(0, 3, 1, 2)
        enabled = self.training and inpt.device.type == 'cuda'
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            value = features; skips = [features]
            for encoder in self.encoders:
                value = encoder(value); skips.append(value)
        with torch.autocast(inpt.device.type, enabled=False):
            normalized = self.recurrent_norm(value.permute(0, 2, 1, 3).reshape(batch, frames, 816).float())
            sequence, hidden = self.recurrent(normalized, hidden)
            memory, memory_state = self.temporal_memory.process(normalized,
                memories.get('temporal_memory', {}) if cache else None)
        with torch.autocast(inpt.device.type, dtype=torch.bfloat16, enabled=enabled):
            projected = self.projection(sequence) + memory
            value = value + projected.reshape(batch, frames, 48, 17).permute(0, 2, 1, 3)
            for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips[:-1])):
                value = up(value)
                if value.shape[-1] != skip.shape[-1]:
                    raise RuntimeError('Frequency decoder lost alignment')
                value = decoder(torch.cat((value, skip), 1))
            correction = self.output(value).float() * scale[:, None]
        output = estimate.float() + correction
        updated = dict(base=base_state, monaural_hidden=hidden.detach().clone(),
                       temporal_memory=memory_state) if cache else None
        return output, updated
