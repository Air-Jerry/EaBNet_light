"""Unregistered standalone prototype: within-frame frequency modeling and causal state memory.

Method reference: TF-SkiMNet, Interspeech2025, doi:10.21437/Interspeech.2025-391.
This independent adaptation keeps the existing compressed8-mic frontend, adds
explicit current gain as an input, and owns every temporal cache per stream.
No author equivalence, real-data quality or target-device speed is established.
"""
import torch
from torch import nn
import torch.nn.functional as F

from light_causal_fast_model import ActivationLayerNorm


def recurrent_state(state, shape, value):
    if state is not None:
        if (not isinstance(state, tuple) or len(state) != 2
                or any(not torch.is_tensor(item) or item.shape != shape
                       or item.device != value.device or item.dtype != value.dtype for item in state)):
            raise ValueError('Temporal memory differs from this stream')
    return state


class BufferedTimeConv(nn.Module):
    def __init__(self, incoming, outgoing, kernel=5, groups=1):
        super().__init__()
        self.incoming, self.history = incoming, kernel - 1
        self.conv = nn.Conv1d(incoming, outgoing, kernel, groups=groups)

    def process(self, value, history, cache):
        batch, frames, bins, channels = value.shape
        if channels != self.incoming:
            raise ValueError('Temporal convolution channel mismatch')
        expected = (batch, bins, channels, self.history)
        if history is None:
            history = value.new_zeros(expected)
        elif (not torch.is_tensor(history) or history.shape != expected
              or history.device != value.device or history.dtype != value.dtype):
            raise ValueError('Convolution history differs from this stream')
        sequence = value.permute(0, 2, 3, 1)
        joined = torch.cat((history, sequence), -1)
        output = self.conv(joined.reshape(batch*bins, channels, -1))
        output = output.reshape(batch, bins, -1, frames).permute(0, 3, 1, 2)
        saved = joined[..., -self.history:].detach().clone() if cache else None
        return output, saved


class FrequencyStateBlock(nn.Module):
    def __init__(self, width, temporal_memory):
        super().__init__()
        self.width, self.temporal_memory = width, temporal_memory
        self.frequency_local = nn.Conv2d(width, width, (1, 5), padding=(0, 2))
        self.frequency_norm = ActivationLayerNorm(width)
        self.frequency_rnn = nn.LSTM(width, width, batch_first=True, bidirectional=True)
        self.frequency_projection = nn.Linear(2*width, width)
        self.local_norm = ActivationLayerNorm(width)
        self.local_time = BufferedTimeConv(width, width, groups=2)
        if temporal_memory:
            self.hidden_time = nn.LSTM(2*width, 2*width, batch_first=True)
            self.cell_time = nn.LSTM(2*width, 2*width, batch_first=True)
            self.hidden_norm = ActivationLayerNorm(2*width)
            self.cell_norm = ActivationLayerNorm(2*width)

    def process(self, value, initial, state, cache):
        state = {} if state is None else state
        if not isinstance(state, dict) or (state and set(state) != {'local', 'hidden', 'cell'}):
            raise ValueError('Use a complete independently owned frequency block state')
        if state and self.temporal_memory and any(state[name] is None for name in ('hidden', 'cell')):
            raise ValueError('Use a complete independently owned frequency block state')
        batch, frames, bins, width = value.shape
        value = value + F.silu(self.frequency_local(value.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        if initial is not None:
            initial = tuple(item.reshape(batch, frames, 2, width).permute(2, 0, 1, 3)
                            .reshape(2, batch*frames, width).contiguous() for item in initial)
        # The bidirectional sequence is frequency; each time frame is a batch item.
        frequency, summaries = self.frequency_rnn(
            self.frequency_norm(value).reshape(batch*frames, bins, width), initial)
        value = value + self.frequency_projection(frequency).reshape(batch, frames, bins, width)
        local, history = self.local_time.process(self.local_norm(value), state.get('local'), cache)
        value = value + F.silu(local)
        memories, updated = None, dict(local=history, hidden=None, cell=None)
        if self.temporal_memory:
            memories = []
            for name, summary in zip(('hidden', 'cell'), summaries):
                summary = summary.transpose(0, 1).reshape(batch, frames, 2*width)
                before = recurrent_state(state.get(name), (1, batch, 2*width), value)
                temporal, after = getattr(self, name+'_time')(summary, before)
                memories.append(summary + getattr(self, name+'_norm')(temporal))
                updated[name] = tuple(item.detach().clone() for item in after) if cache else None
            memories = tuple(memories)
        elif state.get('hidden') is not None or state.get('cell') is not None:
            raise ValueError('The final block has no outgoing temporal memory')
        return value, memories, updated if cache else None


class CausalFrequencyStateMemory(nn.Module):
    def __init__(self, width=32, blocks=6):
        super().__init__()
        if (isinstance(width, bool) or not isinstance(width, int) or width < 2 or width % 2
                or isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2):
            raise ValueError('Use an even width and at least two frequency-state blocks')
        self.M, self.is_causal = 8, True
        self.context = 4*(blocks+2)
        self.encoder = BufferedTimeConv(17, width)
        self.encoder_norm = ActivationLayerNorm(width)
        self.blocks = nn.ModuleList(FrequencyStateBlock(width, i < blocks-1) for i in range(blocks))
        self.decoder = BufferedTimeConv(width, 2)
        nn.init.zeros_(self.decoder.conv.weight)
        nn.init.zeros_(self.decoder.conv.bias)

    def process(self, inpt, state=None):
        if (inpt.ndim != 5 or inpt.shape[-3:] != (257, 8, 2) or min(inpt.shape[:2]) < 1
                or inpt.dtype != torch.float32):
            raise ValueError('Expected FP32 compressed STFT(B,T,257,8,2)')
        cache = state is not None
        memories = {} if state is None else state
        if not isinstance(memories, dict) or (memories and set(memories) != {'encoder', 'blocks', 'decoder'}):
            raise ValueError('Use a complete independently owned spectral-memory state')
        if memories and (not isinstance(memories['blocks'], tuple) or len(memories['blocks']) != len(self.blocks)
                         or memories['encoder'] is None or memories['decoder'] is None
                         or any(not isinstance(item, dict) or set(item) != {'local', 'hidden', 'cell'}
                                or item['local'] is None for item in memories['blocks'])):
            raise ValueError('Use a complete independently owned spectral-memory state')
        reference = inpt[..., 0, :]
        scale = torch.linalg.vector_norm(reference, dim=-1).mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
        features = torch.cat(((inpt/scale.unsqueeze(-1)).flatten(-2),
                              scale.log().expand(-1, -1, 257, -1)), -1)
        value, encoder = self.encoder.process(features, memories.get('encoder'), cache)
        value = F.silu(self.encoder_norm(value))
        initial, states = None, []
        for index, block in enumerate(self.blocks):
            previous = memories['blocks'][index] if memories else None
            value, initial, updated = block.process(value, initial, previous, cache)
            states.append(updated)
        delta, decoder = self.decoder.process(value, memories.get('decoder'), cache)
        output = (reference + scale*delta).permute(0, 3, 1, 2).contiguous()
        updated = dict(encoder=encoder, blocks=tuple(states), decoder=decoder) if cache else None
        return output, updated

    def forward(self, inpt, frame_lengths=None):
        if frame_lengths is not None:
            lengths = torch.as_tensor(frame_lengths)
            if lengths.numel() != inpt.shape[0] or torch.any(lengths < 1) or torch.any(lengths > inpt.shape[1]):
                raise ValueError('One valid frame length is required per sample')
        return self.process(inpt)[0]

    def forward_stream(self, inpt, state=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streaming requires eval mode and disabled gradients')
        return self.process(inpt, {} if state is None else state)
