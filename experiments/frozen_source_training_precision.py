"""Use the inherited training precision on a stateless frozen source, without enabling its gradients."""
from contextlib import contextmanager
import torch


@contextmanager
def frozen_source_training_precision(model):
    if any(p.requires_grad for p in model.base.parameters()):
        raise ValueError('Freeze every source parameter before using its training precision')
    dropout_types=(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d,torch.nn.Dropout3d,
                   torch.nn.AlphaDropout,torch.nn.FeatureAlphaDropout)
    recurrent_types=(torch.nn.RNN,torch.nn.GRU,torch.nn.LSTM,torch.nn.MultiheadAttention)
    if any(getattr(m,'track_running_stats',False) or isinstance(m,dropout_types) and m.p
           or isinstance(m,recurrent_types) and m.dropout
           for m in model.base.modules()):
        raise ValueError('The frozen source must have no running statistics or nonzero dropout')
    original=model.train
    existing=model.__dict__.get('train')

    def train(mode=True):
        original(mode)
        if mode:model.base.train(True)
        return model

    model.train=train
    try:yield
    finally:
        if existing is None:del model.train
        else:model.train=existing
        model.train(model.training)
