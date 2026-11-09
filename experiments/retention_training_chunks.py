"""Change retention training work partitioning without changing saved parameters."""
from contextlib import contextmanager

from experiments.causal_retention_mapping import CausalRetention


@contextmanager
def retention_training_chunks(model, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('Use a positive integer retention chunk size')
    if not model.training:
        raise ValueError('Scope this context to the training epoch only')
    modules = [(module, module.chunk_size) for module in model.modules()
               if isinstance(module, CausalRetention)]
    try:
        for module, _ in modules:
            module.chunk_size = chunk_size
        yield
    finally:
        for module, original in modules:
            module.chunk_size = original
