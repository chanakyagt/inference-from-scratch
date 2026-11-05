from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 172
    max_positions: int = 128
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("dim must divide by n_heads and n_heads by n_kv_heads")

    @property
    def head_dim(self):
        return self.dim // self.n_heads
