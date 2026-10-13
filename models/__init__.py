"""Recovered single-EDFA models; see docs/SOURCE_MAP.md for provenance."""

MODEL_NAMES = ("wide-cnn", "residual-attention-cnn", "mlp")


def build_model(name):
    if name == "wide-cnn":
        from .wide_cnn import NN
    elif name == "residual-attention-cnn":
        from .residual_attention_cnn import NN
    elif name == "mlp":
        from .mlp import NN
    else:
        raise ValueError(f"Unknown model: {name}")
    return NN()
