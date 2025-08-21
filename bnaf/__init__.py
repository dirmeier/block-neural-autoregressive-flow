"""bnaf: Block neural autoregressive flows in Flax."""

__version__ = "0.1.0"

from bnaf.block_naf import get_bnaf_model
from bnaf.inverse import get_inverter_model

__all__ = ["get_bnaf_model", "get_inverter_model"]
