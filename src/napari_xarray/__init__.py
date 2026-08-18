try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._sample_data import cells3d

__all__ = (
    "cells3d",
)
