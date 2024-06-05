
from .Classification import Classification
# from .dmvfn import DMVFN

method_maps = {
    'classification': Classification,
}

__all__ = [
    "classification"
]