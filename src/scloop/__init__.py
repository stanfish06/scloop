import warnings
from types import ModuleType

from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="colorspacious")

pl: ModuleType | None = None
try:
    from . import plotting as pl  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import plotting modules: {e}")

pp: ModuleType | None = None
try:
    from . import preprocessing as pp  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import preprocessing modules: {e}")

tl: ModuleType | None = None
try:
    from . import tools as tl  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import tools modules: {e}")

__all__ = [pkg for pkg in ["pl", "pp", "tl"] if globals().get(pkg) is not None]
