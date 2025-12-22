import warnings

from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="colorspacious")

try:
    from . import plotting as pl
except ImportError as e:
    logger.warning(f"Could not import plotting modules: {e}")
    pl = None

try:
    from . import preprocessing as pp
except ImportError as e:
    logger.warning(f"Could not import preprocessing modules: {e}")
    pp = None

try:
    from . import tools as tl
except ImportError as e:
    logger.warning(f"Could not import tools modules: {e}")
    tl = None

__all__ = [pkg for pkg in ["pl", "pp", "tl"] if globals().get(pkg) is not None]
