from .m4ri_lib import solve_gf2 as solve_gf2_m4ri  # type: ignore[import-not-found]
from .m4ri_lib import (
    solve_multiple_gf2 as solve_multiple_gf2_m4ri,  # type: ignore[import-not-found]
)

try:
    from .gf2toolkit_lib import (
        solve_gf2 as solve_gf2_toolkit,  # type: ignore[import-not-found]
    )
    from .gf2toolkit_lib import (
        solve_multiple_gf2 as solve_multiple_gf2_toolkit,  # type: ignore[import-not-found]
    )
except ImportError:
    solve_gf2_toolkit = None  # type: ignore[assignment,misc]
    solve_multiple_gf2_toolkit = None  # type: ignore[assignment,misc]

__all__ = [
    "solve_gf2_m4ri",
    "solve_multiple_gf2_m4ri",
    "solve_gf2_toolkit",
    "solve_multiple_gf2_toolkit",
]
