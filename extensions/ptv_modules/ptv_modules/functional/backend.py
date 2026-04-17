# Import the pre-compiled backend module
# This module is built during installation via setup.py
try:
    from . import _pvt_backend
except ImportError as e:
    raise ImportError(
        "Failed to import _pvt_backend. "
        "Please install ptv_modules by running: "
        "pip install -e /path/to/extensions/ptv_modules"
    ) from e

_backend = _pvt_backend
