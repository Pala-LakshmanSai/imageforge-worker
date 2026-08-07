"""ImageForge's durable single-GPU worker."""

from .app import create_app
from .config import WorkerSettings
from .constants import WORKER_VERSION

__all__ = ["WorkerSettings", "create_app"]
# Derived rather than duplicated: a second hand-maintained version literal had
# already drifted to 0.1.3 while the served contract was 0.1.4.
__version__ = WORKER_VERSION
