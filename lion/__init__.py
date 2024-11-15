"""The LION framework."""

import logging

from dotenv import load_dotenv

from .core.session import Branch
from .integrations.litellm_.imodel import iModel
from .protocols.operatives.step import Step
from .settings import Settings
from .version import __version__

load_dotenv()


__all__ = [
    "Settings",
    "__version__",
    "iModel",
    "Branch",
    "Step",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
