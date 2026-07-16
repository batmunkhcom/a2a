"""A2A Protocol — AI-to-AI Latent Space Communication Protocol."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("batmunkh-a2a")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]
