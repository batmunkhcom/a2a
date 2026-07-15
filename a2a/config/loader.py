"""Config loader — discovers and loads a2a.yaml with search path priority.

Search order:
1. A2A_CONFIG environment variable
2. ./a2a.yaml (current working directory)
3. ~/.config/a2a/a2a.yaml (user config)
4. /etc/a2a/a2a.yaml (system config)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from a2a.config.schema import A2AConfig

# ── Default config path search order ───────────────────────────


def _search_paths() -> list[Path]:
    """Build list of paths to try, in priority order."""
    paths: list[Path] = []

    # 1. A2A_CONFIG env variable
    env_path = os.environ.get("A2A_CONFIG")
    if env_path:
        paths.append(Path(env_path))

    # 2. Working directory
    paths.append(Path.cwd() / "a2a.yaml")

    # 3. User config directory
    paths.append(Path.home() / ".config" / "a2a" / "a2a.yaml")

    # 4. System config
    paths.append(Path("/etc/a2a/a2a.yaml"))

    return paths


# ── Public API ─────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> A2AConfig:
    """Load the A2A configuration from the first available source.

    Args:
        path: Explicit path to a2a.yaml. If None, uses search order.

    Returns:
        Validated A2AConfig instance.

    Raises:
        ConfigNotFoundError: If no config file is found.
    """
    if path is not None:
        try:
            return A2AConfig.from_yaml(path)
        except FileNotFoundError as exc:
            raise ConfigNotFoundError(str(exc)) from exc

    for candidate in _search_paths():
        if candidate.exists():
            return A2AConfig.from_yaml(candidate)

    raise ConfigNotFoundError(
        "No a2a.yaml found. Searched:\n"
        + "\n".join(f"  - {p}" for p in _search_paths())
        + "\nSet A2A_CONFIG environment variable or create ./a2a.yaml"
    )


def load_plugin_config(plugin_path: str | Path) -> dict[str, Any]:
    """Load plugin-local config.yaml from a plugin directory.

    Searches for config.yaml next to the plugin.py file.

    Args:
        plugin_path: Path to the plugin directory or plugin.py file.

    Returns:
        Plugin config dict. Empty dict if no config file found.
    """
    import yaml

    pp = Path(plugin_path)
    if pp.is_file():
        pp = pp.parent

    config_file = pp / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}

    return {}


class ConfigNotFoundError(FileNotFoundError):
    """Raised when no a2a.yaml configuration file is found."""
