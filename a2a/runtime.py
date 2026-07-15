"""A2ARuntime — application entry point that wires everything together.

Startup sequence:
1. Load a2a.yaml configuration
2. Set up structured logging
3. Load/enable plugins via PluginManager
4. Start gRPC transport server
5. Start metrics/health server
"""

from __future__ import annotations

import logging
from pathlib import Path

from a2a.agent.base import PluginEntry
from a2a.agent.manager import PluginManager
from a2a.agent.router import RouteConfig, SemanticRouter
from a2a.config.loader import load_config
from a2a.config.schema import A2AConfig
from a2a.transport.server import A2AServer

logger = logging.getLogger("a2a.runtime")


class A2ARuntime:
    """Orchestrates the full A2A runtime lifecycle."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config: A2AConfig | None = None
        self._config_path = config_path
        self._plugin_manager = PluginManager()
        self._router = SemanticRouter()
        self._route_config = RouteConfig()
        self._server: A2AServer | None = None
        self._started = False

    # ── Lifecycle ───────────────────────────────────────────────

    async def start(self) -> None:
        """Start the full A2A runtime.

        Raises:
            ConfigNotFoundError: If config cannot be found.
            ValueError: If config validation fails.
        """
        # 1. Load configuration
        self._config = load_config(self._config_path)
        warnings = self._config.validate_config()
        for w in warnings:
            logger.warning("Config warning: %s", w)

        # 2. Setup structured logging
        _setup_logging(self._config.runtime.log_level)

        # 3. Load plugins
        await self._load_plugins()

        # 4. Setup routes
        self._route_config = RouteConfig.from_dict(self._config.routes)

        # 5. Start transport server
        self._server = A2AServer(
            host=self._config.transport.host,
            port=self._config.transport.port,
            plugin_manager=self._plugin_manager,
        )
        self._server.start()

        self._started = True
        logger.info(
            "A2A Runtime started on %s:%d — %d plugins loaded",
            self._config.transport.host,
            self._config.transport.port,
            self._plugin_manager.plugin_count,
        )

    async def stop(self) -> None:
        """Gracefully stop the runtime."""
        if self._server:
            self._server.stop()
        self._started = False
        logger.info("A2A Runtime stopped")

    # ── Query ───────────────────────────────────────────────────

    @property
    def config(self) -> A2AConfig:
        if self._config is None:
            raise RuntimeError("Runtime not started — no config loaded")
        return self._config

    @property
    def plugin_manager(self) -> PluginManager:
        return self._plugin_manager

    @property
    def router(self) -> SemanticRouter:
        return self._router

    @property
    def is_running(self) -> bool:
        return self._started

    # ── Internal ───────────────────────────────────────────────

    async def _load_plugins(self) -> None:
        """Load all enabled plugins from config."""
        if self._config is None:
            return

        for pid, cfg in self._config.plugins.items():
            if not cfg.enabled:
                continue

            entry = PluginEntry(
                plugin_id=pid,
                module=cfg.module,
                class_name=cfg.class_name,
                model=cfg.model,
                enabled=cfg.enabled,
                priority=cfg.priority,
                config_path=cfg.config_path,
            )

            try:
                plugin = self._plugin_manager.load_plugin(
                    entry, self._config
                )
                logger.info(
                    "Loaded plugin: %s (model=%s)",
                    pid,
                    cfg.model or "none",
                )
                await plugin.on_load(self)
            except Exception as exc:
                logger.error(
                    "Failed to load plugin '%s': %s",
                    pid,
                    exc,
                    exc_info=True,
                )


# ── Helpers ────────────────────────────────────────────────────


def _setup_logging(level: str) -> None:
    """Configure structured JSON logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


__all__ = ["A2ARuntime"]
