"""Semantic label router — matches tensor metadata labels to plugins.

Handles:
- Finding plugins that listen to a specific semantic label.
- Multi-cast routing (one tensor → multiple plugins).
- Priority-based ordering when multiple plugins match.
"""

from __future__ import annotations

from a2a.agent.base import BasePlugin, PluginRegistry


class SemanticRouter:
    """Routes tensors to plugins based on semantic labels."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self._registry = registry or PluginRegistry()

    def route(
        self,
        label: str,
    ) -> list[BasePlugin]:
        """Find all plugins that subscribe to a semantic label.

        Results are ordered by plugin priority (higher = first).

        Args:
            label: Semantic label string, e.g. "error_context".

        Returns:
            Ordered list of matching plugins.
        """
        plugins = self._registry.get_subscribers(label)
        # Sort by priority descending
        plugins.sort(
            key=lambda p: -min(
                (c.priority for c in p.get_capabilities()), default=0
            )
        )
        return plugins

    def multi_route(
        self,
        labels: list[str],
    ) -> dict[str, list[BasePlugin]]:
        """Route multiple labels at once.

        Args:
            labels: List of semantic label strings.

        Returns:
            Dict mapping each label to its subscriber plugins.
        """
        return {label: self.route(label) for label in labels}

    def has_subscribers(self, label: str) -> bool:
        """Check if any plugin is listening for a given label."""
        return len(self._registry.get_subscribers(label)) > 0

    @property
    def active_labels(self) -> list[str]:
        """All semantic labels currently being listened to."""
        return self._registry.labels


class RouteConfig:
    """Parsed routing rules from a2a.yaml routes section."""

    def __init__(self, routes: dict[str, list[str]] | None = None) -> None:
        """Routes config: {"label": ["plugin_id", ...]}"""
        self._routes: dict[str, list[str]] = routes or {}

    def targets_for(self, label: str) -> list[str]:
        """Get target plugin IDs for a semantic label."""
        return list(self._routes.get(label, []))

    def all_labels(self) -> list[str]:
        return list(self._routes.keys())

    def add_route(self, label: str, target_plugin_id: str) -> None:
        self._routes.setdefault(label, []).append(target_plugin_id)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> RouteConfig:
        return cls(routes=data)


__all__ = ["SemanticRouter", "RouteConfig"]
