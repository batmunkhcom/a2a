import a2a  # noqa: F401 — verify top-level import


def test_import_a2a():
    assert a2a.__version__ is not None
    assert a2a.__version__.startswith("0")


def test_subpackage_imports():
    """Verify all subpackage stubs are present and importable."""
    # fmt: off
    from a2a.config import defaults, loader, schema  # noqa: F401,I001
    from a2a.transport import client, codec, discovery, server  # noqa: F401,I001
    from a2a.tensor import dtype, extractor, injector, serializer  # noqa: F401,I001
    from a2a.projection import adapter, auto_trainer, dataset, registry, trainer  # noqa: F401,I001
    from a2a.agent import base, capabilities, manager, router  # noqa: F401,I001
    from a2a.protocol import errors, messages  # noqa: F401,I001
    from a2a.security import auth, tls  # noqa: F401,I001
    from a2a.monitoring import metrics, rate_limiter  # noqa: F401,I001
    from a2a.utils import async_utils, logging  # noqa: F401,I001
    from a2a.plugins.code_fixer import plugin as code_fixer  # noqa: F401,I001
    from a2a.plugins.log_reader import plugin as log_reader  # noqa: F401,I001
    # fmt: on

    assert loader
    assert defaults
    assert server
    assert client
    assert codec
    assert discovery
    assert extractor
    assert injector
    assert serializer
    assert dtype
    assert adapter
    assert trainer
    assert dataset
    assert auto_trainer
    assert registry
    assert base
    assert manager
    assert router
    assert capabilities
    assert messages
    assert errors
    assert auth
    assert tls
    assert metrics
    assert rate_limiter
    assert logging
    assert async_utils
    assert log_reader
    assert code_fixer
