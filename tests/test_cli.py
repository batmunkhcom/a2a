from typer.testing import CliRunner

from a2a.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "A2A Protocol" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "A2A Protocol" in result.stdout


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start A2A runtime" in result.stdout


def test_config_validate():
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0
