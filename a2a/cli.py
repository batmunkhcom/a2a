"""A2A Protocol CLI — powered by typer."""

import typer

app = typer.Typer(
    name="a2a",
    help="A2A Protocol — AI-to-AI Latent Space Communication Protocol",
    no_args_is_help=True,
)


@app.command()
def serve(
    config: str = typer.Option(
        None, "--config", "-c", help="Path to a2a.yaml config file"
    ),
    port: int = typer.Option(9090, "--port", "-p", help="gRPC server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="gRPC server host"),
) -> None:
    """Start A2A runtime with plugins."""
    typer.echo(f"Starting A2A runtime on {host}:{port}...")
    typer.echo("(stub — populated in Sprint 3)")


@app.command()
def discover() -> None:
    """Discover A2A agents on the network."""
    typer.echo("Discovering A2A agents...")
    typer.echo("(stub — populated in Sprint 5)")


@app.command(name="config")
def config_command(
    action: str = typer.Argument("validate", help="Action: validate | show"),
    section: str = typer.Option(
        None, "--section", "-s", help="Config section to show (models, plugins, routes)"
    ),
) -> None:
    """Validate or show A2A configuration."""
    if action == "validate":
        typer.echo("Validating a2a.yaml...")
        typer.echo("(stub — populated in Sprint 3)")
    elif action == "show":
        if section:
            typer.echo(f"Showing config section: {section}")
        else:
            typer.echo("Showing full config...")
        typer.echo("(stub — populated in Sprint 3)")
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(code=1)


@app.command()
def train_projection(
    src: str = typer.Option(..., "--src", help="Source model ID"),
    tgt: str = typer.Option(..., "--tgt", help="Target model ID"),
    data: str = typer.Option(
        "./projection_corpus.txt", "--data", "-d", help="Path to training corpus"
    ),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Training epochs"),
) -> None:
    """Train a projection model between two model architectures."""
    typer.echo(f"Training projection: {src} → {tgt}")
    typer.echo(f"Corpus: {data}, Epochs: {epochs}")
    typer.echo("(stub — populated in Sprint 4)")


@app.command()
def benchmark(
    task: str = typer.Option(
        "error-resolution", "--task", "-t", help="Benchmark task"
    ),
    iterations: int = typer.Option(
        100, "--iterations", "-n", help="Number of iterations"
    ),
) -> None:
    """Benchmark A2A vs Text API."""
    typer.echo(f"Running benchmark: {task} ({iterations} iterations)")
    typer.echo("(stub — populated in Sprint 5)")


@app.command()
def version() -> None:
    """Show A2A Protocol version."""
    from a2a._version import __version__

    typer.echo(f"A2A Protocol v{__version__}")


if __name__ == "__main__":
    app()
