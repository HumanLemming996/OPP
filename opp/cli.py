"""
OPP CLI — Command-line tool for minting and verifying images.

Usage:
    opp mint <image> --generator <name> --server <url>
    opp verify <image> --server <url>
    opp status <id> --server <url>
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

DEFAULT_SERVER = "http://localhost:8000"


def _encode_image(image_path: str) -> str:
    """Read and base64-encode an image file."""
    path = Path(image_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {image_path}")
        sys.exit(1)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _match_color(level: str) -> str:
    """Get color for match level."""
    return {
        "exact": "bright_green",
        "likely": "green",
        "possible": "yellow",
        "none": "red",
    }.get(level, "white")


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="opp")
def cli():
    """OPP — Open Provenance Protocol

    Shazam for AI-generated images. Mint and verify provenance signatures.
    """
    pass


# ---------------------------------------------------------------------------
# Mint Command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--generator", "-g", required=True, help="Generator name (e.g. 'dall-e-3')")
@click.option("--model-version", "-m", default=None, help="Model version")
@click.option("--server", "-s", default=DEFAULT_SERVER, help="OPP server URL")
@click.option("--tags", "-t", multiple=True, help="Optional tags")
def mint(image_path: str, generator: str, model_version: str, server: str, tags: tuple):
    """Mint a provenance signature for an image."""
    console.print(f"\n[bold blue]⚡ Minting signature...[/bold blue]")
    console.print(f"   Image: {image_path}")
    console.print(f"   Generator: {generator}")
    console.print(f"   Server: {server}\n")

    image_b64 = _encode_image(image_path)

    payload = {
        "image_base64": image_b64,
        "metadata": {
            "generator": generator,
            "model_version": model_version,
            "tags": list(tags),
        },
    }

    try:
        with console.status("[bold green]Generating 3-layer signature..."):
            response = httpx.post(
                f"{server}/v1/mint",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to server at {server}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} — {e.response.text}")
        sys.exit(1)

    result = response.json()

    panel = Panel(
        f"[green]✓ Signature minted successfully[/green]\n\n"
        f"[bold]ID:[/bold]    {result['signature_id']}\n"
        f"[bold]SHA-256:[/bold] {result['l1_hash'][:16]}...{result['l1_hash'][-8:]}\n"
        f"[bold]PDQ:[/bold]    {result['l2_hash'][:16]}...{result['l2_hash'][-8:]}",
        title="[bold]OPP Mint Result[/bold]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# Verify Command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--server", "-s", default=DEFAULT_SERVER, help="OPP server URL")
@click.option("--max-results", "-n", default=10, help="Max results to return")
@click.option("--min-similarity", default=0.80, help="Minimum similarity threshold")
def verify(image_path: str, server: str, max_results: int, min_similarity: float):
    """Verify an image against the provenance registry."""
    console.print(f"\n[bold blue]🔍 Verifying image...[/bold blue]")
    console.print(f"   Image: {image_path}")
    console.print(f"   Server: {server}\n")

    image_b64 = _encode_image(image_path)

    payload = {
        "image_base64": image_b64,
        "max_results": max_results,
        "min_similarity": min_similarity,
    }

    try:
        with console.status("[bold green]Running tiered matching pipeline..."):
            response = httpx.post(
                f"{server}/v1/verify",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to server at {server}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error:[/red] {e.response.status_code} — {e.response.text}")
        sys.exit(1)

    result = response.json()
    matches = result.get("matches", [])

    if not matches:
        panel = Panel(
            "[yellow]No matches found.[/yellow]\n\n"
            "This image does not appear in the OPP registry.\n"
            "It may not be AI-generated, or the generator hasn't minted it.",
            title="[bold]OPP Verify Result[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
        )
        console.print(panel)
        return

    # Display matches table
    table = Table(
        title=f"[bold]OPP Matches ({len(matches)} found)[/bold]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Level", style="bold")
    table.add_column("Similarity", justify="right")
    table.add_column("PDQ Dist", justify="right")
    table.add_column("Generator", style="cyan")
    table.add_column("Signature ID")

    for match in matches:
        level = match["match_level"]
        color = _match_color(level)
        table.add_row(
            f"[{color}]{level.upper()}[/{color}]",
            f"{match['cosine_similarity']:.4f}",
            str(match.get("pdq_hamming_distance", "—")),
            match["metadata"]["generator"],
            match["signature_id"][:12] + "...",
        )

    console.print(table)
    console.print(
        f"\n   Searched [bold]{result['total_signatures_searched']}[/bold] signatures"
    )


# ---------------------------------------------------------------------------
# Status Command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("signature_id")
@click.option("--server", "-s", default=DEFAULT_SERVER, help="OPP server URL")
def status(signature_id: str, server: str):
    """Look up a specific signature by ID."""
    try:
        response = httpx.get(
            f"{server}/v1/signatures/{signature_id}",
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        console.print(f"[red]Error:[/red] Cannot connect to server at {server}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            console.print(f"[yellow]Signature not found:[/yellow] {signature_id}")
        else:
            console.print(f"[red]Error:[/red] {e.response.status_code}")
        sys.exit(1)

    result = response.json()
    console.print_json(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Health Command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--server", "-s", default=DEFAULT_SERVER, help="OPP server URL")
def health(server: str):
    """Check server health and stats."""
    try:
        response = httpx.get(f"{server}/health", timeout=10.0)
        result = response.json()

        status_color = "green" if result.get("status") == "healthy" else "red"
        console.print(f"\n[{status_color}]● {result['status'].upper()}[/{status_color}]")
        console.print(f"  Version:    {result.get('version', '?')}")
        console.print(f"  Protocol:   {result.get('protocol', '?')}")
        console.print(f"  Signatures: {result.get('total_signatures', '?')}")
        console.print(f"  Qdrant:     {result.get('qdrant_status', '?')}\n")
    except httpx.ConnectError:
        console.print(f"[red]● Cannot connect to {server}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
