"""
Main entry point. This file's only job is wiring commands onto the Typer app —
it contains no logic of its own.

To add a new command later:
  1. write a function in commands/your_command.py (see ingest.py / query.py for the pattern)
  2. import it below and add one line: app.command()(your_command)

Run via:
    python -m ragapp ingest
    python -m ragapp query "..."
  or, once installed (pip install -e .):
    ragapp ingest
    ragapp query "..."
"""

import typer

from ragapp.commands.ingest import ingest
from ragapp.commands.query import query

app = typer.Typer(help="Ask My Docs — a local RAG CLI.")

app.command()(ingest)
app.command()(query)


if __name__ == "__main__":
    app()