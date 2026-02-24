from pathlib import Path

import click

from .check import check_program
from .syntax import Program
from .to_python import to_ast_program


@click.command(
    context_settings=dict(
        help_option_names=["-h", "--help"],
        max_content_width=120,
    ),
)
@click.option(
    "--check/--no-check",
    default=True,
    show_default=True,
    help="Enable or disable semantic analysis",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    default=None,
    help="Output file (defaults to <INPUT>.py)",
)
@click.argument(
    "input",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path),
)
def main(output: Path | None, check: bool, input: Path):
    program = Program.model_validate_json(input.read_text())

    if check:
        check_program(program)

    module = to_ast_program(program)

    (output or input.with_suffix(".py")).write_text(module)