from pathlib import Path

import click
from L1.to_python import to_ast_program
from L2.cps_convert import cps_convert_program
from L2.optimize import optimize_program
from L3.check import check_program
from L3.eliminate_letrec import eliminate_letrec_program
from L3.uniqify import uniqify_program

from L4.convert import convert_to_l3, dummy_parse


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
    "--optimize/--no-optimize",
    default=True,
    show_default=True,
    help="Enable or disable optimization",
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
def main(
    output: Path | None,
    check: bool,
    optimize: bool,
    input: Path,
) -> None:
    l4 = dummy_parse(input.read_text())
    l3 = convert_to_l3(l4)

    if check:
        check_program(l3)

    fresh, l3 = uniqify_program(l3)

    l2 = eliminate_letrec_program(l3)

    if optimize:
        l2 = optimize_program(l2)

    l1 = cps_convert_program(l2, fresh)

    module = to_ast_program(l1)

    (output or input.with_suffix(".py")).write_text(module)
