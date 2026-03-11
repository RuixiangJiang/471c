import pytest
from L3.syntax import (
    Abstract,
    Allocate,
    Apply,
    Begin,
    Branch,
    Immediate,
    Let,
    LetRec,
    Load,
    Primitive,
    Program,
    Reference,
    Store,
)
from L3.uniqify import Context, uniqify_program, uniqify_term
from util.sequential_name_generator import SequentialNameGenerator


def test_uniqify_term_reference():
    term = Reference(name="x")

    context: Context = {"x": "y"}
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh=fresh)

    expected = Reference(name="y")

    assert actual == expected


def test_uniqify_immediate():
    term = Immediate(value=42)

    context: Context = dict[str, str]()
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh)

    expected = Immediate(value=42)

    assert actual == expected


def test_uniqify_term_let():
    term = Let(
        bindings=[
            ("x", Immediate(value=1)),
            ("y", Reference(name="x")),
        ],
        body=Apply(
            target=Reference(name="x"),
            arguments=[
                Reference(name="y"),
            ],
        ),
    )

    context: Context = {"x": "y"}
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh)

    expected = Let(
        bindings=[
            ("x0", Immediate(value=1)),
            ("y0", Reference(name="y")),
        ],
        body=Apply(
            target=Reference(name="x0"),
            arguments=[
                Reference(name="y0"),
            ],
        ),
    )

    assert actual == expected


def test_uniqify_term_letrec():
    term = LetRec(
        bindings=[
            (
                "f",
                Abstract(
                    parameters=["x"],
                    body=Apply(
                        target=Reference(name="f"),
                        arguments=[Reference(name="x")],
                    ),
                ),
            )
        ],
        body=Reference(name="f"),
    )

    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, {}, fresh)

    expected = LetRec(
        bindings=[
            (
                "f0",
                Abstract(
                    parameters=["x0"],
                    body=Apply(
                        target=Reference(name="f0"),
                        arguments=[Reference(name="x0")],
                    ),
                ),
            )
        ],
        body=Reference(name="f0"),
    )

    assert actual == expected


def test_uniqify_term_structural_forms():
    term = Begin(
        effects=[
            Store(
                base=Reference(name="vec"),
                index=0,
                value=Primitive(
                    operator="+",
                    left=Reference(name="x"),
                    right=Immediate(value=1),
                ),
            )
        ],
        value=Branch(
            operator="==",
            left=Load(base=Reference(name="vec"), index=0),
            right=Immediate(value=0),
            consequent=Reference(name="x"),
            otherwise=Apply(
                target=Reference(name="f"),
                arguments=[Reference(name="x")],
            ),
        ),
    )

    context: Context = {"vec": "vec7", "x": "x3", "f": "f9"}
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh)

    expected = Begin(
        effects=[
            Store(
                base=Reference(name="vec7"),
                index=0,
                value=Primitive(
                    operator="+",
                    left=Reference(name="x3"),
                    right=Immediate(value=1),
                ),
            )
        ],
        value=Branch(
            operator="==",
            left=Load(base=Reference(name="vec7"), index=0),
            right=Immediate(value=0),
            consequent=Reference(name="x3"),
            otherwise=Apply(
                target=Reference(name="f9"),
                arguments=[Reference(name="x3")],
            ),
        ),
    )

    assert actual == expected


def test_uniqify_program_renames_parameters_and_nested_binders():
    program = Program(
        parameters=["x", "f"],
        body=Let(
            bindings=[
                ("x", Immediate(value=1)),
                (
                    "g",
                    Abstract(
                        parameters=["f"],
                        body=Apply(
                            target=Reference(name="f"),
                            arguments=[Reference(name="x")],
                        ),
                    ),
                ),
            ],
            body=Apply(
                target=Reference(name="g"),
                arguments=[Reference(name="f")],
            ),
        ),
    )

    _, actual = uniqify_program(program)

    expected = Program(
        parameters=["x0", "f0"],
        body=Let(
            bindings=[
                ("x1", Immediate(value=1)),
                (
                    "g0",
                    Abstract(
                        parameters=["f1"],
                        body=Apply(
                            target=Reference(name="f1"),
                            arguments=[Reference(name="x0")],
                        ),
                    ),
                ),
            ],
            body=Apply(
                target=Reference(name="g0"),
                arguments=[Reference(name="f0")],
            ),
        ),
    )

    assert actual == expected

def test_uniqify_term_invalid_term_raises_type_error():
    fresh = SequentialNameGenerator()

    with pytest.raises(TypeError, match="Unhandled L3 term in uniqify_term"):
        uniqify_term("not a term", {}, fresh)


def test_uniqify_term_allocate():
    term = Allocate(count=0)

    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, {}, fresh)

    expected = Allocate(count=0)
    assert actual == expected