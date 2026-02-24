import pytest

from L2 import syntax as L2
from L3 import syntax as L3
from L3.eliminate_letrec import Context, eliminate_letrec_program, eliminate_letrec_term


def test_eliminate_letrec_term_let() -> None:
    term = L3.Let(
        bindings=[("x", L3.Immediate(value=0))],
        body=L3.Reference(name="x"),
    )

    context: Context = {}

    expected = L2.Let(
        bindings=[("x", L2.Immediate(value=0))],
        body=L2.Reference(name="x"),
    )

    match term:
        case L3.Let():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_letrec() -> None:
    term = L3.LetRec(
        bindings=[
            (
                "f",
                L3.Abstract(
                    parameters=["x"],
                    body=L3.Apply(
                        target=L3.Reference(name="f"),
                        arguments=[L3.Reference(name="x")],
                    ),
                ),
            )
        ],
        body=L3.Apply(
            target=L3.Reference(name="f"),
            arguments=[L3.Immediate(value=0)],
        ),
    )

    context: Context = {}

    expected = L2.Let(
        bindings=[("f", L2.Allocate(count=1))],
        body=L2.Begin(
            effects=[
                L2.Store(
                    base=L2.Reference(name="f"),
                    index=0,
                    value=L2.Abstract(
                        parameters=["x"],
                        body=L2.Apply(
                            target=L2.Load(base=L2.Reference(name="f"), index=0),
                            arguments=[L2.Reference(name="x")],
                        ),
                    ),
                ),
            ],
            value=L2.Apply(
                target=L2.Load(base=L2.Reference(name="f"), index=0),
                arguments=[L2.Immediate(value=0)],
            ),
        ),
    )

    match term:
        case L3.LetRec():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_reference_nonrecursive() -> None:
    term = L3.Reference(name="x")

    context: Context = {}

    expected = L2.Reference(name="x")

    match term:
        case L3.Reference():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_reference_recursive() -> None:
    term = L3.Reference(name="f")

    context: Context = {"f": None}

    expected = L2.Load(
        base=L2.Reference(name="f"),
        index=0,
    )

    match term:
        case L3.Reference():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_abstract() -> None:
    term = L3.Abstract(
        parameters=["x"],
        body=L3.Reference(name="f"),
    )

    context: Context = {"f": None}

    expected = L2.Abstract(
        parameters=["x"],
        body=L2.Load(base=L2.Reference(name="f"), index=0),
    )

    match term:
        case L3.Abstract():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_apply() -> None:
    term = L3.Apply(
        target=L3.Reference(name="g"),
        arguments=[L3.Reference(name="f")],
    )

    context: Context = {"f": None}

    expected = L2.Apply(
        target=L2.Reference(name="g"),
        arguments=[L2.Load(base=L2.Reference(name="f"), index=0)],
    )

    match term:
        case L3.Apply():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_immediate() -> None:
    term = L3.Immediate(value=7)

    context: Context = {}

    expected = L2.Immediate(value=7)

    match term:
        case L3.Immediate():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_primitive() -> None:
    term = L3.Primitive(
        operator="+",
        left=L3.Reference(name="f"),
        right=L3.Immediate(value=1),
    )

    context: Context = {"f": None}

    expected = L2.Primitive(
        operator="+",
        left=L2.Load(base=L2.Reference(name="f"), index=0),
        right=L2.Immediate(value=1),
    )

    match term:
        case L3.Primitive():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_branch() -> None:
    term = L3.Branch(
        operator="<",
        left=L3.Reference(name="f"),
        right=L3.Immediate(value=2),
        consequent=L3.Immediate(value=1),
        otherwise=L3.Immediate(value=0),
    )

    context: Context = {"f": None}

    expected = L2.Branch(
        operator="<",
        left=L2.Load(base=L2.Reference(name="f"), index=0),
        right=L2.Immediate(value=2),
        consequent=L2.Immediate(value=1),
        otherwise=L2.Immediate(value=0),
    )

    match term:
        case L3.Branch():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_allocate() -> None:
    term = L3.Allocate(count=3)

    context: Context = {}

    expected = L2.Allocate(count=3)

    match term:
        case L3.Allocate():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_load() -> None:
    term = L3.Load(
        base=L3.Allocate(count=1),
        index=0,
    )

    context: Context = {}

    expected = L2.Load(
        base=L2.Allocate(count=1),
        index=0,
    )

    match term:
        case L3.Load():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_store() -> None:
    term = L3.Store(
        base=L3.Allocate(count=1),
        index=0,
        value=L3.Reference(name="f"),
    )

    context: Context = {"f": None}

    expected = L2.Store(
        base=L2.Allocate(count=1),
        index=0,
        value=L2.Load(base=L2.Reference(name="f"), index=0),
    )

    match term:
        case L3.Store():
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_begin() -> None:
    term = L3.Begin(
        effects=[L3.Reference(name="f")],
        value=L3.Immediate(value=0),
    )

    context: Context = {"f": None}

    expected = L2.Begin(
        effects=[L2.Load(base=L2.Reference(name="f"), index=0)],
        value=L2.Immediate(value=0),
    )

    match term:
        case L3.Begin():  # pragma: no branch
            actual = eliminate_letrec_term(term, context)
            assert actual == expected


def test_eliminate_letrec_term_unhandled_raises() -> None:
    term = object()

    match term:
        case _:
            with pytest.raises(TypeError):
                eliminate_letrec_term(term, context={})  # type: ignore[arg-type]


def test_eliminate_letrec_program() -> None:
    program = L3.Program(
        parameters=["x"],
        body=L3.Immediate(value=0),
    )

    expected = L2.Program(
        parameters=["x"],
        body=L2.Immediate(value=0),
    )

    match program:
        case L3.Program():  # pragma: no branch
            actual = eliminate_letrec_program(program)
            assert actual == expected