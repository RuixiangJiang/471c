import pytest

from L3.check import check_program, check_term
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


def test_check_program() -> None:
    program = Program(
        parameters=["x"],
        body=Reference(name="x"),
    )

    match program:
        case Program():  # pragma: no branch
            check_program(program)


def test_check_program_duplicate_parameters() -> None:
    program = Program(
        parameters=["x", "x"],
        body=Immediate(value=0),
    )

    match program:
        case Program():  # pragma: no branch
            with pytest.raises(ValueError):
                check_program(program)


def test_check_term_let() -> None:
    term = Let(
        bindings=[("x", Immediate(value=1))],
        body=Reference(name="x"),
    )

    match term:
        case Let():
            check_term(term, context={})


def test_check_term_let_duplicate_binders() -> None:
    term = Let(
        bindings=[("x", Immediate(value=1)), ("x", Immediate(value=2))],
        body=Reference(name="x"),
    )

    match term:
        case Let():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_check_term_letrec() -> None:
    term = LetRec(
        bindings=[("f", Abstract(parameters=["x"], body=Reference(name="x")))],
        body=Apply(
            target=Reference(name="f"),
            arguments=[Immediate(value=0)],
        ),
    )

    match term:
        case LetRec():
            check_term(term, context={})


def test_check_term_letrec_duplicate_binders() -> None:
    term = LetRec(
        bindings=[("f", Immediate(value=0)), ("f", Immediate(value=1))],
        body=Immediate(value=0),
    )

    match term:
        case LetRec():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_check_term_reference_bound() -> None:
    term = Reference(name="x")

    match term:
        case Reference():
            check_term(term, context={"x": None})


def test_check_term_reference_unbound() -> None:
    term = Reference(name="x")

    match term:
        case Reference():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_check_term_abstract() -> None:
    term = Abstract(
        parameters=["x"],
        body=Reference(name="x"),
    )

    match term:
        case Abstract():
            check_term(term, context={})


def test_check_term_abstract_duplicate_parameters() -> None:
    term = Abstract(
        parameters=["x", "x"],
        body=Reference(name="x"),
    )

    match term:
        case Abstract():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_check_term_apply() -> None:
    term = Apply(
        target=Abstract(parameters=["x"], body=Reference(name="x")),
        arguments=[Immediate(value=1)],
    )

    match term:
        case Apply():
            check_term(term, context={})


def test_check_term_immediate() -> None:
    term = Immediate(value=0)

    match term:
        case Immediate():
            check_term(term, context={})


def test_check_term_primitive() -> None:
    term = Primitive(
        operator="+",
        left=Immediate(value=1),
        right=Immediate(value=2),
    )

    match term:
        case Primitive():
            check_term(term, context={})


def test_check_term_branch() -> None:
    term = Branch(
        operator="<",
        left=Immediate(value=1),
        right=Immediate(value=2),
        consequent=Immediate(value=1),
        otherwise=Immediate(value=2),
    )

    match term:
        case Branch():
            check_term(term, context={})


def test_check_term_allocate() -> None:
    term = Allocate(count=3)

    match term:
        case Allocate():
            check_term(term, context={})


def test_check_term_load() -> None:
    term = Load(
        base=Allocate(count=1),
        index=0,
    )

    match term:
        case Load():
            check_term(term, context={})


def test_check_term_store() -> None:
    term = Store(
        base=Allocate(count=1),
        index=0,
        value=Immediate(value=7),
    )

    match term:
        case Store():
            check_term(term, context={})


def test_check_term_begin() -> None:
    term = Begin(
        effects=[Immediate(value=0), Immediate(value=1)],
        value=Immediate(value=2),
    )

    match term:
        case Begin():  # pragma: no branch
            check_term(term, context={})