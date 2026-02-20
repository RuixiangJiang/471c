import pytest
from L3.check import Context, check_program, check_term
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
  

@pytest.mark.skip
def test_check_term_let():
    term = Let(
        bindings=[
            ("x", Immediate(value=0)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_let_scope():
    term = Let(
        bindings=[
            ("x", Immediate(value=0)),
            ("y", Reference(name="x")),
        ],
        body=Reference(name="y"),
    )

    context: Context = {}

    with pytest.raises(ValueError):
        check_term(term, context)


@pytest.mark.skip
def test_check_term_let_duplicate_binders():
    term = Let(
        bindings=[
            ("x", Immediate(value=0)),
            ("x", Immediate(value=1)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    with pytest.raises(ValueError):
        check_term(term, context)


@pytest.mark.skip
def test_check_term_letrec():
    term = LetRec(
        bindings=[
            ("x", Immediate(value=0)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_letrec_scope():
    term = LetRec(
        bindings=[
            ("y", Reference(name="x")),
            ("x", Immediate(value=0)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_letrec_duplicate_binders():
    term = LetRec(
        bindings=[
            ("x", Immediate(value=0)),
            ("x", Immediate(value=1)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    with pytest.raises(ValueError):
        check_term(term, context)


@pytest.mark.skip
def test_check_term_reference_bound():
    term = Reference(name="x")

    match term:
        case Reference():
            check_term(term, context={"x": None})


def test_abstract() -> None:
    term = Abstract(
        parameters=["x"],
        body=Reference(name="x"),
    )

    match term:
        case Abstract():
            check_term(term, context={})


def test_apply() -> None:
    term = Apply(
        target=Abstract(parameters=["x"], body=Reference(name="x")),
        arguments=[Immediate(value=1)],
    )

    match term:
        case Apply():
            check_term(term, context={})


def test_immediate() -> None:
    term = Immediate(value=0)

    match term:
        case Immediate():
            check_term(term, context={})


def test_primitive() -> None:
    term = Primitive(
        operator="+",
        left=Immediate(value=1),
        right=Immediate(value=2),
    )

    match term:
        case Primitive():
            check_term(term, context={})


def test_branch() -> None:
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


@pytest.mark.skip
def test_check_term_reference_free():
    term = Reference(name="x")

    match term:
        case Reference():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_let_duplicate_binding_raises() -> None:
    term = Let(
        bindings=[("x", Immediate(value=1)), ("x", Immediate(value=2))],
        body=Reference(name="x"),
    )

    match term:
        case Let():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_letrec_duplicate_binding_raises() -> None:
    term = LetRec(
        bindings=[("f", Immediate(value=0)), ("f", Immediate(value=1))],
        body=Immediate(value=0),
    )

    match term:
        case LetRec():
            with pytest.raises(ValueError):
                check_term(term, context={})


def test_abstract_duplicate_param_raises() -> None:
    term = Abstract(
        parameters=["x", "x"],
        body=Reference(name="x"),
    )

    with pytest.raises(ValueError):
        check_term(term, context＝{})


@pytest.mark.skip
def test_check_term_abstract():
    term = Abstract(
        parameters=["x"],
        body=Immediate(value=0),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_abstract_duplicate_parameters():
    term = Abstract(
        parameters=["x", "x"],
        body=Immediate(value=0),
    )

    context: Context = {}

    with pytest.raises(ValueError):
        check_term(term, context)


@pytest.mark.skip
def test_check_term_apply():
    term = Apply(
        target=Reference(name="x"),
        arguments=[Immediate(value=0)],
    )

    context: Context = {
        "x": None,
    }

    check_term(term, context)


@pytest.mark.skip
def test_check_term_immediate():
    term = Immediate(value=0)

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_primitive():
    term = Primitive(
        operator="+",
        left=Immediate(value=1),
        right=Immediate(value=2),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_branch():
    term = Branch(
        operator="<",
        left=Immediate(value=1),
        right=Immediate(value=2),
        consequent=Immediate(value=0),
        otherwise=Immediate(value=1),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_allocate():
    term = Allocate(count=0)

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_term_load():
    term = Load(
        base=Reference(name="x"),
        index=0,
    )

    context: Context = {
        "x": None,
    }

    check_term(term, context)


@pytest.mark.skip
def test_check_term_store():
    term = Store(
        base=Reference(name="x"),
        index=0,
        value=Immediate(value=0),
    )

    context: Context = {
        "x": None,
    }

    check_term(term, context)


@pytest.mark.skip
def test_check_term_begin():
    term = Begin(
        effects=[Immediate(value=0)],
        value=Immediate(value=0),
    )

    context: Context = {}

    check_term(term, context)


@pytest.mark.skip
def test_check_program():
    program = Program(
        parameters=[],
        body=Immediate(value=0),
    )

    check_program(program)


@pytest.mark.skip
def test_check_program_duplicate_parameters():
    program = Program(
        parameters=["x", "x"],
        body=Immediate(value=0),
    )

    with pytest.raises(ValueError):
        check_program(program)
