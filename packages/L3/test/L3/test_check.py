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


def test_check_term_let():
    term = Let(
        bindings=[
            ("x", Immediate(value=0)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    check_term(term, context)


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


def test_check_term_letrec():
    term = LetRec(
        bindings=[
            ("x", Immediate(value=0)),
        ],
        body=Reference(name="x"),
    )

    context: Context = {}

    check_term(term, context)


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


def test_let_sum():
    term = Let(
        bindings=[
            (
                "make_adder",
                Abstract(
                    parameters=["x"],
                    body=Abstract(
                        parameters=["y"],
                        body=Primitive(operator="+", left=Reference(name="x"), right=Reference(name="y")),
                    ),
                ),
            )
        ],
        body=Let(
            bindings=[("adder", Apply(target=Reference(name="make_adder"), arguments=[Reference(name="m")]))],
            body=Apply(target=Reference(name="adder"), arguments=[Reference(name="n")]),
        ),
    )
    context: Context = {
        "m": None,
        "n": None,
    }

    check_term(term, context)


def test_letrec_fibonacci():
    term = LetRec(
        bindings=[
            (
                "fib",
                Abstract(
                    parameters=["n"],
                    body=Branch(
                        operator="<",
                        left=Reference(name="n"),
                        right=Immediate(value=2),
                        consequent=Reference(name="n"),
                        otherwise=Primitive(
                            operator="+",
                            left=Apply(
                                target=Reference(name="fib"),
                                arguments=[Primitive(operator="-", left=Reference(name="n"), right=Immediate(value=1))],
                            ),
                            right=Apply(
                                target=Reference(name="fib"),
                                arguments=[Primitive(operator="-", left=Reference(name="n"), right=Immediate(value=2))],
                            ),
                        ),
                    ),
                ),
            )
        ],
        body=Apply(target=Reference(name="fib"), arguments=[Reference(name="n")]),
    )

    context: Context = {
        "n": None,
    }

    check_term(term, context)


def test_letrec_sum():
    letrec = LetRec(
        bindings=[
            (
                "loop",
                Abstract(
                    parameters=[],
                    body=Branch(
                        operator="<",
                        left=Load(base=Reference(name="i"), index=0),
                        right=Primitive(operator="+", left=Reference(name="n"), right=Immediate(value=1)),
                        consequent=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="acc"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="acc"), index=0),
                                        right=Load(base=Reference(name="i"), index=0),
                                    ),
                                ),
                                Store(
                                    base=Reference(name="acc"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="i"), index=0),
                                        right=Immediate(value=1),
                                    ),
                                ),
                            ],
                            value=Apply(target=Reference(name="loop"), arguments=[]),
                        ),
                        otherwise=Load(base=Reference(name="acc"), index=0),
                    ),
                ),
            )
        ],
        body=Apply(target=Reference(name="loop"), arguments=[]),
    )
    term = Let(
        bindings=[("i", Allocate(count=1)), ("acc", Allocate(count=1))],
        body=Begin(
            effects=[
                Store(base=Reference(name="i"), index=0, value=Immediate(value=0)),
                Store(base=Reference(name="acc"), index=0, value=Immediate(value=0)),
            ],
            value=letrec,
        ),
    )

    context: Context = {
        "n": None,
    }

    check_term(term, context)


def test_check_reference():
    term = Reference(name="x")
    context: Context = {
        "x": None,
    }
    check_term(term, context)


def test_check_reference_fail():
    term = Reference(name="x")
    context: Context = {}
    with pytest.raises(ValueError):
        check_term(term, context)


def test_check_abstract():
    term = Abstract(parameters=["x"], body=Reference(name="x"))
    context: Context = {}
    check_term(term, context)

    term_no_param = Abstract(parameters=[], body=Reference(name="x"))
    context: Context = {"x": None}
    check_term(term_no_param, context)


def test_check_abstract_fail():
    term = Abstract(parameters=["x"], body=Reference(name="y"))

    context: Context = {}
    with pytest.raises(ValueError):
        check_term(term, context)

    term_duplicate = Abstract(parameters=["x", "x"], body=Immediate(value=0))

    context: Context = {}
    with pytest.raises(ValueError):
        check_term(term_duplicate, context)


def test_check_apply():
    term = Apply(target=Reference(name="x"), arguments=[Reference(name="y"), Reference(name="x")])
    context: Context = {"x": None, "y": None}
    check_term(term, context)


def test_check_apply_fail():
    term_both_fail = Apply(target=Reference(name="x"), arguments=[Reference(name="y"), Reference(name="x")])
    context: Context = {}
    with pytest.raises(ValueError):
        check_term(term_both_fail, context)

    term_argument_fail = Apply(target=Reference(name="x"), arguments=[Reference(name="y"), Reference(name="x")])
    context: Context = {"x": None}
    with pytest.raises(ValueError):
        check_term(term_argument_fail, context)

    term_target_fail = Apply(target=Reference(name="z"), arguments=[Reference(name="x")])
    context: Context = {"x": None}
    with pytest.raises(ValueError):
        check_term(term_target_fail, context)


def test_check_immediate():
    term = Immediate(value=0)
    context: Context = {}
    check_term(term, context)


def test_check_primitive():
    term = Primitive(operator="+", left=Reference(name="x"), right=Reference(name="y"))
    context: Context = {"x": None, "y": None}
    check_term(term, context)

    term = Primitive(operator="*", left=Reference(name="x"), right=Reference(name="y"))
    check_term(term, context)

    term = Primitive(operator="-", left=Reference(name="x"), right=Reference(name="y"))
    check_term(term, context)


def test_check_primitive_fail():
    term_left_fail = Primitive(operator="+", left=Reference(name="a"), right=Reference(name="y"))
    context: Context = {"x": None, "y": None}
    with pytest.raises(ValueError):
        check_term(term_left_fail, context)

    term_right_fail = Primitive(operator="+", left=Reference(name="x"), right=Reference(name="a"))
    with pytest.raises(ValueError):
        check_term(term_right_fail, context)


def test_check_branch():
    term = Branch(
        operator="<",
        left=Reference(name="x"),
        right=Reference(name="y"),
        consequent=Reference(name="z"),
        otherwise=Reference(name="w"),
    )
    context: Context = {"x": None, "y": None, "z": None, "w": None}
    check_term(term, context)

    term = Branch(
        operator="==",
        left=Reference(name="x"),
        right=Reference(name="y"),
        consequent=Reference(name="z"),
        otherwise=Reference(name="w"),
    )
    check_term(term, context)


def test_check_branch_fail():
    term_left_fail = Branch(
        operator="<",
        left=Reference(name="a"),
        right=Reference(name="y"),
        consequent=Reference(name="z"),
        otherwise=Reference(name="w"),
    )
    context: Context = {"x": None, "y": None, "z": None, "w": None}
    with pytest.raises(ValueError):
        check_term(term_left_fail, context)

    term_right_fail = Branch(
        operator="<",
        left=Reference(name="x"),
        right=Reference(name="a"),
        consequent=Reference(name="z"),
        otherwise=Reference(name="w"),
    )
    with pytest.raises(ValueError):
        check_term(term_right_fail, context)

    term_consequent_fail = Branch(
        operator="<",
        left=Reference(name="x"),
        right=Reference(name="y"),
        consequent=Reference(name="a"),
        otherwise=Reference(name="w"),
    )
    with pytest.raises(ValueError):
        check_term(term_consequent_fail, context)

    term_otherwise_fail = Branch(
        operator="<",
        left=Reference(name="x"),
        right=Reference(name="y"),
        consequent=Reference(name="z"),
        otherwise=Reference(name="a"),
    )
    with pytest.raises(ValueError):
        check_term(term_otherwise_fail, context)


def test_check_allocate():
    term = Allocate(count=0)
    context: Context = {}
    check_term(term, context)

    term = LetRec(bindings=[("x", Allocate(count=0))], body=Reference(name="x"))
    context: Context = {}
    check_term(term, context)


def test_check_load():
    term = Load(base=Reference(name="x"), index=0)
    context: Context = {"x": None}
    check_term(term, context)


def test_check_load_fail():
    term = Load(base=Reference(name="a"), index=0)
    context: Context = {"x": None}
    with pytest.raises(ValueError):
        check_term(term, context)


def test_check_store():
    term = Store(base=Reference(name="x"), index=0, value=Reference(name="y"))
    context: Context = {"x": None, "y": None}
    check_term(term, context)


def test_check_store_fail():
    term_base_fail = Store(base=Reference(name="a"), index=0, value=Reference(name="y"))
    context: Context = {"x": None, "y": None}
    with pytest.raises(ValueError):
        check_term(term_base_fail, context)

    term_value_fail = Store(base=Reference(name="x"), index=0, value=Reference(name="a"))
    with pytest.raises(ValueError):
        check_term(term_value_fail, context)


def test_check_begin():
    term = Begin(effects=[Reference(name="x"), Reference(name="y")], value=Reference(name="z"))
    context: Context = {"x": None, "y": None, "z": None}
    check_term(term, context)


def test_check_begin_fail():
    term_effects_fail = Begin(effects=[Reference(name="x"), Reference(name="a")], value=Reference(name="z"))
    context: Context = {"x": None, "y": None, "z": None}
    with pytest.raises(ValueError):
        check_term(term_effects_fail, context)

    term_value_fail = Begin(effects=[Reference(name="x"), Reference(name="y")], value=Reference(name="a"))
    context: Context = {"x": None, "y": None, "z": None}
    with pytest.raises(ValueError):
        check_term(term_value_fail, context)


def test_check_program():
    program = Program(parameters=["x"], body=Reference(name="x"))
    check_program(program)


def test_check_program_fail():
    program = Program(parameters=["x", "x"], body=Immediate(value=0))
    with pytest.raises(ValueError):
        check_program(program)
