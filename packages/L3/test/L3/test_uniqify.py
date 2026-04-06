from L3.syntax import (
    Apply,
    Immediate,
    Let,
    Reference,
    LetRec,
    Allocate,
    Primitive,
    Begin,
    Branch,
    Abstract,
    Store,
    Load,
    Program,
)
from L3.uniqify import Context, uniqify_term, uniqify_program
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
        "m": "4",
        "n": "6",
    }
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh)

    expected = Let(
        bindings=[
            (
                "make_adder0",
                Abstract(
                    parameters=["x0"],
                    body=Abstract(
                        parameters=["y0"],
                        body=Primitive(operator="+", left=Reference(name="x0"), right=Reference(name="y0")),
                    ),
                ),
            )
        ],
        body=Let(
            bindings=[("adder0", Apply(target=Reference(name="make_adder0"), arguments=[Reference(name="4")]))],
            body=Apply(target=Reference(name="adder0"), arguments=[Reference(name="6")]),
        ),
    )

    assert actual == expected


def test_letrec():
    letrec = LetRec(
        bindings=[
            (
                "loop",
                Abstract(
                    parameters=[],
                    body=Branch(
                        operator="<",
                        left=Load(base=Reference(name="x"), index=0),
                        right=Primitive(operator="+", left=Reference(name="n"), right=Immediate(value=1)),
                        consequent=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="acc"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="acc"), index=0),
                                        right=Load(base=Reference(name="x"), index=0),
                                    ),
                                ),
                                Store(
                                    base=Reference(name="acc"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="x"), index=0),
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
        bindings=[("x", Allocate(count=1)), ("acc", Allocate(count=1))],
        body=Begin(
            effects=[
                Store(base=Reference(name="x"), index=0, value=Immediate(value=0)),
                Store(base=Reference(name="acc"), index=0, value=Immediate(value=0)),
            ],
            value=letrec,
        ),
    )

    context: Context = {
        "n": "1",
    }
    fresh = SequentialNameGenerator()
    actual = uniqify_term(term, context, fresh)

    expected_letrec = LetRec(
        bindings=[
            (
                "loop0",
                Abstract(
                    parameters=[],
                    body=Branch(
                        operator="<",
                        left=Load(base=Reference(name="x0"), index=0),
                        right=Primitive(operator="+", left=Reference(name="1"), right=Immediate(value=1)),
                        consequent=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="acc0"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="acc0"), index=0),
                                        right=Load(base=Reference(name="x0"), index=0),
                                    ),
                                ),
                                Store(
                                    base=Reference(name="acc0"),
                                    index=0,
                                    value=Primitive(
                                        operator="+",
                                        left=Load(base=Reference(name="x0"), index=0),
                                        right=Immediate(value=1),
                                    ),
                                ),
                            ],
                            value=Apply(target=Reference(name="loop0"), arguments=[]),
                        ),
                        otherwise=Load(base=Reference(name="acc0"), index=0),
                    ),
                ),
            )
        ],
        body=Apply(target=Reference(name="loop0"), arguments=[]),
    )
    expected = Let(
        bindings=[("x0", Allocate(count=1)), ("acc0", Allocate(count=1))],
        body=Begin(
            effects=[
                Store(base=Reference(name="x0"), index=0, value=Immediate(value=0)),
                Store(base=Reference(name="acc0"), index=0, value=Immediate(value=0)),
            ],
            value=expected_letrec,
        ),
    )
    assert actual == expected


def test_program():
    body = Reference(name="x")
    parameters = ["x"]
    program = Program(parameters=parameters, body=body)
    _, actual_program = uniqify_program(program=program)

    expected_body = Reference(name="x0")
    expected_parameters = ["x0"]
    expected_program = Program(parameters=expected_parameters, body=expected_body)

    assert actual_program == expected_program
