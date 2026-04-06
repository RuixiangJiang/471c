from L2 import syntax as L2
from L2.optimize import optimize_program


def test_optimize_program():
    program = L2.Program(
        parameters=[],
        body=L2.Primitive(
            operator="+",
            left=L2.Immediate(value=1),
            right=L2.Immediate(value=1),
        ),
    )

    expected = L2.Program(
        parameters=[],
        body=L2.Immediate(value=2),
    )

    actual = optimize_program(program)

    assert actual == expected


def test_sum_no_change():
    body = L2.Let(
        bindings=[
            (
                "make_adder",
                L2.Abstract(
                    parameters=["x"],
                    body=L2.Abstract(
                        parameters=["y"],
                        body=L2.Primitive(operator="+", left=L2.Reference(name="x"), right=L2.Reference(name="y")),
                    ),
                ),
            )
        ],
        body=L2.Let(
            bindings=[("adder", L2.Apply(target=L2.Reference(name="make_adder"), arguments=[L2.Reference(name="m")]))],
            body=L2.Apply(target=L2.Reference(name="adder"), arguments=[L2.Reference(name="n")]),
        ),
    )
    program = L2.Program(
        parameters=[],
        body=body,
    )

    expected = L2.Program(
        parameters=[],
        body=body,
    )

    actual = optimize_program(program)

    assert actual == expected

    body_let = L2.Let(
        bindings=[
            (
                "loop",
                L2.Abstract(
                    parameters=[],
                    body=L2.Branch(
                        operator="<",
                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                        right=L2.Primitive(operator="+", left=L2.Reference(name="n"), right=L2.Immediate(value=1)),
                        consequent=L2.Begin(
                            effects=[
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="acc"), index=0),
                                        right=L2.Load(base=L2.Reference(name="i"), index=0),
                                    ),
                                ),
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                                        right=L2.Immediate(value=1),
                                    ),
                                ),
                            ],
                            value=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
                        ),
                        otherwise=L2.Load(base=L2.Reference(name="acc"), index=0),
                    ),
                ),
            )
        ],
        body=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
    )
    body = L2.Let(
        bindings=[("i", L2.Allocate(count=1)), ("acc", L2.Allocate(count=1))],
        body=L2.Begin(
            effects=[
                L2.Store(base=L2.Reference(name="i"), index=0, value=L2.Immediate(value=0)),
                L2.Store(base=L2.Reference(name="acc"), index=0, value=L2.Immediate(value=0)),
            ],
            value=body_let,
        ),
    )

    program = L2.Program(
        parameters=[],
        body=body,
    )

    expected = L2.Program(
        parameters=[],
        body=body,
    )

    actual = optimize_program(program)

    assert actual == expected


def test_fib_no_change():
    body = L2.Let(
        bindings=[
            (
                "fib",
                L2.Abstract(
                    parameters=["n"],
                    body=L2.Branch(
                        operator="<",
                        left=L2.Reference(name="n"),
                        right=L2.Immediate(value=2),
                        consequent=L2.Reference(name="n"),
                        otherwise=L2.Primitive(
                            operator="+",
                            left=L2.Apply(
                                target=L2.Reference(name="fib"),
                                arguments=[
                                    L2.Primitive(operator="-", left=L2.Reference(name="n"), right=L2.Immediate(value=1))
                                ],
                            ),
                            right=L2.Apply(
                                target=L2.Reference(name="fib"),
                                arguments=[
                                    L2.Primitive(operator="-", left=L2.Reference(name="n"), right=L2.Immediate(value=2))
                                ],
                            ),
                        ),
                    ),
                ),
            )
        ],
        body=L2.Apply(target=L2.Reference(name="fib"), arguments=[L2.Reference(name="n")]),
    )
    program = L2.Program(
        parameters=[],
        body=body,
    )

    expected = L2.Program(
        parameters=[],
        body=body,
    )

    actual = optimize_program(program)

    assert actual == expected


def test_optimize():
    body = L2.Let(
        bindings=[
            ("unused1", L2.Immediate(value=2)),
            ("unused2", L2.Reference(name="n")),
            ("c", L2.Immediate(value=8)),
            (
                "fib",
                L2.Abstract(
                    parameters=["n"],
                    body=L2.Branch(
                        operator="<",
                        left=L2.Immediate(value=1),
                        right=L2.Immediate(value=4),
                        consequent=L2.Reference(name="c"),
                        otherwise=L2.Primitive(
                            operator="+",
                            left=L2.Apply(
                                target=L2.Reference(name="fib"),
                                arguments=[
                                    L2.Primitive(operator="-", left=L2.Reference(name="n"), right=L2.Immediate(value=1))
                                ],
                            ),
                            right=L2.Apply(
                                target=L2.Reference(name="fib"),
                                arguments=[
                                    L2.Primitive(operator="-", left=L2.Reference(name="n"), right=L2.Immediate(value=2))
                                ],
                            ),
                        ),
                    ),
                ),
            ),
        ],
        body=L2.Apply(target=L2.Reference(name="fib"), arguments=[L2.Reference(name="n")]),
    )
    program = L2.Program(
        parameters=[],
        body=body,
    )

    expected = L2.Program(
        parameters=[],
        body=L2.Let(
            bindings=[
                (
                    "fib",
                    L2.Abstract(parameters=["n"], body=L2.Immediate(value=8)),
                ),
            ],
            body=L2.Apply(target=L2.Reference(name="fib"), arguments=[L2.Reference(name="n")]),
        ),
    )

    actual = optimize_program(program)

    assert actual == expected

    body_let = L2.Let(
        bindings=[
            ("x", L2.Immediate(value=1)),
            (
                "loop",
                L2.Abstract(
                    parameters=[],
                    body=L2.Branch(
                        operator="<",
                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                        right=L2.Primitive(
                            operator="+",
                            left=L2.Reference(name="n"),
                            right=L2.Store(
                                base=L2.Reference(name="i"),
                                index=0,
                                value=L2.Primitive(
                                    operator="+", left=L2.Immediate(value=0), right=L2.Immediate(value=2)
                                ),
                            ),
                        ),
                        consequent=L2.Begin(
                            effects=[
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="acc"), index=0),
                                        right=L2.Load(base=L2.Reference(name="i"), index=0),
                                    ),
                                ),
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                                        right=L2.Immediate(value=1),
                                    ),
                                ),
                            ],
                            value=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
                        ),
                        otherwise=L2.Load(base=L2.Reference(name="acc"), index=0),
                    ),
                ),
            ),
        ],
        body=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
    )
    body = L2.Let(
        bindings=[("i", L2.Allocate(count=1)), ("acc", L2.Allocate(count=1))],
        body=L2.Begin(
            effects=[
                L2.Store(base=L2.Reference(name="i"), index=0, value=L2.Immediate(value=0)),
                L2.Store(base=L2.Reference(name="acc"), index=0, value=L2.Immediate(value=0)),
                L2.Allocate(count=2),
            ],
            value=body_let,
        ),
    )

    program = L2.Program(
        parameters=[],
        body=body,
    )

    body_let = L2.Let(
        bindings=[
            (
                "loop",
                L2.Abstract(
                    parameters=[],
                    body=L2.Branch(
                        operator="<",
                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                        right=L2.Primitive(
                            operator="+",
                            left=L2.Reference(name="n"),
                            right=L2.Store(base=L2.Reference(name="i"), index=0, value=L2.Immediate(value=2)),
                        ),
                        consequent=L2.Begin(
                            effects=[
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="acc"), index=0),
                                        right=L2.Load(base=L2.Reference(name="i"), index=0),
                                    ),
                                ),
                                L2.Store(
                                    base=L2.Reference(name="acc"),
                                    index=0,
                                    value=L2.Primitive(
                                        operator="+",
                                        left=L2.Load(base=L2.Reference(name="i"), index=0),
                                        right=L2.Immediate(value=1),
                                    ),
                                ),
                            ],
                            value=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
                        ),
                        otherwise=L2.Load(base=L2.Reference(name="acc"), index=0),
                    ),
                ),
            ),
        ],
        body=L2.Apply(target=L2.Reference(name="loop"), arguments=[]),
    )
    body = L2.Let(
        bindings=[("i", L2.Allocate(count=1)), ("acc", L2.Allocate(count=1))],
        body=L2.Begin(
            effects=[
                L2.Store(base=L2.Reference(name="i"), index=0, value=L2.Immediate(value=0)),
                L2.Store(base=L2.Reference(name="acc"), index=0, value=L2.Immediate(value=0)),
                L2.Allocate(count=2),
            ],
            value=body_let,
        ),
    )

    expected = L2.Program(
        parameters=[],
        body=body,
    )

    actual = optimize_program(program)

    assert actual == expected
