import pytest

from L2.optimize import (
    _extend_env,
    _flatten_begin,
    _free_variables,
    _has_effect,
    _is_constant,
    _optimize_term,
    optimize_program,
)
from L2.syntax import (
    Abstract,
    Allocate,
    Apply,
    Begin,
    Branch,
    Immediate,
    Let,
    Load,
    Primitive,
    Program,
    Reference,
    Store,
)


def test_is_constant():
    assert _is_constant(Immediate(value=1)) is True
    assert _is_constant(Reference(name="x")) is True
    assert _is_constant(Allocate(count=1)) is False


def test_has_effect_all_cases():
    assert _has_effect(Immediate(value=1)) is False
    assert _has_effect(Reference(name="x")) is False
    assert _has_effect(Abstract(parameters=["x"], body=Reference(name="x"))) is False
    assert _has_effect(Allocate(count=1)) is False

    assert _has_effect(
        Primitive(
            operator="+",
            left=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
            right=Immediate(value=2),
        )
    ) is True
    assert _has_effect(
        Primitive(
            operator="+",
            left=Immediate(value=1),
            right=Store(base=Allocate(count=1), index=0, value=Immediate(value=2)),
        )
    ) is True
    assert _has_effect(
        Primitive(operator="+", left=Immediate(value=1), right=Immediate(value=2))
    ) is False

    assert _has_effect(
        Branch(
            operator="<",
            left=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
            right=Immediate(value=0),
            consequent=Immediate(value=2),
            otherwise=Immediate(value=3),
        )
    ) is True
    assert _has_effect(
        Branch(
            operator="<",
            left=Immediate(value=0),
            right=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
            consequent=Immediate(value=2),
            otherwise=Immediate(value=3),
        )
    ) is True
    assert _has_effect(
        Branch(
            operator="<",
            left=Immediate(value=0),
            right=Immediate(value=1),
            consequent=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
            otherwise=Immediate(value=3),
        )
    ) is True
    assert _has_effect(
        Branch(
            operator="<",
            left=Immediate(value=0),
            right=Immediate(value=1),
            consequent=Immediate(value=2),
            otherwise=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
        )
    ) is True
    assert _has_effect(
        Branch(
            operator="<",
            left=Immediate(value=0),
            right=Immediate(value=1),
            consequent=Immediate(value=2),
            otherwise=Immediate(value=3),
        )
    ) is False

    assert _has_effect(Load(base=Immediate(value=1), index=0)) is False
    assert _has_effect(
        Load(
            base=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
            index=0,
        )
    ) is True

    assert _has_effect(
        Let(
            bindings=[("x", Store(base=Allocate(count=1), index=0, value=Immediate(value=1)))],
            body=Immediate(value=0),
        )
    ) is True
    assert _has_effect(
        Let(
            bindings=[("x", Immediate(value=1))],
            body=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
        )
    ) is True
    assert _has_effect(
        Let(bindings=[("x", Immediate(value=1))], body=Reference(name="x"))
    ) is False

    assert _has_effect(Begin(effects=[], value=Immediate(value=0))) is False
    assert _has_effect(Begin(effects=[Immediate(value=1)], value=Immediate(value=2))) is True
    assert _has_effect(
        Begin(
            effects=[Store(base=Allocate(count=1), index=0, value=Immediate(value=1))],
            value=Immediate(value=2),
        )
    ) is True
    assert _has_effect(
        Begin(
            effects=[],
            value=Store(base=Allocate(count=1), index=0, value=Immediate(value=1)),
        )
    ) is True

    assert _has_effect(Apply(target=Reference(name="f"), arguments=[])) is True
    assert _has_effect(
        Store(base=Allocate(count=1), index=0, value=Immediate(value=1))
    ) is True


def test_free_variables_all_cases():
    assert _free_variables(Immediate(value=1)) == set()
    assert _free_variables(Allocate(count=1)) == set()
    assert _free_variables(Reference(name="x")) == {"x"}

    assert _free_variables(
        Abstract(
            parameters=["x", "y"],
            body=Primitive(
                operator="+",
                left=Reference(name="x"),
                right=Reference(name="z"),
            ),
        )
    ) == {"z"}

    assert _free_variables(
        Apply(
            target=Reference(name="f"),
            arguments=[
                Reference(name="x"),
                Primitive(
                    operator="+",
                    left=Reference(name="y"),
                    right=Immediate(value=1),
                ),
            ],
        )
    ) == {"f", "x", "y"}

    assert _free_variables(
        Primitive(operator="+", left=Reference(name="x"), right=Reference(name="y"))
    ) == {"x", "y"}

    assert _free_variables(
        Branch(
            operator="<",
            left=Reference(name="a"),
            right=Reference(name="b"),
            consequent=Reference(name="c"),
            otherwise=Reference(name="d"),
        )
    ) == {"a", "b", "c", "d"}

    assert _free_variables(Load(base=Reference(name="arr"), index=0)) == {"arr"}
    assert _free_variables(
        Store(base=Reference(name="arr"), index=0, value=Reference(name="val"))
    ) == {"arr", "val"}

    assert _free_variables(
        Begin(
            effects=[
                Reference(name="u"),
                Store(base=Reference(name="a"), index=0, value=Reference(name="b")),
            ],
            value=Reference(name="v"),
        )
    ) == {"u", "a", "b", "v"}

    assert _free_variables(
        Let(
            bindings=[
                ("x", Reference(name="a")),
                ("y", Primitive(operator="+", left=Reference(name="x"), right=Reference(name="b"))),
            ],
            body=Primitive(operator="+", left=Reference(name="y"), right=Reference(name="c")),
        )
    ) == {"a", "b", "c"}


def test_extend_env_and_flatten_begin():
    env = {"x": Immediate(value=1), "y": Immediate(value=2)}
    assert _extend_env(env, "z", Immediate(value=3)) == {
        "x": Immediate(value=1),
        "y": Immediate(value=2),
        "z": Immediate(value=3),
    }
    assert _extend_env(env, "z", Reference(name="x")) == {
        "x": Immediate(value=1),
        "y": Immediate(value=2),
        "z": Reference(name="x"),
    }
    assert _extend_env(env, "x", Allocate(count=1)) == {"y": Immediate(value=2)}

    assert _flatten_begin([], Immediate(value=5)) == Immediate(value=5)
    assert _flatten_begin(
        [
            Immediate(value=1),
            Begin(
                effects=[
                    Immediate(value=2),
                    Store(base=Allocate(count=1), index=0, value=Immediate(value=3)),
                ],
                value=Immediate(value=4),
            ),
        ],
        Begin(
            effects=[Store(base=Allocate(count=1), index=0, value=Immediate(value=5))],
            value=Immediate(value=6),
        ),
    ) == Begin(
        effects=[
            Store(base=Allocate(count=1), index=0, value=Immediate(value=3)),
            Store(base=Allocate(count=1), index=0, value=Immediate(value=5)),
        ],
        value=Immediate(value=6),
    )


def test_optimize_term_main_paths():
    assert _optimize_term(Reference(name="x"), {"x": Immediate(value=7)}) == Immediate(value=7)
    assert _optimize_term(Reference(name="y"), {"x": Immediate(value=7)}) == Reference(name="y")
    assert _optimize_term(Immediate(value=1), {}) == Immediate(value=1)
    assert _optimize_term(Allocate(count=2), {}) == Allocate(count=2)

    assert _optimize_term(
        Abstract(
            parameters=["x"],
            body=Primitive(operator="+", left=Reference(name="x"), right=Reference(name="y")),
        ),
        {"x": Immediate(value=100), "y": Immediate(value=2)},
    ) == Abstract(
        parameters=["x"],
        body=Primitive(operator="+", left=Reference(name="x"), right=Immediate(value=2)),
    )

    assert _optimize_term(
        Apply(
            target=Reference(name="f"),
            arguments=[
                Primitive(operator="+", left=Immediate(value=2), right=Immediate(value=3)),
                Reference(name="x"),
            ],
        ),
        {"f": Reference(name="g"), "x": Immediate(value=9)},
    ) == Apply(
        target=Reference(name="g"),
        arguments=[Immediate(value=5), Immediate(value=9)],
    )

    assert _optimize_term(
        Primitive(operator="+", left=Immediate(value=1), right=Immediate(value=2)),
        {},
    ) == Immediate(value=3)
    assert _optimize_term(
        Primitive(operator="-", left=Immediate(value=5), right=Immediate(value=2)),
        {},
    ) == Immediate(value=3)
    assert _optimize_term(
        Primitive(operator="*", left=Immediate(value=4), right=Immediate(value=3)),
        {},
    ) == Immediate(value=12)
    non_fold_prim = Primitive(operator="+", left=Reference(name="x"), right=Immediate(value=2))
    assert _optimize_term(non_fold_prim, {}) == non_fold_prim

    assert _optimize_term(
        Branch(
            operator="<",
            left=Immediate(value=1),
            right=Immediate(value=2),
            consequent=Immediate(value=10),
            otherwise=Immediate(value=20),
        ),
        {},
    ) == Immediate(value=10)
    assert _optimize_term(
        Branch(
            operator="<",
            left=Immediate(value=2),
            right=Immediate(value=1),
            consequent=Immediate(value=10),
            otherwise=Immediate(value=20),
        ),
        {},
    ) == Immediate(value=20)
    assert _optimize_term(
        Branch(
            operator="==",
            left=Immediate(value=2),
            right=Immediate(value=2),
            consequent=Immediate(value=10),
            otherwise=Immediate(value=20),
        ),
        {},
    ) == Immediate(value=10)
    assert _optimize_term(
        Branch(
            operator="==",
            left=Immediate(value=2),
            right=Immediate(value=3),
            consequent=Immediate(value=10),
            otherwise=Immediate(value=20),
        ),
        {},
    ) == Immediate(value=20)
    assert _optimize_term(
        Branch(
            operator="<",
            left=Reference(name="x"),
            right=Immediate(value=0),
            consequent=Primitive(operator="*", left=Immediate(value=2), right=Immediate(value=3)),
            otherwise=Primitive(operator="-", left=Immediate(value=8), right=Immediate(value=1)),
        ),
        {},
    ) == Branch(
        operator="<",
        left=Reference(name="x"),
        right=Immediate(value=0),
        consequent=Immediate(value=6),
        otherwise=Immediate(value=7),
    )

    assert _optimize_term(
        Load(base=Let(bindings=[("a", Immediate(value=1))], body=Reference(name="a")), index=0),
        {},
    ) == Load(base=Immediate(value=1), index=0)

    assert _optimize_term(
        Store(
            base=Let(bindings=[("a", Immediate(value=1))], body=Reference(name="a")),
            index=0,
            value=Primitive(operator="+", left=Immediate(value=3), right=Immediate(value=4)),
        ),
        {},
    ) == Store(base=Immediate(value=1), index=0, value=Immediate(value=7))

    assert _optimize_term(
        Begin(
            effects=[
                Immediate(value=1),
                Begin(
                    effects=[
                        Immediate(value=2),
                        Store(base=Allocate(count=1), index=0, value=Immediate(value=9)),
                    ],
                    value=Immediate(value=3),
                ),
            ],
            value=Begin(
                effects=[
                    Immediate(value=4),
                    Store(base=Allocate(count=1), index=0, value=Immediate(value=8)),
                ],
                value=Immediate(value=5),
            ),
        ),
        {},
    ) == Begin(
        effects=[
            Store(base=Allocate(count=1), index=0, value=Immediate(value=9)),
            Store(base=Allocate(count=1), index=0, value=Immediate(value=8)),
        ],
        value=Immediate(value=5),
    )

    assert _optimize_term(
        Let(
            bindings=[
                ("x", Immediate(value=5)),
                ("y", Reference(name="x")),
                ("z", Primitive(operator="+", left=Reference(name="y"), right=Immediate(value=1))),
            ],
            body=Reference(name="z"),
        ),
        {},
    ) == Immediate(value=6)

    assert _optimize_term(
        Let(
            bindings=[
                ("x", Primitive(operator="+", left=Reference(name="a"), right=Immediate(value=1))),
                ("y", Reference(name="x")),
            ],
            body=Reference(name="y"),
        ),
        {},
    ) == Let(
        bindings=[
            ("x", Primitive(operator="+", left=Reference(name="a"), right=Immediate(value=1))),
        ],
        body=Reference(name="x"),
    )

    assert _optimize_term(
        Let(
            bindings=[("x", Store(base=Allocate(count=1), index=0, value=Immediate(value=42)))],
            body=Immediate(value=7),
        ),
        {},
    ) == Begin(
        effects=[Store(base=Allocate(count=1), index=0, value=Immediate(value=42))],
        value=Immediate(value=7),
    )

    assert _optimize_term(
        Let(
            bindings=[
                ("x", Immediate(value=1)),
                ("y", Primitive(operator="+", left=Immediate(value=2), right=Immediate(value=3))),
            ],
            body=Immediate(value=9),
        ),
        {},
    ) == Immediate(value=9)

    assert _optimize_term(
        Let(
            bindings=[("x", Store(base=Reference(name="arr"), index=0, value=Reference(name="val")))],
            body=Immediate(value=0),
        ),
        {},
    ) == Begin(
        effects=[Store(base=Reference(name="arr"), index=0, value=Reference(name="val"))],
        value=Immediate(value=0),
    )

    assert _optimize_term(
        Let(
            bindings=[
                ("x", Let(bindings=[("a", Reference(name="z"))], body=Reference(name="a"))),
            ],
            body=Reference(name="x"),
        ),
        {},
    ) == Reference(name="z")

    assert _optimize_term(
        Let(
            bindings=[
                ("x", Begin(effects=[Reference(name="u")], value=Reference(name="v"))),
            ],
            body=Reference(name="x"),
        ),
        {},
    ) == Reference(name="v")

    store_term = Let(
        bindings=[
            ("x", Store(base=Reference(name="arr"), index=0, value=Reference(name="val"))),
        ],
        body=Reference(name="x"),
    )
    assert _optimize_term(store_term, {}) == store_term


def test_optimize_program_and_error_paths():
    assert optimize_program(
        Program(
            parameters=[],
            body=Let(
                bindings=[
                    ("x", Immediate(value=1)),
                    ("y", Reference(name="x")),
                    ("z", Primitive(operator="+", left=Reference(name="y"), right=Immediate(value=2))),
                ],
                body=Reference(name="z"),
            ),
        )
    ) == Program(parameters=[], body=Immediate(value=3))

    immediate_program = Program(parameters=[], body=Immediate(value=42))
    reference_program = Program(parameters=["x"], body=Reference(name="x"))
    allocate_program = Program(parameters=[], body=Allocate(count=4))
    assert optimize_program(immediate_program) == immediate_program
    assert optimize_program(reference_program) == reference_program
    assert optimize_program(allocate_program) == allocate_program

    with pytest.raises(TypeError, match="unsupported term"):
        _free_variables(object())

    with pytest.raises(TypeError, match="unsupported term"):
        _optimize_term(object(), {})

def test_has_effect_invalid_term_raises_type_error():
    with pytest.raises(TypeError, match="unsupported term"):
        _has_effect(object())

def test_optimize_term_primitive_invalid_operator_raises_value_error():
    term = Primitive.model_construct(
        operator="/",
        left=Immediate(value=8),
        right=Immediate(value=2),
    )

    with pytest.raises(ValueError, match="unsupported primitive operator"):
        _optimize_term(term, {})


def test_optimize_term_branch_invalid_operator_raises_value_error():
    term = Branch.model_construct(
        operator=">",
        left=Immediate(value=8),
        right=Immediate(value=2),
        consequent=Immediate(value=1),
        otherwise=Immediate(value=0),
    )

    with pytest.raises(ValueError, match="unsupported branch operator"):
        _optimize_term(term, {})