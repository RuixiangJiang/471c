import pytest
from L4 import syntax as L4

from L5 import syntax as L5
from L5.minor_convert import (
    LoopContext,
    SequentialNameGenerator,
    guarded_void,
    lower_bunch,
    lower_for,
    lower_foreach,
    lower_short_circuit,
    lower_switch,
    lower_while,
    make_bool,
)


def identity_lower(expression, **kwargs):
    return expression


def identity_lower_type(typeof, class_env):
    return typeof


def test_sequential_name_generator():
    fresh = SequentialNameGenerator()
    assert fresh("x") == "x0"
    assert fresh("x") == "x1"
    assert fresh("y") == "y0"


def test_make_bool():
    assert make_bool(True) == L4.Immediate(value=True)
    assert make_bool(False) == L4.Immediate(value=False)


def test_guarded_void_without_loop():
    expr = L4.Immediate(value=1)
    assert guarded_void(expr, loop=None) == expr


def test_guarded_void_with_loop():
    loop = LoopContext(break_flag="b", continue_flag="c")
    expr = L4.Empty()
    actual = guarded_void(expr, loop=loop)
    assert isinstance(actual, L4.If)


def test_lower_short_circuit_and_or():
    actual = lower_short_circuit(
        L5.ShortCircuit(
            operator="&&",
            left=L4.Immediate(value=True),
            right=L4.Immediate(value=False),
        ),
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        loop=None,
        lower_expr=identity_lower,
    )
    assert actual == L4.If(
        condition=L4.Immediate(value=True),
        consequent=L4.Immediate(value=False),
        otherwise=L4.Immediate(value=False),
    )

    actual = lower_short_circuit(
        L5.ShortCircuit(
            operator="||",
            left=L4.Immediate(value=False),
            right=L4.Immediate(value=True),
        ),
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        loop=None,
        lower_expr=identity_lower,
    )
    assert actual == L4.If(
        condition=L4.Immediate(value=False),
        consequent=L4.Immediate(value=True),
        otherwise=L4.Immediate(value=True),
    )


def test_lower_short_circuit_invalid_operator():
    bogus = L5.ShortCircuit.model_construct(  # type: ignore[arg-type]
        tag="shortcircuit",
        operator="??",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )
    with pytest.raises(ValueError, match="unknown short-circuit operator"):
        lower_short_circuit(
            bogus,
            fresh=SequentialNameGenerator(),
            context={},
            class_env={},
            current_class=None,
            loop=None,
            lower_expr=identity_lower,
        )


def test_lower_switch():
    actual = lower_switch(
        L5.Switch(
            scrutinee=L4.Reference(name="x"),
            cases=[
                L5.SwitchCase(value=1, body=L4.Immediate(value=10)),
                L5.SwitchCase(value=2, body=L4.Immediate(value=20)),
            ],
            default=L4.Immediate(value=0),
        ),
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        loop=None,
        lower_expr=identity_lower,
    )
    assert isinstance(actual, L4.Let)


def test_lower_bunch_empty_and_guarded():
    actual = lower_bunch(
        [],
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        loop=None,
        lower_expr=identity_lower,
    )
    assert actual == L4.Empty()

    loop = LoopContext(break_flag="b", continue_flag="c")
    actual = lower_bunch(
        [L4.Immediate(value=1), L4.Immediate(value=2)],
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        loop=loop,
        lower_expr=identity_lower,
    )
    assert isinstance(actual, L4.Bunch)
    assert isinstance(actual.expressions[1], L4.If)


def test_lower_while():
    actual = lower_while(
        L4.Immediate(value=True),
        L4.Empty(),
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        lower_expr=identity_lower,
    )
    assert isinstance(actual, L4.LetRec)


def test_lower_for_int_and_expr_times():
    actual = lower_for(
        3,
        L4.Empty(),
        fresh=SequentialNameGenerator(),
        context={},
        class_env={},
        current_class=None,
        lower_expr=identity_lower,
    )
    assert isinstance(actual, L4.LetRec)

    actual = lower_for(
        L4.Reference(name="n"),
        L4.Empty(),
        fresh=SequentialNameGenerator(),
        context={"n": L4.Int()},
        class_env={},
        current_class=None,
        lower_expr=identity_lower,
    )
    assert isinstance(actual, L4.LetRec)


def test_lower_foreach_zero_and_nonzero():
    actual = lower_foreach(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=0,
            run=L4.Reference(name="x"),
        ),
        fresh=SequentialNameGenerator(),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env={},
        current_class=None,
        lower_expr=identity_lower,
        lower_type=identity_lower_type,
    )
    assert isinstance(actual, L4.Let)
    assert actual.body == L4.Empty()

    actual = lower_foreach(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=2,
            run=L4.Reference(name="x"),
        ),
        fresh=SequentialNameGenerator(),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env={},
        current_class=None,
        lower_expr=identity_lower,
        lower_type=identity_lower_type,
    )
    assert isinstance(actual, L4.Let)
    assert isinstance(actual.body, L4.Bunch)
    assert len(actual.body.expressions) == 2