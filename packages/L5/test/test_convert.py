import pytest
from L3.syntax import Program as L3Program
from L4 import syntax as L4

from L5 import convert_to_l3, convert_to_l4, dummy_parse
from L5 import syntax as L5
from L5.convert import (
    LoopContext,
    SequentialNameGenerator,
    _guarded_void,
    _lower_expression,
)


def test_init_exports_are_available():
    assert callable(convert_to_l3)
    assert callable(convert_to_l4)
    assert callable(dummy_parse)
    assert L5.Program is not None


def test_dummy_parse_returns_minimal_program():
    actual = dummy_parse("hello")
    assert isinstance(actual, L5.Program)
    assert actual.definitions == [("hello", L4.Void(), L4.Empty())]
    assert actual.body == L4.Empty()


def test_sequential_name_generator_counts_per_prefix():
    fresh = SequentialNameGenerator()
    assert fresh("x") == "x0"
    assert fresh("x") == "x1"
    assert fresh("y") == "y0"
    assert fresh("x") == "x2"


def test_guarded_void_without_loop_returns_same_expression():
    expr = L4.Immediate(value=1)
    assert _guarded_void(expr, loop=None) == expr


def test_guarded_void_with_loop_wraps_in_if():
    expr = L4.Empty()
    loop = LoopContext(break_flag="b", continue_flag="c")

    actual = _guarded_void(expr, loop=loop)

    assert isinstance(actual, L4.If)
    assert actual.consequent == L4.Empty()
    assert actual.otherwise == expr
    assert isinstance(actual.condition, L4.If)
    assert actual.condition.condition == L4.Get(target=L4.Reference(name="b"), index=0)
    assert actual.condition.consequent == L4.Immediate(value=True)
    assert actual.condition.otherwise == L4.Get(target=L4.Reference(name="c"), index=0)


def test_convert_short_circuit_and():
    program = L5.Program(
        definitions=[],
        body=L5.ShortCircuit(
            operator="&&",
            left=L4.Reference(name="a"),
            right=L4.Reference(name="b"),
        ),
    )

    actual = convert_to_l4(program)

    assert actual == L4.Program(
        definitions=[],
        body=L4.If(
            condition=L4.Reference(name="a"),
            consequent=L4.Reference(name="b"),
            otherwise=L4.Immediate(value=False),
        ),
    )


def test_convert_short_circuit_or():
    program = L5.Program(
        definitions=[],
        body=L5.ShortCircuit(
            operator="||",
            left=L4.Reference(name="a"),
            right=L4.Reference(name="b"),
        ),
    )

    actual = convert_to_l4(program)

    assert actual == L4.Program(
        definitions=[],
        body=L4.If(
            condition=L4.Reference(name="a"),
            consequent=L4.Immediate(value=True),
            otherwise=L4.Reference(name="b"),
        ),
    )


def test_lower_short_circuit_invalid_operator_branch():
    fresh = SequentialNameGenerator()
    bogus = L5.ShortCircuit.model_construct(
        tag="shortcircuit",
        operator="??",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )
    with pytest.raises(ValueError, match="unknown short-circuit operator"):
        _lower_expression(bogus, fresh=fresh, context={}, loop=None)


def test_convert_switch_on_int():
    program = L5.Program(
        definitions=[],
        body=L5.Switch(
            scrutinee=L4.Reference(name="x"),
            cases=[
                L5.SwitchCase(value=1, body=L4.Immediate(value=10)),
                L5.SwitchCase(value=2, body=L4.Immediate(value=20)),
            ],
            default=L4.Immediate(value=99),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.Let)
    [(name, ty, ex)] = actual.body.bindings
    assert name == "switch_scrutinee0"
    assert ty == L4.Int()
    assert ex == L4.Reference(name="x")

    nested = actual.body.body
    assert isinstance(nested, L4.If)
    assert nested.condition == L4.Operation(
        operator="==",
        left=L4.Reference(name="switch_scrutinee0"),
        right=L4.Immediate(value=1),
    )
    assert nested.consequent == L4.Immediate(value=10)

    assert isinstance(nested.otherwise, L4.If)
    assert nested.otherwise.condition == L4.Operation(
        operator="==",
        left=L4.Reference(name="switch_scrutinee0"),
        right=L4.Immediate(value=2),
    )
    assert nested.otherwise.consequent == L4.Immediate(value=20)
    assert nested.otherwise.otherwise == L4.Immediate(value=99)


def test_convert_switch_on_bool_uses_bool_binding_type():
    program = L5.Program(
        definitions=[],
        body=L5.Switch(
            scrutinee=L4.Reference(name="flag"),
            cases=[
                L5.SwitchCase(value=True, body=L4.Immediate(value=1)),
            ],
            default=L4.Immediate(value=0),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.Let)
    [(name, ty, ex)] = actual.body.bindings
    assert name == "switch_scrutinee0"
    assert ty == L4.Bool()
    assert ex == L4.Reference(name="flag")


def test_break_outside_loop_raises():
    program = L5.Program(definitions=[], body=L5.Break())
    with pytest.raises(ValueError, match="break used outside of a loop"):
        convert_to_l4(program)


def test_continue_outside_loop_raises():
    program = L5.Program(definitions=[], body=L5.Continue())
    with pytest.raises(ValueError, match="continue used outside of a loop"):
        convert_to_l4(program)


def test_break_and_continue_inside_loop_context_lower():
    fresh = SequentialNameGenerator()
    loop = LoopContext(break_flag="b", continue_flag="c")

    break_ex = _lower_expression(L5.Break(), fresh=fresh, context={}, loop=loop)
    continue_ex = _lower_expression(L5.Continue(), fresh=fresh, context={}, loop=loop)

    assert break_ex == L4.Set(
        target=L4.Reference(name="b"),
        index=0,
        value=L4.Immediate(value=True),
    )
    assert continue_ex == L4.Set(
        target=L4.Reference(name="c"),
        index=0,
        value=L4.Immediate(value=True),
    )


def test_lower_l4_while_is_lowered():
    fresh = SequentialNameGenerator()

    expr = _lower_expression(
        L4.While(
            condition=L4.Immediate(value=True),
            run=L4.Empty(),
        ),
        fresh=fresh,
        context={},
        loop=None,
    )

    assert isinstance(expr, L4.LetRec)
    assert len(expr.bindings) == 3

    names = [name for name, _, _ in expr.bindings]
    assert names == ["break0", "continue0", "while0"]

    break_binding = expr.bindings[0]
    continue_binding = expr.bindings[1]
    loop_binding = expr.bindings[2]

    assert break_binding[1] == L4.Mutable(oftype=L4.Bool())
    assert continue_binding[1] == L4.Mutable(oftype=L4.Bool())
    assert break_binding[2] == L4.HeapAllocate(val=L4.Immediate(value=False))
    assert continue_binding[2] == L4.HeapAllocate(val=L4.Immediate(value=False))

    assert loop_binding[1] == L4.FuncType(parameters=[], result=L4.Void())
    assert isinstance(loop_binding[2], L4.Function)
    assert isinstance(loop_binding[2].body, L4.If)

    assert expr.body == L4.Call(target=L4.Reference(name="while0"), arguments=[])


def test_for_with_int_times_is_lowered():
    program = L5.Program(
        definitions=[],
        body=L4.For(
            times=3,
            run=L4.Empty(),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.LetRec)
    assert len(actual.body.bindings) == 4

    counter_binding = actual.body.bindings[0]
    break_binding = actual.body.bindings[1]
    continue_binding = actual.body.bindings[2]
    loop_binding = actual.body.bindings[3]

    assert counter_binding[0] == "for_counter0"
    assert counter_binding[1] == L4.Mutable(oftype=L4.Int())
    assert counter_binding[2] == L4.Immediate(value=3)

    assert break_binding[0] == "break0"
    assert continue_binding[0] == "continue0"
    assert loop_binding[0] == "for0"

    assert actual.body.body == L4.Call(target=L4.Reference(name="for0"), arguments=[])


def test_for_with_expression_times_is_lowered():
    program = L5.Program(
        definitions=[],
        body=L4.For(
            times=L4.Reference(name="n"),
            run=L4.Empty(),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.LetRec)
    counter_binding = actual.body.bindings[0]
    assert counter_binding[2] == L4.Reference(name="n")


def test_foreach_static_unroll_shape():
    program = L5.Program(
        definitions=[],
        body=L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=2,
            run=L4.Reference(name="x"),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.Let)
    assert len(actual.body.bindings) == 2
    assert actual.body.bindings[0] == (
        "break0",
        L4.Mutable(oftype=L4.Bool()),
        L4.HeapAllocate(val=L4.Immediate(value=False)),
    )
    assert actual.body.bindings[1] == (
        "continue0",
        L4.Mutable(oftype=L4.Bool()),
        L4.HeapAllocate(val=L4.Immediate(value=False)),
    )

    bunch = actual.body.body
    assert isinstance(bunch, L4.Bunch)
    assert len(bunch.expressions) == 2

    first_iter = bunch.expressions[0]
    assert isinstance(first_iter, L4.Bunch)

    reset_continue = first_iter.expressions[0]
    assert reset_continue == L4.Set(
        target=L4.Reference(name="continue0"),
        index=0,
        value=L4.Immediate(value=False),
    )

    bind_x = first_iter.expressions[1]
    assert isinstance(bind_x, L4.Let)
    [(binder, ty, value)] = bind_x.bindings
    assert binder == "x"
    assert ty == L4.Int()
    assert value == L4.Get(target=L4.Reference(name="xs"), index=0)

    second_iter_guard = bunch.expressions[1]
    assert isinstance(second_iter_guard, L4.If)
    assert second_iter_guard.condition == L4.Get(target=L4.Reference(name="break0"), index=0)

    second_iter = second_iter_guard.otherwise
    assert isinstance(second_iter, L4.Bunch)
    bind_x_second = second_iter.expressions[1]
    assert isinstance(bind_x_second, L4.Let)
    assert bind_x_second.bindings[0][2] == L4.Get(target=L4.Reference(name="xs"), index=1)


def test_convert_to_l4_lowers_definitions_and_body():
    program = L5.Program(
        definitions=[
            (
                "f",
                L4.FuncType(parameters=[L4.Bool()], result=L4.Bool()),
                L4.Function(
                    params=[("b", L4.Bool())],
                    body=L4.Reference(name="b"),
                ),
            )
        ],
        body=L5.Switch(
            scrutinee=L4.Immediate(value=True),
            cases=[L5.SwitchCase(value=True, body=L4.Immediate(value=1))],
            default=L4.Immediate(value=0),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual, L4.Program)
    assert len(actual.definitions) == 1
    name, ty, ex = actual.definitions[0]
    assert name == "f"
    assert ty == L4.FuncType(parameters=[L4.Bool()], result=L4.Bool())
    assert isinstance(ex, L4.Function)
    assert ex.body == L4.Reference(name="b")

    assert isinstance(actual.body, L4.Let)


def test_convert_to_l3_returns_l3_program():
    program = L5.Program(
        definitions=[],
        body=L5.ShortCircuit(
            operator="&&",
            left=L4.Immediate(value=True),
            right=L4.Immediate(value=False),
        ),
    )
    actual = convert_to_l3(program)
    assert isinstance(actual, L3Program)


def test_lower_expression_handles_remaining_passthrough_nodes():
    fresh = SequentialNameGenerator()
    context = {"p": L4.Pair(type1=L4.Int(), type2=L4.Bool())}

    fn = L4.Function(params=[("x", L4.Int())], body=L4.Reference(name="x"))
    if_ex = L4.If(
        condition=L4.Immediate(value=True),
        consequent=L4.Immediate(value=1),
        otherwise=L4.Immediate(value=0),
    )
    op = L4.Operation(operator="+", left=L4.Immediate(value=1), right=L4.Immediate(value=2))
    call = L4.Call(target=L4.Reference(name="f"), arguments=[L4.Immediate(value=1)])
    heap = L4.HeapAllocate(val=L4.Immediate(value=5))
    newpair = L4.NewPair(
        val1=L4.Immediate(value=1),
        val2=L4.Immediate(value=False),
        typeof=L4.Pair(type1=L4.Int(), type2=L4.Bool()),
    )
    set_ex = L4.Set(target=L4.Reference(name="m"), index=0, value=L4.Immediate(value=9))
    capsule = L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=7))
    ref = L4.Reference(name="x")
    imm = L4.Immediate(value=3)
    empty = L4.Empty()
    get_ex = L4.Get(target=L4.Reference(name="p"), index=0)
    newlist = L4.NewList(size=3, typeof=L4.Int())

    assert _lower_expression(fn, fresh=fresh, context=context, loop=None) == fn
    assert _lower_expression(if_ex, fresh=fresh, context=context, loop=None) == if_ex
    assert _lower_expression(op, fresh=fresh, context=context, loop=None) == op
    assert _lower_expression(call, fresh=fresh, context=context, loop=None) == call
    assert _lower_expression(heap, fresh=fresh, context=context, loop=None) == heap
    assert _lower_expression(newpair, fresh=fresh, context=context, loop=None) == newpair
    assert _lower_expression(set_ex, fresh=fresh, context=context, loop=None) == set_ex
    assert _lower_expression(capsule, fresh=fresh, context=context, loop=None) == capsule
    assert _lower_expression(ref, fresh=fresh, context=context, loop=None) == ref
    assert _lower_expression(imm, fresh=fresh, context=context, loop=None) == imm
    assert _lower_expression(empty, fresh=fresh, context=context, loop=None) == empty
    assert _lower_expression(get_ex, fresh=fresh, context=context, loop=None) == get_ex
    assert _lower_expression(newlist, fresh=fresh, context=context, loop=None) == newlist


def test_lower_expression_handles_let_and_letrec():
    fresh = SequentialNameGenerator()

    let_ex = L4.Let(
        bindings=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )
    letrec_ex = L4.LetRec(
        bindings=[
            (
                "f",
                L4.FuncType(parameters=[L4.Int()], result=L4.Int()),
                L4.Function(params=[("n", L4.Int())], body=L4.Reference(name="n")),
            )
        ],
        body=L4.Reference(name="f"),
    )

    lowered_let = _lower_expression(let_ex, fresh=fresh, context={}, loop=None)
    lowered_letrec = _lower_expression(letrec_ex, fresh=fresh, context={}, loop=None)

    assert lowered_let == let_ex
    assert lowered_letrec == letrec_ex


def test_lower_expression_handles_bunch_with_loop_guarding():
    fresh = SequentialNameGenerator()
    loop = LoopContext(break_flag="breakX", continue_flag="continueX")

    expr = L4.Bunch(
        expressions=[
            L4.Immediate(value=1),
            L4.Immediate(value=2),
            L4.Immediate(value=3),
        ]
    )

    actual = _lower_expression(expr, fresh=fresh, context={}, loop=loop)

    assert isinstance(actual, L4.Bunch)
    assert actual.expressions[0] == L4.Immediate(value=1)
    assert isinstance(actual.expressions[1], L4.If)
    assert isinstance(actual.expressions[2], L4.If)


def test_lower_expression_unhandled_branch():
    fresh = SequentialNameGenerator()
    with pytest.raises(TypeError, match="unhandled L5 expression"):
        _lower_expression(object(), fresh=fresh, context={}, loop=None)


def test_lower_bunch_empty_returns_empty():
    fresh = SequentialNameGenerator()
    actual = _lower_expression(
        L4.Bunch(expressions=[]),
        fresh=fresh,
        context={},
        loop=None,
    )
    assert actual == L4.Empty()


def test_foreach_zero_count_lowers_to_empty_body():
    program = L5.Program(
        definitions=[],
        body=L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=0,
            run=L4.Reference(name="x"),
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual.body, L4.Let)
    assert len(actual.body.bindings) == 2
    assert actual.body.bindings[0] == (
        "break0",
        L4.Mutable(oftype=L4.Bool()),
        L4.HeapAllocate(val=L4.Immediate(value=False)),
    )
    assert actual.body.bindings[1] == (
        "continue0",
        L4.Mutable(oftype=L4.Bool()),
        L4.HeapAllocate(val=L4.Immediate(value=False)),
    )
    assert actual.body.body == L4.Empty()