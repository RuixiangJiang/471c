from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from L3 import syntax as L3
from L4 import syntax as L4
from L4.convert import convert_to_l3 as convert_l4_to_l3

from . import syntax as L5

type Context = Mapping[str, L4.Type]


class SequentialNameGenerator:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def __call__(self, candidate: str) -> str:
        current = self._counters[candidate]
        self._counters[candidate] += 1
        return f"{candidate}{current}"


@dataclass(frozen=True)
class LoopContext:
    break_flag: str
    continue_flag: str


def dummy_parse(code: str) -> L5.Program:
    return L5.Program(definitions=[(code, L4.Void(), L4.Empty())], body=L4.Empty())


def _bool(value: bool) -> L4.Immediate:
    return L4.Immediate(value=value)


def _guarded_void(
    expr: L4.Expression,
    *,
    loop: LoopContext | None,
) -> L4.Expression:
    if loop is None:
        return expr

    stop_test = L4.If(
        condition=L4.Get(target=L4.Reference(name=loop.break_flag), index=0),
        consequent=_bool(True),
        otherwise=L4.Get(target=L4.Reference(name=loop.continue_flag), index=0),
    )
    return L4.If(condition=stop_test, consequent=L4.Empty(), otherwise=expr)


def _lower_bunch(
    expressions: Sequence[L5.Expression | L4.Expression],
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    loop: LoopContext | None,
) -> L4.Expression:
    if not expressions:
        return L4.Empty()

    lowered: list[L4.Expression] = []
    for i, ex in enumerate(expressions):
        item = _lower_expression(ex, fresh=fresh, context=context, loop=loop)
        if i == 0:
            lowered.append(item)
        else:
            lowered.append(_guarded_void(item, loop=loop))
    return L4.Bunch(expressions=lowered)


def _lower_short_circuit(expr: L5.ShortCircuit, *, fresh: SequentialNameGenerator, context: Context, loop: LoopContext | None) -> L4.Expression:
    left = _lower_expression(expr.left, fresh=fresh, context=context, loop=loop)
    right = _lower_expression(expr.right, fresh=fresh, context=context, loop=loop)

    if expr.operator == "&&":
        return L4.If(condition=left, consequent=right, otherwise=_bool(False))
    if expr.operator == "||":
        return L4.If(condition=left, consequent=_bool(True), otherwise=right)
    raise ValueError(f"unknown short-circuit operator: {expr.operator}")


def _lower_switch(expr: L5.Switch, *, fresh: SequentialNameGenerator, context: Context, loop: LoopContext | None) -> L4.Expression:
    scrutinee_name = fresh("switch_scrutinee")
    scrutinee_ref = L4.Reference(name=scrutinee_name)

    type_hint = (
        L4.Bool()
        if all(isinstance(c.value, bool) for c in expr.cases)
        else L4.Int()
    )

    result = _lower_expression(
        expr.default,
        fresh=fresh,
        context={**context, scrutinee_name: type_hint},
        loop=loop,
    )

    for case in reversed(expr.cases):
        body = _lower_expression(
            case.body,
            fresh=fresh,
            context={**context, scrutinee_name: type_hint},
            loop=loop,
        )
        result = L4.If(
            condition=L4.Operation(
                operator="==",
                left=scrutinee_ref,
                right=L4.Immediate(value=case.value),
            ),
            consequent=body,
            otherwise=result,
        )

    return L4.Let(
        bindings=[
            (
                scrutinee_name,
                type_hint,
                _lower_expression(expr.scrutinee, fresh=fresh, context=context, loop=loop),
            )
        ],
        body=result,
    )


def _lower_while(
    condition: L5.Expression | L4.Expression,
    run: L5.Expression | L4.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
) -> L4.Expression:
    loop_name = fresh("while")
    break_flag = fresh("break")
    continue_flag = fresh("continue")

    inner_loop = LoopContext(break_flag=break_flag, continue_flag=continue_flag)

    lowered_condition = _lower_expression(condition, fresh=fresh, context=context, loop=None)
    lowered_run = _lower_expression(run, fresh=fresh, context=context, loop=inner_loop)

    condition_without_break = L4.If(
        condition=L4.Get(target=L4.Reference(name=break_flag), index=0),
        consequent=_bool(False),
        otherwise=lowered_condition,
    )

    body = _lower_bunch(
        [
            L4.Set(
                target=L4.Reference(name=continue_flag),
                index=0,
                value=_bool(False),
            ),
            lowered_run,
            L4.If(
                condition=L4.Get(target=L4.Reference(name=break_flag), index=0),
                consequent=L4.Empty(),
                otherwise=L4.Call(target=L4.Reference(name=loop_name), arguments=[]),
            ),
        ],
        fresh=fresh,
        context={
            **context,
            break_flag: L4.Mutable(oftype=L4.Bool()),
            continue_flag: L4.Mutable(oftype=L4.Bool()),
        },
        loop=None,
    )

    return L4.LetRec(
        bindings=[
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (
                loop_name,
                L4.FuncType(parameters=[], result=L4.Void()),
                L4.Function(
                    params=[],
                    body=L4.If(
                        condition=condition_without_break,
                        consequent=body,
                        otherwise=L4.Empty(),
                    ),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name=loop_name), arguments=[]),
    )


def _lower_for(
    times: int | L5.Expression | L4.Expression,
    run: L5.Expression | L4.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
) -> L4.Expression:
    loop_name = fresh("for")
    counter_name = fresh("for_counter")
    break_flag = fresh("break")
    continue_flag = fresh("continue")

    inner_loop = LoopContext(break_flag=break_flag, continue_flag=continue_flag)

    lowered_times: L4.Expression
    if isinstance(times, int):
        lowered_times = L4.Immediate(value=times)
    else:
        lowered_times = _lower_expression(times, fresh=fresh, context=context, loop=None)

    lowered_run = _lower_expression(run, fresh=fresh, context=context, loop=inner_loop)

    check_times = L4.Operation(
        operator="<",
        left=L4.Immediate(value=0),
        right=L4.Get(target=L4.Reference(name=counter_name), index=0),
    )

    decrement = L4.Set(
        target=L4.Reference(name=counter_name),
        index=0,
        value=L4.Operation(
            operator="-",
            left=L4.Get(target=L4.Reference(name=counter_name), index=0),
            right=L4.Immediate(value=1),
        ),
    )

    body = _lower_bunch(
        [
            L4.Set(
                target=L4.Reference(name=continue_flag),
                index=0,
                value=_bool(False),
            ),
            decrement,
            lowered_run,
            L4.If(
                condition=L4.Get(target=L4.Reference(name=break_flag), index=0),
                consequent=L4.Empty(),
                otherwise=L4.Call(target=L4.Reference(name=loop_name), arguments=[]),
            ),
        ],
        fresh=fresh,
        context={
            **context,
            counter_name: L4.Mutable(oftype=L4.Int()),
            break_flag: L4.Mutable(oftype=L4.Bool()),
            continue_flag: L4.Mutable(oftype=L4.Bool()),
        },
        loop=None,
    )

    return L4.LetRec(
        bindings=[
            (counter_name, L4.Mutable(oftype=L4.Int()), lowered_times),
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (
                loop_name,
                L4.FuncType(parameters=[], result=L4.Void()),
                L4.Function(
                    params=[],
                    body=L4.If(condition=check_times, consequent=body, otherwise=L4.Empty()),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name=loop_name), arguments=[]),
    )


def _lower_foreach(
    expr: L5.Foreach,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
) -> L4.Expression:
    break_flag = fresh("break")
    continue_flag = fresh("continue")
    loop = LoopContext(break_flag=break_flag, continue_flag=continue_flag)

    per_item: list[L4.Expression] = []

    extended_context = {
        **context,
        break_flag: L4.Mutable(oftype=L4.Bool()),
        continue_flag: L4.Mutable(oftype=L4.Bool()),
        expr.binder: expr.typeof,
    }

    for i in range(expr.count):
        iteration_body = _lower_expression(expr.run, fresh=fresh, context=extended_context, loop=loop)

        one_iteration = L4.Bunch(
            expressions=[
                L4.Set(
                    target=L4.Reference(name=continue_flag),
                    index=0,
                    value=_bool(False),
                ),
                L4.Let(
                    bindings=[
                        (
                            expr.binder,
                            expr.typeof,
                            L4.Get(target=expr.target, index=i),
                        )
                    ],
                    body=iteration_body,
                ),
            ]
        )

        if i == 0:
            per_item.append(one_iteration)
        else:
            per_item.append(
                L4.If(
                    condition=L4.Get(target=L4.Reference(name=break_flag), index=0),
                    consequent=L4.Empty(),
                    otherwise=one_iteration,
                )
            )

    if not per_item:
        foreach_body: L4.Expression = L4.Empty()
    else:
        foreach_body = L4.Bunch(expressions=per_item)

    return L4.Let(
        bindings=[
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
        ],
        body=foreach_body,
    )


def _lower_expression(
    expression: L5.Expression | L4.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    loop: LoopContext | None,
) -> L4.Expression:
    match expression:
        case L5.ShortCircuit():
            return _lower_short_circuit(expression, fresh=fresh, context=context, loop=loop)

        case L5.Switch():
            return _lower_switch(expression, fresh=fresh, context=context, loop=loop)

        case L5.Break():
            if loop is None:
                raise ValueError("break used outside of a loop")
            return L4.Set(
                target=L4.Reference(name=loop.break_flag),
                index=0,
                value=_bool(True),
            )

        case L5.Continue():
            if loop is None:
                raise ValueError("continue used outside of a loop")
            return L4.Set(
                target=L4.Reference(name=loop.continue_flag),
                index=0,
                value=_bool(True),
            )

        case L5.Foreach():
            return _lower_foreach(expression, fresh=fresh, context=context)

        case L4.While(condition=condition, run=run):
            return _lower_while(condition, run, fresh=fresh, context=context)

        case L4.For(times=times, run=run):
            return _lower_for(times, run, fresh=fresh, context=context)

        case L4.Bunch(expressions=expressions):
            return _lower_bunch(expressions, fresh=fresh, context=context, loop=loop)

        case L4.Let(bindings=bindings, body=body):
            lowered_bindings = [
                (name, ty, _lower_expression(ex, fresh=fresh, context=context, loop=None))
                for name, ty, ex in bindings
            ]
            local = {name: ty for name, ty, _ in bindings}
            return L4.Let(
                bindings=lowered_bindings,
                body=_lower_expression(body, fresh=fresh, context={**context, **local}, loop=loop),
            )

        case L4.LetRec(bindings=bindings, body=body):
            local = {name: ty for name, ty, _ in bindings}
            lowered_bindings = [
                (name, ty, _lower_expression(ex, fresh=fresh, context={**context, **local}, loop=None))
                for name, ty, ex in bindings
            ]
            return L4.LetRec(
                bindings=lowered_bindings,
                body=_lower_expression(body, fresh=fresh, context={**context, **local}, loop=loop),
            )

        case L4.Function(params=params, body=body):
            local = {name: ty for name, ty in params}
            return L4.Function(
                params=params,
                body=_lower_expression(body, fresh=fresh, context={**context, **local}, loop=None),
            )

        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            return L4.If(
                condition=_lower_expression(condition, fresh=fresh, context=context, loop=None),
                consequent=_lower_expression(consequent, fresh=fresh, context=context, loop=loop),
                otherwise=_lower_expression(otherwise, fresh=fresh, context=context, loop=loop),
            )

        case L4.Operation(operator=operator, left=left, right=right):
            return L4.Operation(
                operator=operator,
                left=_lower_expression(left, fresh=fresh, context=context, loop=None),
                right=_lower_expression(right, fresh=fresh, context=context, loop=None),
            )

        case L4.Call(target=target, arguments=arguments):
            return L4.Call(
                target=_lower_expression(target, fresh=fresh, context=context, loop=None),
                arguments=[
                    _lower_expression(arg, fresh=fresh, context=context, loop=None) for arg in arguments
                ],
            )

        case L4.HeapAllocate(val=val):
            return L4.HeapAllocate(val=_lower_expression(val, fresh=fresh, context=context, loop=None))

        case L4.NewPair(val1=val1, val2=val2, typeof=typeof):
            return L4.NewPair(
                val1=_lower_expression(val1, fresh=fresh, context=context, loop=None),
                val2=_lower_expression(val2, fresh=fresh, context=context, loop=None),
                typeof=typeof,
            )

        case L4.Set(target=target, index=index, value=value):
            return L4.Set(
                target=target,
                index=index,
                value=_lower_expression(value, fresh=fresh, context=context, loop=None),
            )

        case L4.Capsule(typeof=typeof, expression=inner):
            return L4.Capsule(
                typeof=typeof,
                expression=_lower_expression(inner, fresh=fresh, context=context, loop=None),
            )

        case (
            L4.Reference()
            | L4.Immediate()
            | L4.Empty()
            | L4.Get()
            | L4.NewList()
        ):
            return expression

        case _:
            raise TypeError(f"unhandled L5 expression: {expression!r}")


def convert_to_l4(program: L5.Program) -> L4.Program:
    fresh = SequentialNameGenerator()

    context = {name: ty for name, ty, _ in program.definitions}

    lowered_defs = []
    for name, ty, ex in program.definitions:
        lowered_defs.append(
            (
                name,
                ty,
                _lower_expression(ex, fresh=fresh, context=context, loop=None),
            )
        )

    lowered_body = _lower_expression(program.body, fresh=fresh, context=context, loop=None)

    return L4.Program(definitions=lowered_defs, body=lowered_body)


def convert_to_l3(program: L5.Program) -> L3.Program:
    return convert_l4_to_l3(convert_to_l4(program))