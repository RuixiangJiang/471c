from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from L4 import syntax as L4

from . import syntax as L5

type Context = Mapping[str, L5.Type]


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


def make_bool(value: bool) -> L4.Immediate:
    return L4.Immediate(value=value)


def guarded_void(
    expr: L4.Expression,
    *,
    loop: LoopContext | None,
) -> L4.Expression:
    if loop is None:
        return expr

    stop_test = L4.If(
        condition=L4.Get(target=L4.Reference(name=loop.break_flag), index=0),
        consequent=make_bool(True),
        otherwise=L4.Get(target=L4.Reference(name=loop.continue_flag), index=0),
    )
    return L4.If(condition=stop_test, consequent=L4.Empty(), otherwise=expr)


def lower_short_circuit(
    expr: L5.ShortCircuit,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    loop: LoopContext | None,
    lower_expr,
) -> L4.Expression:
    left = lower_expr(
        expr.left,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )
    right = lower_expr(
        expr.right,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )

    if expr.operator == "&&":
        return L4.If(condition=left, consequent=right, otherwise=make_bool(False))
    if expr.operator == "||":
        return L4.If(condition=left, consequent=make_bool(True), otherwise=right)
    raise ValueError(f"unknown short-circuit operator: {expr.operator}")


def lower_switch(
    expr: L5.Switch,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    loop: LoopContext | None,
    lower_expr,
) -> L4.Expression:
    scrutinee_name = fresh("switch_scrutinee")
    scrutinee_ref = L4.Reference(name=scrutinee_name)

    type_hint = (
        L4.Bool()
        if all(isinstance(c.value, bool) for c in expr.cases)
        else L4.Int()
    )

    result = lower_expr(
        expr.default,
        fresh=fresh,
        context={**context, scrutinee_name: type_hint},
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )

    for case in reversed(expr.cases):
        body = lower_expr(
            case.body,
            fresh=fresh,
            context={**context, scrutinee_name: type_hint},
            class_env=class_env,
            current_class=current_class,
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

    lowered_scrutinee = lower_expr(
        expr.scrutinee,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )

    return L4.Let(
        bindings=[(scrutinee_name, type_hint, lowered_scrutinee)],
        body=result,
    )


def lower_bunch(
    expressions: Sequence[L5.Expression],
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    loop: LoopContext | None,
    lower_expr,
) -> L4.Expression:
    if not expressions:
        return L4.Empty()

    lowered: list[L4.Expression] = []
    for i, ex in enumerate(expressions):
        item = lower_expr(
            ex,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            loop=loop,
        )
        if i == 0:
            lowered.append(item)
        else:
            lowered.append(guarded_void(item, loop=loop))
    return L4.Bunch(expressions=lowered)


def lower_while(
    condition: L5.Expression,
    run: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    lower_expr,
) -> L4.Expression:
    loop_name = fresh("while")
    break_flag = fresh("break")
    continue_flag = fresh("continue")

    inner_loop = LoopContext(break_flag=break_flag, continue_flag=continue_flag)

    lowered_condition = lower_expr(
        condition,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )
    lowered_run = lower_expr(
        run,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=inner_loop,
    )

    condition_without_break = L4.If(
        condition=L4.Get(target=L4.Reference(name=break_flag), index=0),
        consequent=make_bool(False),
        otherwise=lowered_condition,
    )

    body = lower_bunch(
        [
            L4.Set(
                target=L4.Reference(name=continue_flag),
                index=0,
                value=make_bool(False),
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
        class_env=class_env,
        current_class=current_class,
        loop=None,
        lower_expr=lower_expr,
    )

    return L4.LetRec(
        bindings=[
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
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


def lower_for(
    times: int | L5.Expression,
    run: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    lower_expr,
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
        lowered_times = lower_expr(
            times,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            loop=None,
        )

    lowered_run = lower_expr(
        run,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=inner_loop,
    )

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

    body = lower_bunch(
        [
            L4.Set(
                target=L4.Reference(name=continue_flag),
                index=0,
                value=make_bool(False),
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
        class_env=class_env,
        current_class=current_class,
        loop=None,
        lower_expr=lower_expr,
    )

    return L4.LetRec(
        bindings=[
            (counter_name, L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=lowered_times)),
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
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


def lower_foreach(
    expr: L5.Foreach,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    lower_expr,
    lower_type,
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
        iteration_body = lower_expr(
            expr.run,
            fresh=fresh,
            context=extended_context,
            class_env=class_env,
            current_class=current_class,
            loop=loop,
        )

        one_iteration = L4.Bunch(
            expressions=[
                L4.Set(
                    target=L4.Reference(name=continue_flag),
                    index=0,
                    value=make_bool(False),
                ),
                L4.Let(
                    bindings=[
                        (
                            expr.binder,
                            lower_type(expr.typeof, class_env),
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

    foreach_body: L4.Expression = L4.Empty() if not per_item else L4.Bunch(expressions=per_item)

    return L4.Let(
        bindings=[
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=make_bool(False))),
        ],
        body=foreach_body,
    )