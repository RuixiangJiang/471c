from __future__ import annotations

from L3 import syntax as L3
from L4 import syntax as L4
from L4.convert import convert_to_l3 as convert_l4_to_l3

from . import syntax as L5
from .class_convert import (
    Context,
    infer_type,
    lower_field_access,
    lower_field_assign,
    lower_method_call,
    lower_method_definition,
    lower_new_object,
    lower_type,
    same_type,
    type_error,
)
from .inheritance_convert import MethodInfo, collect_classes
from .minor_convert import (
    LoopContext,
    SequentialNameGenerator,
    lower_bunch,
    lower_for,
    lower_foreach,
    lower_short_circuit,
    lower_switch,
    lower_while,
    make_bool,
)


def dummy_parse(code: str) -> L5.Program:
    return L5.Program(
        classes=[],
        definitions=[(code, L4.Void(), L4.Empty())],
        body=L4.Empty(),
    )


def _lower_expression(
    expression: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env,
    current_class: str | None,
    loop: LoopContext | None,
) -> L4.Expression:
    match expression:
        case L5.This():
            if current_class is None:
                raise ValueError("this used outside of a method body")
            return L4.Reference(name="this")

        case L5.NewObject():
            return lower_new_object(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                lower_expr=_lower_expression,
            )

        case L5.FieldAccess():
            return lower_field_access(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                infer_expr=infer_type,
                lower_expr=_lower_expression,
            )

        case L5.FieldAssign():
            return lower_field_assign(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                infer_expr=infer_type,
                lower_expr=_lower_expression,
            )

        case L5.MethodCall():
            return lower_method_call(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                infer_expr=infer_type,
                lower_expr=_lower_expression,
            )

        case L5.ShortCircuit():
            return lower_short_circuit(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
                lower_expr=_lower_expression,
            )

        case L5.Switch():
            return lower_switch(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
                lower_expr=_lower_expression,
            )

        case L5.Break():
            if loop is None:
                raise ValueError("break used outside of a loop")
            return L4.Set(
                target=L4.Reference(name=loop.break_flag),
                index=0,
                value=make_bool(True),
            )

        case L5.Continue():
            if loop is None:
                raise ValueError("continue used outside of a loop")
            return L4.Set(
                target=L4.Reference(name=loop.continue_flag),
                index=0,
                value=make_bool(True),
            )

        case L5.Foreach():
            return lower_foreach(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                lower_expr=_lower_expression,
                lower_type=lower_type,
            )

        case L4.While(condition=condition, run=run):
            return lower_while(
                condition,
                run,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                lower_expr=_lower_expression,
            )

        case L4.For(times=times, run=run):
            return lower_for(
                times,
                run,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                lower_expr=_lower_expression,
            )

        case L4.Bunch(expressions=expressions):
            return lower_bunch(
                expressions,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
                lower_expr=_lower_expression,
            )

        case L4.Let(bindings=bindings, body=body):
            lowered_bindings = [
                (
                    name,
                    lower_type(ty, class_env),
                    _lower_expression(
                        ex,
                        fresh=fresh,
                        context=context,
                        class_env=class_env,
                        current_class=current_class,
                        loop=None,
                    ),
                )
                for name, ty, ex in bindings
            ]
            local = {name: ty for name, ty, _ in bindings}
            return L4.Let(
                bindings=lowered_bindings,
                body=_lower_expression(
                    body,
                    fresh=fresh,
                    context={**context, **local},
                    class_env=class_env,
                    current_class=current_class,
                    loop=loop,
                ),
            )

        case L4.LetRec(bindings=bindings, body=body):
            local = {name: ty for name, ty, _ in bindings}
            lowered_bindings = [
                (
                    name,
                    lower_type(ty, class_env),
                    _lower_expression(
                        ex,
                        fresh=fresh,
                        context={**context, **local},
                        class_env=class_env,
                        current_class=current_class,
                        loop=None,
                    ),
                )
                for name, ty, ex in bindings
            ]
            return L4.LetRec(
                bindings=lowered_bindings,
                body=_lower_expression(
                    body,
                    fresh=fresh,
                    context={**context, **local},
                    class_env=class_env,
                    current_class=current_class,
                    loop=loop,
                ),
            )

        case L4.Function(params=params, body=body):
            local = {name: ty for name, ty in params}
            return L4.Function(
                params=[(name, lower_type(ty, class_env)) for name, ty in params],
                body=_lower_expression(
                    body,
                    fresh=fresh,
                    context={**context, **local},
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
            )

        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            return L4.If(
                condition=_lower_expression(
                    condition,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
                consequent=_lower_expression(
                    consequent,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=loop,
                ),
                otherwise=_lower_expression(
                    otherwise,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=loop,
                ),
            )

        case L4.Operation(operator=operator, left=left, right=right):
            return L4.Operation(
                operator=operator,
                left=_lower_expression(
                    left,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
                right=_lower_expression(
                    right,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
            )

        case L4.Call(target=target, arguments=arguments):
            return L4.Call(
                target=_lower_expression(
                    target,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
                arguments=[
                    _lower_expression(
                        arg,
                        fresh=fresh,
                        context=context,
                        class_env=class_env,
                        current_class=current_class,
                        loop=None,
                    )
                    for arg in arguments
                ],
            )

        case L4.HeapAllocate(val=val):
            return L4.HeapAllocate(
                val=_lower_expression(
                    val,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                )
            )

        case L4.NewPair(val1=val1, val2=val2, typeof=typeof):
            return L4.NewPair(
                val1=_lower_expression(
                    val1,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
                val2=_lower_expression(
                    val2,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
                typeof=lower_type(typeof, class_env),
            )

        case L4.Set(target=target, index=index, value=value):
            return L4.Set(
                target=target,
                index=index,
                value=_lower_expression(
                    value,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
            )

        case L4.Capsule(typeof=typeof, expression=inner):
            return L4.Capsule(
                typeof=lower_type(typeof, class_env),
                expression=_lower_expression(
                    inner,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                ),
            )

        case L4.Reference() | L4.Immediate() | L4.Empty() | L4.Get() | L4.NewList():
            return expression

        case _:
            raise TypeError(f"unhandled L5 expression: {expression!r}")


def convert_to_l4(program: L5.Program) -> L4.Program:
    fresh = SequentialNameGenerator()
    class_env = collect_classes(program.classes)

    lowered_defs: list[tuple[str, L4.Type, L4.Expression]] = []

    generated_names: set[str] = set()
    for cls in program.classes:
        for method in cls.methods:
            lowered = lower_method_definition(
                cls.name,
                MethodInfo(
                    name=method.name,
                    parameters=method.parameters,
                    returns=method.returns,
                    body=method.body,
                    owner=cls.name,
                ),
                fresh=fresh,
                class_env=class_env,
                infer_expr=infer_type,
                lower_expr=_lower_expression,
            )
            if lowered[0] in generated_names:
                raise type_error(f"duplicate generated method name {lowered[0]!r}")
            generated_names.add(lowered[0])
            lowered_defs.append(lowered)

    context: dict[str, L5.Type] = {name: ty for name, ty, _ in program.definitions}
    for name, _, _ in lowered_defs:
        context[name] = L4.Void()

    for name, ty, ex in program.definitions:
        actual = infer_type(
            ex,
            context=context,
            class_env=class_env,
            current_class=None,
        )
        if not same_type(actual, ty):
            raise type_error(f"definition {name!r} expected {ty!r}, got {actual!r}")

        lowered_defs.append(
            (
                name,
                lower_type(ty, class_env),
                _lower_expression(
                    ex,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=None,
                    loop=None,
                ),
            )
        )

    lowered_body = _lower_expression(
        program.body,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=None,
        loop=None,
    )

    return L4.Program(definitions=lowered_defs, body=lowered_body)


def convert_to_l3(program: L5.Program) -> L3.Program:
    return convert_l4_to_l3(convert_to_l4(program))