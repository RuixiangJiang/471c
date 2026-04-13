from __future__ import annotations

from collections.abc import Mapping, Sequence

from L4 import syntax as L4

from . import syntax as L5
from .inheritance_convert import ClassInfo, MethodInfo, field_index, field_info, method_info

type Context = Mapping[str, L5.Type]


def type_error(message: str) -> TypeError:
    return TypeError(message)


def same_type(left: L5.Type, right: L5.Type) -> bool:
    return left == right


def class_payload_type(info: ClassInfo, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    lowered = [lower_type(field.typeof, class_env) for field in info.all_fields]

    if not lowered:
        return L4.Void()

    current = lowered[-1]
    for item in reversed(lowered[:-1]):
        current = L4.Pair(type1=item, type2=current)
    return current


def class_rep_type(info: ClassInfo, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    return L4.Mutable(oftype=class_payload_type(info, class_env))


def lower_type(typeof: L5.Type, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    match typeof:
        case L5.ClassType(name=name):
            if name not in class_env:
                raise type_error(f"unknown class type {name!r}")
            return class_rep_type(class_env[name], class_env)

        case (
            L4.Int()
            | L4.Bool()
            | L4.Void()
            | L4.Symbol()
            | L4.Mutable()
            | L4.List()
            | L4.Pair()
            | L4.FuncType()
        ):
            return typeof

        case _:
            raise type_error(f"cannot lower type {typeof!r}")


def infer_type(
    expression: L5.Expression,
    *,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
) -> L5.Type:
    match expression:
        case L5.This():
            if current_class is None:
                raise type_error("this used outside of a method body")
            return L5.ClassType(name=current_class)

        case L5.NewObject(name=name, arguments=arguments):
            if name not in class_env:
                raise type_error(f"unknown class {name!r}")
            info = class_env[name]
            if len(arguments) != len(info.all_fields):
                raise type_error(
                    f"new {name} expects {len(info.all_fields)} arguments, got {len(arguments)}"
                )
            for arg, field in zip(arguments, info.all_fields, strict=True):
                actual = infer_type(
                    arg,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not same_type(actual, field.typeof):
                    raise type_error(
                        f"constructor argument for field {field.name!r} expected {field.typeof!r}, got {actual!r}"
                    )
            return L5.ClassType(name=name)

        case L5.FieldAccess(target=target, field=field):
            target_type = infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    return field_info(class_env[name], field).typeof
                case _:
                    raise type_error(f"field access requires class target, got {target_type!r}")

        case L5.FieldAssign(target=target, field=field, value=value):
            target_type = infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    expected = field_info(class_env[name], field).typeof
                case _:
                    raise type_error(f"field assignment requires class target, got {target_type!r}")

            actual = infer_type(
                value,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not same_type(actual, expected):
                raise type_error(f"field assignment expected {expected!r}, got {actual!r}")
            return L4.Void()

        case L5.MethodCall(target=target, method=method, arguments=arguments):
            target_type = infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    resolved = method_info(class_env[name], method)
                case _:
                    raise type_error(f"method call requires class target, got {target_type!r}")

            if len(arguments) != len(resolved.parameters):
                raise type_error(
                    f"method {name}.{method} expects {len(resolved.parameters)} arguments, got {len(arguments)}"
                )

            for arg, (_, expected) in zip(arguments, resolved.parameters, strict=True):
                actual = infer_type(
                    arg,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not same_type(actual, expected):
                    raise type_error(f"method argument expected {expected!r}, got {actual!r}")
            return resolved.returns

        case L5.ShortCircuit(left=left, right=right):
            left_type = infer_type(
                left,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            right_type = infer_type(
                right,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if left_type != L4.Bool() or right_type != L4.Bool():
                raise type_error("short-circuit operators require bool operands")
            return L4.Bool()

        case L5.Switch(scrutinee=scrutinee, cases=cases, default=default):
            scrutinee_type = infer_type(
                scrutinee,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if scrutinee_type not in (L4.Int(), L4.Bool()):
                raise type_error("switch scrutinee must be int or bool")

            default_type = infer_type(
                default,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            for case in cases:
                case_type = infer_type(
                    case.body,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not same_type(case_type, default_type):
                    raise type_error("all switch branches must have the same type")
            return default_type

        case L5.Break() | L5.Continue():
            return L4.Void()

        case L5.Foreach(binder=binder, typeof=typeof, target=target, run=run):
            target_type = context.get(target.name)
            if not isinstance(target_type, L4.List):
                raise type_error("foreach target must be a list reference")
            if not same_type(target_type.typeof, typeof):
                raise type_error("foreach binder type does not match list element type")
            return infer_type(
                run,
                context={**context, binder: typeof},
                class_env=class_env,
                current_class=current_class,
            )

        case L4.Reference(name=name):
            if name not in context:
                raise type_error(f"unknown reference {name!r}")
            return context[name]

        case L4.Immediate(value=value):
            if isinstance(value, bool):
                return L4.Bool()
            if isinstance(value, int):
                return L4.Int()
            return L4.Void()

        case L4.Operation(operator=operator, left=left, right=right):
            left_type = infer_type(
                left,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            right_type = infer_type(
                right,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if operator in {"+", "-", "*", "<"}:
                if left_type != L4.Int() or right_type != L4.Int():
                    raise type_error(f"operator {operator!r} requires int operands")
                return L4.Bool() if operator == "<" else L4.Int()
            if operator == "==":
                if not same_type(left_type, right_type):
                    raise type_error("== requires operands of the same type")
                return L4.Bool()
            raise type_error(f"unknown operator {operator!r}")

        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            cond_type = infer_type(
                condition,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if cond_type != L4.Bool():
                raise type_error("if condition must be bool")
            c_type = infer_type(
                consequent,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            o_type = infer_type(
                otherwise,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not same_type(c_type, o_type):
                raise type_error("if branches must have the same type")
            return c_type

        case L4.Let(bindings=bindings, body=body):
            next_context = dict(context)
            for name, ty, ex in bindings:
                actual = infer_type(
                    ex,
                    context=next_context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not same_type(actual, ty):
                    raise type_error(f"let binding {name!r} expected {ty!r}, got {actual!r}")
                next_context[name] = ty
            return infer_type(
                body,
                context=next_context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.LetRec(bindings=bindings, body=body):
            next_context = dict(context)
            for name, ty, _ in bindings:
                next_context[name] = ty
            return infer_type(
                body,
                context=next_context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.Function():
            raise type_error("cannot infer a bare function expression here without an expected type")

        case L4.Call():
            raise type_error("method dispatch is supported via MethodCall; bare call inference is not implemented")

        case L4.Empty():
            return L4.Void()

        case L4.NewList(typeof=typeof):
            return L4.List(typeof=typeof)

        case L4.NewPair(typeof=typeof):
            return typeof

        case L4.HeapAllocate(val=val):
            return L4.Mutable(
                oftype=infer_type(
                    val,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
            )

        case L4.Get(target=target, index=index):
            target_type = context.get(target.name)
            if target_type is None:
                raise type_error(f"unknown reference {target.name!r}")
            match target_type:
                case L4.Mutable(oftype=oftype):
                    if index != 0:
                        raise type_error("mutable get only supports index 0")
                    return oftype
                case L4.Pair(type1=t1, type2=t2):
                    if index == 0:
                        return t1
                    if index == 1:
                        return t2
                    raise type_error("pair get index must be 0 or 1")
                case L4.List(typeof=typeof):
                    return typeof
                case _:
                    raise type_error(f"get is not supported for type {target_type!r}")

        case L4.Set(target=target, index=index, value=value):
            target_type = context.get(target.name)
            if target_type is None:
                raise type_error(f"unknown reference {target.name!r}")
            value_type = infer_type(
                value,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L4.Mutable(oftype=oftype):
                    if index != 0 or not same_type(value_type, oftype):
                        raise type_error("invalid mutable set")
                case L4.Pair(type1=t1, type2=t2):
                    if index not in (0, 1):
                        raise type_error("pair set index must be 0 or 1")
                    expected = t1 if index == 0 else t2
                    if not same_type(value_type, expected):
                        raise type_error("invalid pair set")
                case L4.List(typeof=typeof):
                    if not same_type(value_type, typeof):
                        raise type_error("invalid list set")
                case _:
                    raise type_error(f"set is not supported for type {target_type!r}")
            return L4.Void()

        case L4.Capsule(typeof=typeof, expression=inner):
            actual = infer_type(
                inner,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not same_type(actual, typeof):
                raise type_error(f"capsule expected {typeof!r}, got {actual!r}")
            return typeof

        case L4.While(condition=condition, run=run):
            cond_type = infer_type(
                condition,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if cond_type != L4.Bool():
                raise type_error("while condition must be bool")
            infer_type(run, context=context, class_env=class_env, current_class=current_class)
            return L4.Void()

        case L4.For(times=times, run=run):
            if isinstance(times, int):
                pass
            else:
                times_type = infer_type(
                    times,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if times_type != L4.Int():
                    raise type_error("for times must be int")
            infer_type(run, context=context, class_env=class_env, current_class=current_class)
            return L4.Void()

        case L4.Bunch(expressions=expressions):
            if not expressions:
                return L4.Void()
            for ex in expressions[:-1]:
                infer_type(ex, context=context, class_env=class_env, current_class=current_class)
            return infer_type(
                expressions[-1],
                context=context,
                class_env=class_env,
                current_class=current_class,
            )

        case _:
            raise type_error(f"cannot infer type for {expression!r}")


def wrap_let(name: str, ty: L4.Type, ex: L4.Expression, body: L4.Expression) -> L4.Expression:
    return L4.Let(bindings=[(name, ty, ex)], body=body)


def field_path(field_index_value: int, field_count: int) -> list[int]:
    if field_count == 0:
        raise type_error("cannot compute field path for empty class")
    if field_index_value < 0 or field_index_value >= field_count:
        raise type_error("invalid field index")
    if field_count == 1:
        return []
    if field_index_value == 0:
        return [0]
    return [1, *field_path(field_index_value - 1, field_count - 1)]


def access_from_ref(
    current_ref: L4.Reference,
    current_type: L4.Type,
    path: Sequence[int],
    *,
    fresh,
) -> L4.Expression:
    if not path:
        return current_ref

    match current_type:
        case L4.Pair(type1=t1, type2=t2):
            step = path[0]
            child_type = t1 if step == 0 else t2
            tmp = fresh("field")
            return wrap_let(
                tmp,
                child_type,
                L4.Get(target=current_ref, index=step),
                access_from_ref(
                    L4.Reference(name=tmp),
                    child_type,
                    path[1:],
                    fresh=fresh,
                ),
            )
        case _:
            raise type_error(f"cannot descend into non-pair type {current_type!r}")


def rebuild_from_ref(
    current_ref: L4.Reference,
    current_type: L4.Type,
    path: Sequence[int],
    new_value: L4.Expression,
    *,
    fresh,
) -> L4.Expression:
    if not path:
        return new_value

    match current_type:
        case L4.Pair(type1=t1, type2=t2):
            step = path[0]
            rest = path[1:]

            if step == 0:
                if rest:
                    child_name = fresh("child")
                    new_left = wrap_let(
                        child_name,
                        t1,
                        L4.Get(target=current_ref, index=0),
                        rebuild_from_ref(
                            L4.Reference(name=child_name),
                            t1,
                            rest,
                            new_value,
                            fresh=fresh,
                        ),
                    )
                else:
                    new_left = new_value

                return L4.NewPair(
                    val1=new_left,
                    val2=L4.Get(target=current_ref, index=1),
                    typeof=L4.Pair(type1=t1, type2=t2),
                )

            if step == 1:
                if rest:
                    child_name = fresh("child")
                    new_right = wrap_let(
                        child_name,
                        t2,
                        L4.Get(target=current_ref, index=1),
                        rebuild_from_ref(
                            L4.Reference(name=child_name),
                            t2,
                            rest,
                            new_value,
                            fresh=fresh,
                        ),
                    )
                else:
                    new_right = new_value

                return L4.NewPair(
                    val1=L4.Get(target=current_ref, index=0),
                    val2=new_right,
                    typeof=L4.Pair(type1=t1, type2=t2),
                )

            raise type_error("pair path step must be 0 or 1")

        case _:
            raise type_error(f"cannot rebuild into non-pair type {current_type!r}")


def wrap_with_reference(
    lowered: L4.Expression,
    typeof: L4.Type,
    *,
    fresh,
    prefix: str,
    body_builder,
) -> L4.Expression:
    if isinstance(lowered, L4.Reference):
        return body_builder(lowered)

    temp = fresh(prefix)
    return L4.Let(
        bindings=[(temp, typeof, lowered)],
        body=body_builder(L4.Reference(name=temp)),
    )


def build_payload_expression(
    info: ClassInfo,
    arguments: Sequence[L4.Expression],
    *,
    class_env: Mapping[str, ClassInfo],
) -> L4.Expression:
    if len(arguments) != len(info.all_fields):
        raise type_error("constructor argument count mismatch")

    if not arguments:
        return L4.Empty()

    current = arguments[-1]
    lowered_types = [lower_type(field.typeof, class_env) for field in info.all_fields]
    current_type = lowered_types[-1]

    for ex, ty in zip(reversed(arguments[:-1]), reversed(lowered_types[:-1]), strict=True):
        current = L4.NewPair(
            val1=ex,
            val2=current,
            typeof=L4.Pair(type1=ty, type2=current_type),
        )
        current_type = L4.Pair(type1=ty, type2=current_type)
    return current


def lower_new_object(
    expr: L5.NewObject,
    *,
    fresh,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    lower_expr,
) -> L4.Expression:
    if expr.name not in class_env:
        raise type_error(f"unknown class {expr.name!r}")
    info = class_env[expr.name]

    lowered_args = [
        lower_expr(
            arg,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            loop=None,
        )
        for arg in expr.arguments
    ]
    payload = build_payload_expression(info, lowered_args, class_env=class_env)
    return L4.HeapAllocate(val=payload)


def lower_field_access(
    expr: L5.FieldAccess,
    *,
    fresh,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    infer_expr,
    lower_expr,
) -> L4.Expression:
    target_type = infer_expr(
        expr.target,
        context=context,
        class_env=class_env,
        current_class=current_class,
    )
    match target_type:
        case L5.ClassType(name=name):
            info = class_env[name]
        case _:
            raise type_error(f"field access requires class target, got {target_type!r}")

    lowered_target = lower_expr(
        expr.target,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )
    target_l4_type = lower_type(target_type, class_env)
    payload_type = class_payload_type(info, class_env)
    path = field_path(field_index(info, expr.field), len(info.all_fields))

    def body_builder(obj_ref: L4.Reference) -> L4.Expression:
        root_name = fresh("root")
        return L4.Let(
            bindings=[(root_name, payload_type, L4.Get(target=obj_ref, index=0))],
            body=access_from_ref(
                L4.Reference(name=root_name),
                payload_type,
                path,
                fresh=fresh,
            ),
        )

    return wrap_with_reference(
        lowered_target,
        target_l4_type,
        fresh=fresh,
        prefix="obj",
        body_builder=body_builder,
    )


def lower_field_assign(
    expr: L5.FieldAssign,
    *,
    fresh,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    infer_expr,
    lower_expr,
) -> L4.Expression:
    target_type = infer_expr(
        expr.target,
        context=context,
        class_env=class_env,
        current_class=current_class,
    )
    match target_type:
        case L5.ClassType(name=name):
            info = class_env[name]
        case _:
            raise type_error(f"field assignment requires class target, got {target_type!r}")

    lowered_target = lower_expr(
        expr.target,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )
    lowered_value = lower_expr(
        expr.value,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )

    payload_type = class_payload_type(info, class_env)
    path = field_path(field_index(info, expr.field), len(info.all_fields))
    target_l4_type = lower_type(target_type, class_env)

    def body_builder(obj_ref: L4.Reference) -> L4.Expression:
        root_name = fresh("root")
        root_ref = L4.Reference(name=root_name)
        rebuilt = rebuild_from_ref(
            root_ref,
            payload_type,
            path,
            lowered_value,
            fresh=fresh,
        )
        return L4.Let(
            bindings=[(root_name, payload_type, L4.Get(target=obj_ref, index=0))],
            body=L4.Set(target=obj_ref, index=0, value=rebuilt),
        )

    return wrap_with_reference(
        lowered_target,
        target_l4_type,
        fresh=fresh,
        prefix="obj",
        body_builder=body_builder,
    )


def lower_method_call(
    expr: L5.MethodCall,
    *,
    fresh,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    infer_expr,
    lower_expr,
) -> L4.Expression:
    target_type = infer_expr(
        expr.target,
        context=context,
        class_env=class_env,
        current_class=current_class,
    )
    match target_type:
        case L5.ClassType(name=name):
            resolved = method_info(class_env[name], expr.method)
            method_name = f"{resolved.owner}_{expr.method}"
        case _:
            raise type_error(f"method call requires class target, got {target_type!r}")

    lowered_target = lower_expr(
        expr.target,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )
    lowered_args = [
        lower_expr(
            arg,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            loop=None,
        )
        for arg in expr.arguments
    ]
    target_l4_type = lower_type(target_type, class_env)

    return wrap_with_reference(
        lowered_target,
        target_l4_type,
        fresh=fresh,
        prefix="obj",
        body_builder=lambda obj_ref: L4.Call(
            target=L4.Reference(name=method_name),
            arguments=[obj_ref, *lowered_args],
        ),
    )


def lower_method_definition(
    class_name: str,
    method: MethodInfo,
    *,
    fresh,
    class_env: Mapping[str, ClassInfo],
    infer_expr,
    lower_expr,
) -> tuple[str, L4.Type, L4.Expression]:
    this_type = L5.ClassType(name=class_name)
    context: dict[str, L5.Type] = {"this": this_type}
    for name, ty in method.parameters:
        context[name] = ty

    actual_returns = infer_expr(
        method.body,
        context=context,
        class_env=class_env,
        current_class=class_name,
    )
    if not same_type(actual_returns, method.returns):
        raise type_error(
            f"method {class_name}.{method.name} expected return {method.returns!r}, got {actual_returns!r}"
        )

    lowered_body = lower_expr(
        method.body,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=class_name,
        loop=None,
    )

    method_name = f"{class_name}_{method.name}"
    func_type = L4.FuncType(
        parameters=[
            lower_type(this_type, class_env),
            *[lower_type(ty, class_env) for _, ty in method.parameters],
        ],
        result=lower_type(method.returns, class_env),
    )
    func_expr = L4.Function(
        params=[
            ("this", lower_type(this_type, class_env)),
            *[(name, lower_type(ty, class_env)) for name, ty in method.parameters],
        ],
        body=lowered_body,
    )
    return method_name, func_type, func_expr