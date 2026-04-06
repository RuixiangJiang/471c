from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from L3 import syntax as L3
from L4 import syntax as L4
from L4.convert import convert_to_l3 as convert_l4_to_l3

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


@dataclass(frozen=True)
class FieldInfo:
    name: str
    typeof: L5.Type


@dataclass(frozen=True)
class MethodInfo:
    name: str
    parameters: Sequence[tuple[str, L5.Type]]
    returns: L5.Type
    body: L5.Expression


@dataclass(frozen=True)
class ClassInfo:
    name: str
    fields: Sequence[FieldInfo]
    methods: Mapping[str, MethodInfo]


def dummy_parse(code: str) -> L5.Program:
    return L5.Program(
        classes=[],
        definitions=[(code, L4.Void(), L4.Empty())],
        body=L4.Empty(),
    )


def _bool(value: bool) -> L4.Immediate:
    return L4.Immediate(value=value)


def _same_type(left: L5.Type, right: L5.Type) -> bool:
    return left == right


def _type_error(message: str) -> TypeError:
    return TypeError(message)


def _field_index(info: ClassInfo, field: str) -> int:
    for i, f in enumerate(info.fields):
        if f.name == field:
            return i
    raise _type_error(f"class {info.name!r} has no field {field!r}")


def _field_info(info: ClassInfo, field: str) -> FieldInfo:
    for f in info.fields:
        if f.name == field:
            return f
    raise _type_error(f"class {info.name!r} has no field {field!r}")


def _method_info(info: ClassInfo, method: str) -> MethodInfo:
    if method not in info.methods:
        raise _type_error(f"class {info.name!r} has no method {method!r}")
    return info.methods[method]


def _collect_classes(classes: Sequence[L5.ClassDef]) -> dict[str, ClassInfo]:
    env: dict[str, ClassInfo] = {}

    for cls in classes:
        if cls.name in env:
            raise _type_error(f"duplicate class definition for {cls.name!r}")

        seen_fields: set[str] = set()
        fields: list[FieldInfo] = []
        for field in cls.fields:
            if field.name in seen_fields:
                raise _type_error(f"duplicate field {field.name!r} in class {cls.name!r}")
            seen_fields.add(field.name)
            fields.append(FieldInfo(name=field.name, typeof=field.typeof))

        seen_methods: set[str] = set()
        methods: dict[str, MethodInfo] = {}
        for method in cls.methods:
            if method.name in seen_methods:
                raise _type_error(f"duplicate method {method.name!r} in class {cls.name!r}")
            seen_methods.add(method.name)
            methods[method.name] = MethodInfo(
                name=method.name,
                parameters=method.parameters,
                returns=method.returns,
                body=method.body,
            )

        env[cls.name] = ClassInfo(name=cls.name, fields=fields, methods=methods)

    return env


def _class_payload_type(info: ClassInfo, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    lowered = [_lower_type(field.typeof, class_env) for field in info.fields]

    if not lowered:
        return L4.Void()

    current = lowered[-1]
    for item in reversed(lowered[:-1]):
        current = L4.Pair(type1=item, type2=current)
    return current


def _class_rep_type(info: ClassInfo, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    return L4.Mutable(oftype=_class_payload_type(info, class_env))


def _lower_type(typeof: L5.Type, class_env: Mapping[str, ClassInfo]) -> L4.Type:
    match typeof:
        case L5.ClassType(name=name):
            if name not in class_env:
                raise _type_error(f"unknown class type {name!r}")
            return _class_rep_type(class_env[name], class_env)

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
            raise _type_error(f"cannot lower type {typeof!r}")


def _infer_type(
    expression: L5.Expression,
    *,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
) -> L5.Type:
    match expression:
        case L5.This():
            if current_class is None:
                raise _type_error("this used outside of a method body")
            return L5.ClassType(name=current_class)

        case L5.NewObject(name=name, arguments=arguments):
            if name not in class_env:
                raise _type_error(f"unknown class {name!r}")
            info = class_env[name]
            if len(arguments) != len(info.fields):
                raise _type_error(
                    f"new {name} expects {len(info.fields)} arguments, got {len(arguments)}"
                )
            for arg, field in zip(arguments, info.fields, strict=True):
                actual = _infer_type(
                    arg,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not _same_type(actual, field.typeof):
                    raise _type_error(
                        f"constructor argument for field {field.name!r} expected {field.typeof!r}, got {actual!r}"
                    )
            return L5.ClassType(name=name)

        case L5.FieldAccess(target=target, field=field):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    return _field_info(class_env[name], field).typeof
                case _:
                    raise _type_error(f"field access requires class target, got {target_type!r}")

        case L5.FieldAssign(target=target, field=field, value=value):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    expected = _field_info(class_env[name], field).typeof
                case _:
                    raise _type_error(f"field assignment requires class target, got {target_type!r}")

            actual = _infer_type(
                value,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not _same_type(actual, expected):
                raise _type_error(
                    f"field assignment expected {expected!r}, got {actual!r}"
                )
            return L4.Void()

        case L5.MethodCall(target=target, method=method, arguments=arguments):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    method_info = _method_info(class_env[name], method)
                case _:
                    raise _type_error(f"method call requires class target, got {target_type!r}")

            if len(arguments) != len(method_info.parameters):
                raise _type_error(
                    f"method {name}.{method} expects {len(method_info.parameters)} arguments, got {len(arguments)}"
                )

            for arg, (_, expected) in zip(arguments, method_info.parameters, strict=True):
                actual = _infer_type(
                    arg,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not _same_type(actual, expected):
                    raise _type_error(
                        f"method argument expected {expected!r}, got {actual!r}"
                    )
            return method_info.returns

        case L5.ShortCircuit(left=left, right=right):
            left_type = _infer_type(
                left,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            right_type = _infer_type(
                right,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if left_type != L4.Bool() or right_type != L4.Bool():
                raise _type_error("short-circuit operators require bool operands")
            return L4.Bool()

        case L5.Switch(scrutinee=scrutinee, cases=cases, default=default):
            scrutinee_type = _infer_type(
                scrutinee,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if scrutinee_type not in (L4.Int(), L4.Bool()):
                raise _type_error("switch scrutinee must be int or bool")

            default_type = _infer_type(
                default,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            for case in cases:
                case_type = _infer_type(
                    case.body,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not _same_type(case_type, default_type):
                    raise _type_error("all switch branches must have the same type")
            return default_type

        case L5.Break() | L5.Continue():
            return L4.Void()

        case L5.Foreach(binder=binder, typeof=typeof, target=target, run=run):
            target_type = context.get(target.name)
            if not isinstance(target_type, L4.List):
                raise _type_error("foreach target must be a list reference")
            if not _same_type(target_type.typeof, typeof):
                raise _type_error("foreach binder type does not match list element type")
            return _infer_type(
                run,
                context={**context, binder: typeof},
                class_env=class_env,
                current_class=current_class,
            )

        case L4.Reference(name=name):
            if name not in context:
                raise _type_error(f"unknown reference {name!r}")
            return context[name]

        case L4.Immediate(value=value):
            if isinstance(value, bool):
                return L4.Bool()
            if isinstance(value, int):
                return L4.Int()
            return L4.Void()

        case L4.Operation(operator=operator, left=left, right=right):
            left_type = _infer_type(
                left,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            right_type = _infer_type(
                right,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if operator in {"+", "-", "*", "<"}:
                if left_type != L4.Int() or right_type != L4.Int():
                    raise _type_error(f"operator {operator!r} requires int operands")
                return L4.Bool() if operator == "<" else L4.Int()
            if operator == "==":
                if not _same_type(left_type, right_type):
                    raise _type_error("== requires operands of the same type")
                return L4.Bool()
            raise _type_error(f"unknown operator {operator!r}")

        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            cond_type = _infer_type(
                condition,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if cond_type != L4.Bool():
                raise _type_error("if condition must be bool")
            c_type = _infer_type(
                consequent,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            o_type = _infer_type(
                otherwise,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not _same_type(c_type, o_type):
                raise _type_error("if branches must have the same type")
            return c_type

        case L4.Let(bindings=bindings, body=body):
            next_context = dict(context)
            for name, ty, ex in bindings:
                actual = _infer_type(
                    ex,
                    context=next_context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if not _same_type(actual, ty):
                    raise _type_error(
                        f"let binding {name!r} expected {ty!r}, got {actual!r}"
                    )
                next_context[name] = ty
            return _infer_type(
                body,
                context=next_context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.LetRec(bindings=bindings, body=body):
            next_context = dict(context)
            for name, ty, _ in bindings:
                next_context[name] = ty
            return _infer_type(
                body,
                context=next_context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.Function():
            raise _type_error("cannot infer a bare function expression here without an expected type")

        case L4.Call():
            raise _type_error("method dispatch is supported via MethodCall; bare call inference is not implemented")

        case L4.Empty():
            return L4.Void()

        case L4.NewList(typeof=typeof):
            return L4.List(typeof=typeof)

        case L4.NewPair(typeof=typeof):
            return typeof

        case L4.HeapAllocate(val=val):
            return L4.Mutable(
                oftype=_infer_type(
                    val,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
            )

        case L4.Get(target=target, index=index):
            target_type = context.get(target.name)
            if target_type is None:
                raise _type_error(f"unknown reference {target.name!r}")
            match target_type:
                case L4.Mutable(oftype=oftype):
                    if index != 0:
                        raise _type_error("mutable get only supports index 0")
                    return oftype
                case L4.Pair(type1=t1, type2=t2):
                    if index == 0:
                        return t1
                    if index == 1:
                        return t2
                    raise _type_error("pair get index must be 0 or 1")
                case L4.List(typeof=typeof):
                    return typeof
                case _:
                    raise _type_error(f"get is not supported for type {target_type!r}")

        case L4.Set(target=target, index=index, value=value):
            target_type = context.get(target.name)
            if target_type is None:
                raise _type_error(f"unknown reference {target.name!r}")
            value_type = _infer_type(
                value,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L4.Mutable(oftype=oftype):
                    if index != 0 or not _same_type(value_type, oftype):
                        raise _type_error("invalid mutable set")
                case L4.Pair(type1=t1, type2=t2):
                    expected = t1 if index == 0 else t2
                    if not _same_type(value_type, expected):
                        raise _type_error("invalid pair set")
                case L4.List(typeof=typeof):
                    if not _same_type(value_type, typeof):
                        raise _type_error("invalid list set")
                case _:
                    raise _type_error(f"set is not supported for type {target_type!r}")
            return L4.Void()

        case L4.Capsule(typeof=typeof, expression=inner):
            actual = _infer_type(
                inner,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if not _same_type(actual, typeof):
                raise _type_error(f"capsule expected {typeof!r}, got {actual!r}")
            return typeof

        case L4.While(condition=condition, run=run):
            cond_type = _infer_type(
                condition,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            if cond_type != L4.Bool():
                raise _type_error("while condition must be bool")
            _infer_type(run, context=context, class_env=class_env, current_class=current_class)
            return L4.Void()

        case L4.For(times=times, run=run):
            if isinstance(times, int):
                pass
            else:
                times_type = _infer_type(
                    times,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                )
                if times_type != L4.Int():
                    raise _type_error("for times must be int")
            _infer_type(run, context=context, class_env=class_env, current_class=current_class)
            return L4.Void()

        case L4.Bunch(expressions=expressions):
            if not expressions:
                return L4.Void()
            for ex in expressions[:-1]:
                _infer_type(ex, context=context, class_env=class_env, current_class=current_class)
            return _infer_type(
                expressions[-1],
                context=context,
                class_env=class_env,
                current_class=current_class,
            )

        case _:
            raise _type_error(f"cannot infer type for {expression!r}")


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


def _wrap_let(name: str, ty: L4.Type, ex: L4.Expression, body: L4.Expression) -> L4.Expression:
    return L4.Let(bindings=[(name, ty, ex)], body=body)


def _field_path(field_index: int, field_count: int) -> list[int]:
    if field_count == 0:
        raise _type_error("cannot compute field path for empty class")
    if field_count == 1:
        if field_index != 0:
            raise _type_error("invalid field index")
        return []

    path: list[int] = []
    idx = field_index
    remaining = field_count
    while remaining > 1:
        if idx == 0:
            path.append(0)
            return path
        path.append(1)
        idx -= 1
        remaining -= 1
    path.append(0)
    return path


def _access_from_ref(
    current_ref: L4.Reference,
    current_type: L4.Type,
    path: Sequence[int],
    *,
    fresh: SequentialNameGenerator,
) -> L4.Expression:
    if not path:
        return current_ref

    match current_type:
        case L4.Pair(type1=t1, type2=t2):
            step = path[0]
            child_type = t1 if step == 0 else t2
            tmp = fresh("field")
            return _wrap_let(
                tmp,
                child_type,
                L4.Get(target=current_ref, index=step),
                _access_from_ref(
                    L4.Reference(name=tmp),
                    child_type,
                    path[1:],
                    fresh=fresh,
                ),
            )
        case _:
            raise _type_error(f"cannot descend into non-pair type {current_type!r}")


def _rebuild_from_ref(
    current_ref: L4.Reference,
    current_type: L4.Type,
    path: Sequence[int],
    new_value: L4.Expression,
    *,
    fresh: SequentialNameGenerator,
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
                    new_left = _wrap_let(
                        child_name,
                        t1,
                        L4.Get(target=current_ref, index=0),
                        _rebuild_from_ref(
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
                    new_right = _wrap_let(
                        child_name,
                        t2,
                        L4.Get(target=current_ref, index=1),
                        _rebuild_from_ref(
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

            raise _type_error("pair path step must be 0 or 1")

        case _:
            raise _type_error(f"cannot rebuild into non-pair type {current_type!r}")


def _wrap_with_reference(
    lowered: L4.Expression,
    typeof: L4.Type,
    *,
    fresh: SequentialNameGenerator,
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


def _lower_short_circuit(
    expr: L5.ShortCircuit,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    loop: LoopContext | None,
) -> L4.Expression:
    left = _lower_expression(
        expr.left,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )
    right = _lower_expression(
        expr.right,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )

    if expr.operator == "&&":
        return L4.If(condition=left, consequent=right, otherwise=_bool(False))
    if expr.operator == "||":
        return L4.If(condition=left, consequent=_bool(True), otherwise=right)
    raise ValueError(f"unknown short-circuit operator: {expr.operator}")


def _lower_switch(
    expr: L5.Switch,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    loop: LoopContext | None,
) -> L4.Expression:
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
        class_env=class_env,
        current_class=current_class,
        loop=loop,
    )

    for case in reversed(expr.cases):
        body = _lower_expression(
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

    lowered_scrutinee = _lower_expression(
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


def _lower_bunch(
    expressions: Sequence[L5.Expression],
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    loop: LoopContext | None,
) -> L4.Expression:
    if not expressions:
        return L4.Empty()

    lowered: list[L4.Expression] = []
    for i, ex in enumerate(expressions):
        item = _lower_expression(
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
            lowered.append(_guarded_void(item, loop=loop))
    return L4.Bunch(expressions=lowered)


def _lower_while(
    condition: L5.Expression,
    run: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
) -> L4.Expression:
    loop_name = fresh("while")
    break_flag = fresh("break")
    continue_flag = fresh("continue")

    inner_loop = LoopContext(break_flag=break_flag, continue_flag=continue_flag)

    lowered_condition = _lower_expression(
        condition,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )
    lowered_run = _lower_expression(
        run,
        fresh=fresh,
        context=context,
        class_env=class_env,
        current_class=current_class,
        loop=inner_loop,
    )

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
        class_env=class_env,
        current_class=current_class,
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
    times: int | L5.Expression,
    run: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
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
        lowered_times = _lower_expression(
            times,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            loop=None,
        )

    lowered_run = _lower_expression(
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
        class_env=class_env,
        current_class=current_class,
        loop=None,
    )

    return L4.LetRec(
        bindings=[
            (counter_name, L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=lowered_times)),
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
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
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
        iteration_body = _lower_expression(
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
                    value=_bool(False),
                ),
                L4.Let(
                    bindings=[
                        (
                            expr.binder,
                            _lower_type(expr.typeof, class_env),
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
            (break_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
            (continue_flag, L4.Mutable(oftype=L4.Bool()), L4.HeapAllocate(val=_bool(False))),
        ],
        body=foreach_body,
    )


def _build_payload_expression(
    info: ClassInfo,
    arguments: Sequence[L4.Expression],
    *,
    class_env: Mapping[str, ClassInfo],
) -> L4.Expression:
    if len(arguments) != len(info.fields):
        raise _type_error("constructor argument count mismatch")

    if not arguments:
        return L4.Empty()

    current = arguments[-1]
    lowered_types = [_lower_type(field.typeof, class_env) for field in info.fields]
    current_type = lowered_types[-1]

    for ex, ty in zip(reversed(arguments[:-1]), reversed(lowered_types[:-1]), strict=True):
        current = L4.NewPair(
            val1=ex,
            val2=current,
            typeof=L4.Pair(type1=ty, type2=current_type),
        )
        current_type = L4.Pair(type1=ty, type2=current_type)
    return current


def _lower_expression(
    expression: L5.Expression,
    *,
    fresh: SequentialNameGenerator,
    context: Context,
    class_env: Mapping[str, ClassInfo],
    current_class: str | None,
    loop: LoopContext | None,
) -> L4.Expression:
    match expression:
        case L5.This():
            if current_class is None:
                raise ValueError("this used outside of a method body")
            return L4.Reference(name="this")

        case L5.NewObject(name=name, arguments=arguments):
            if name not in class_env:
                raise _type_error(f"unknown class {name!r}")
            info = class_env[name]

            lowered_args = [
                _lower_expression(
                    arg,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                )
                for arg in arguments
            ]
            payload = _build_payload_expression(info, lowered_args, class_env=class_env)
            return L4.HeapAllocate(val=payload)

        case L5.FieldAccess(target=target, field=field):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    info = class_env[name]
                case _:
                    raise _type_error(f"field access requires class target, got {target_type!r}")

            lowered_target = _lower_expression(
                target,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=None,
            )
            target_l4_type = _lower_type(target_type, class_env)
            payload_type = _class_payload_type(info, class_env)
            path = _field_path(_field_index(info, field), len(info.fields))

            def body_builder(obj_ref: L4.Reference) -> L4.Expression:
                root_name = fresh("root")
                return L4.Let(
                    bindings=[(root_name, payload_type, L4.Get(target=obj_ref, index=0))],
                    body=_access_from_ref(
                        L4.Reference(name=root_name),
                        payload_type,
                        path,
                        fresh=fresh,
                    ),
                )

            return _wrap_with_reference(
                lowered_target,
                target_l4_type,
                fresh=fresh,
                prefix="obj",
                body_builder=body_builder,
            )

        case L5.FieldAssign(target=target, field=field, value=value):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    info = class_env[name]
                case _:
                    raise _type_error(f"field assignment requires class target, got {target_type!r}")

            lowered_target = _lower_expression(
                target,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=None,
            )
            lowered_value = _lower_expression(
                value,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=None,
            )

            payload_type = _class_payload_type(info, class_env)
            path = _field_path(_field_index(info, field), len(info.fields))
            target_l4_type = _lower_type(target_type, class_env)

            def body_builder(obj_ref: L4.Reference) -> L4.Expression:
                root_name = fresh("root")
                root_ref = L4.Reference(name=root_name)
                rebuilt = _rebuild_from_ref(
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

            return _wrap_with_reference(
                lowered_target,
                target_l4_type,
                fresh=fresh,
                prefix="obj",
                body_builder=body_builder,
            )

        case L5.MethodCall(target=target, method=method, arguments=arguments):
            target_type = _infer_type(
                target,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )
            match target_type:
                case L5.ClassType(name=name):
                    _method_info(class_env[name], method)
                    method_name = f"{name}_{method}"
                case _:
                    raise _type_error(f"method call requires class target, got {target_type!r}")

            lowered_target = _lower_expression(
                target,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=None,
            )
            lowered_args = [
                _lower_expression(
                    arg,
                    fresh=fresh,
                    context=context,
                    class_env=class_env,
                    current_class=current_class,
                    loop=None,
                )
                for arg in arguments
            ]
            target_l4_type = _lower_type(target_type, class_env)

            return _wrap_with_reference(
                lowered_target,
                target_l4_type,
                fresh=fresh,
                prefix="obj",
                body_builder=lambda obj_ref: L4.Call(
                    target=L4.Reference(name=method_name),
                    arguments=[obj_ref, *lowered_args],
                ),
            )

        case L5.ShortCircuit():
            return _lower_short_circuit(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
            )

        case L5.Switch():
            return _lower_switch(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
            )

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
            return _lower_foreach(
                expression,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.While(condition=condition, run=run):
            return _lower_while(
                condition,
                run,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.For(times=times, run=run):
            return _lower_for(
                times,
                run,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
            )

        case L4.Bunch(expressions=expressions):
            return _lower_bunch(
                expressions,
                fresh=fresh,
                context=context,
                class_env=class_env,
                current_class=current_class,
                loop=loop,
            )

        case L4.Let(bindings=bindings, body=body):
            lowered_bindings = [
                (
                    name,
                    _lower_type(ty, class_env),
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
                    _lower_type(ty, class_env),
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
                params=[(name, _lower_type(ty, class_env)) for name, ty in params],
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
                typeof=_lower_type(typeof, class_env),
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
                typeof=_lower_type(typeof, class_env),
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


def _lower_method_definition(
    class_name: str,
    method: MethodInfo,
    *,
    fresh: SequentialNameGenerator,
    class_env: Mapping[str, ClassInfo],
) -> tuple[str, L4.Type, L4.Expression]:
    info = class_env[class_name]
    this_type = L5.ClassType(name=class_name)
    context: dict[str, L5.Type] = {"this": this_type}
    for name, ty in method.parameters:
        context[name] = ty

    actual_returns = _infer_type(
        method.body,
        context=context,
        class_env=class_env,
        current_class=class_name,
    )
    if not _same_type(actual_returns, method.returns):
        raise _type_error(
            f"method {class_name}.{method.name} expected return {method.returns!r}, got {actual_returns!r}"
        )

    lowered_body = _lower_expression(
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
            _lower_type(this_type, class_env),
            *[_lower_type(ty, class_env) for _, ty in method.parameters],
        ],
        result=_lower_type(method.returns, class_env),
    )
    func_expr = L4.Function(
        params=[
            ("this", _lower_type(this_type, class_env)),
            *[(name, _lower_type(ty, class_env)) for name, ty in method.parameters],
        ],
        body=lowered_body,
    )
    return method_name, func_type, func_expr


def convert_to_l4(program: L5.Program) -> L4.Program:
    fresh = SequentialNameGenerator()
    class_env = _collect_classes(program.classes)

    lowered_defs: list[tuple[str, L4.Type, L4.Expression]] = []

    generated_names: set[str] = set()
    for cls in program.classes:
        for method in cls.methods:
            lowered = _lower_method_definition(
                cls.name,
                MethodInfo(
                    name=method.name,
                    parameters=method.parameters,
                    returns=method.returns,
                    body=method.body,
                ),
                fresh=fresh,
                class_env=class_env,
            )
            if lowered[0] in generated_names:
                raise _type_error(f"duplicate generated method name {lowered[0]!r}")
            generated_names.add(lowered[0])
            lowered_defs.append(lowered)

    context: dict[str, L5.Type] = {name: ty for name, ty, _ in program.definitions}
    for name, _, _ in lowered_defs:
        # generated methods are only meant to be called through MethodCall lowering
        context[name] = L4.Void()

    for name, ty, ex in program.definitions:
        actual = _infer_type(
            ex,
            context=context,
            class_env=class_env,
            current_class=None,
        )
        if not _same_type(actual, ty):
            raise _type_error(f"definition {name!r} expected {ty!r}, got {actual!r}")

        lowered_defs.append(
            (
                name,
                _lower_type(ty, class_env),
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