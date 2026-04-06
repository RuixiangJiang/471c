from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from functools import partial

from L3 import syntax as L3

from . import syntax as L4

type Context = Mapping[str, L4.Type]
type Symbols = Mapping[str, L4.Type]


class SequentialNameGenerator:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict[str, int](int)

    def __call__(self, candidate: str) -> str:
        current: int = self._counters[candidate]
        self._counters[candidate] += 1
        return f"{candidate}{current}"


def resolve_symbol(sym: L4.Symbol, symbols: Symbols) -> L4.Type:
    visited: set[L4.VName] = set()
    res = sym
    while isinstance(res, L4.Symbol):
        if res.name in visited:
            raise ValueError("Symbol table referencing is circular")
        visited.add(res.name)
        if res.name in symbols:
            res = symbols[res.name]
        else:
            res = res.payload
    return res


def resolve_type(type: L4.Type, symbols: Symbols) -> L4.Type:
    if isinstance(type, L4.Symbol):
        return resolve_symbol(type, symbols=symbols)
    return type


def assert_type_equality(type1: L4.Type, type2: L4.Type, symbols: Symbols) -> None:
    t1 = resolve_type(type1, symbols=symbols)
    t2 = resolve_type(type2, symbols=symbols)
    if t1 != t2:
        raise ValueError(f"Type mismatch between {t1} and {t2}")


def process_types(
    id: str, type: L4.Type, expression: L4.Expression, context: Context, symbols: Symbols, fresh: Callable[[str], str]
) -> tuple[str, L3.Term]:
    _process_ex = partial(process_expression, context=context, symbols=symbols, fresh=fresh)
    match type:
        case L4.Mutable():
            if isinstance(expression, L4.HeapAllocate):
                return (id, _process_ex(expression=expression))
            else:
                mut_id = fresh("mutable")
                val_id = fresh("mutableval")
                return (
                    id,
                    L3.Let(
                        bindings=[(val_id, _process_ex(expression=expression)), (mut_id, L3.Allocate(count=1))],
                        body=L3.Begin(
                            effects=[
                                L3.Store(base=L3.Reference(name=mut_id), index=0, value=L3.Reference(name=val_id))
                            ],
                            value=L3.Reference(name=mut_id),
                        ),
                    ),
                )
        case L4.Symbol(name=name, payload=_):
            resolved = resolve_symbol(type, symbols=symbols)
            local = {name: resolved}
            return process_types(
                id=id, type=resolved, expression=expression, context=context, symbols={**symbols, **local}, fresh=fresh
            )
        case _:
            return (id, _process_ex(expression=expression))


def process_expression(
    expression: L4.Expression, context: Context, symbols: Symbols, fresh: Callable[[str], str]
) -> L3.Term:
    _process = partial(process_expression, context=context, symbols=symbols, fresh=fresh)
    match expression:
        case L4.LetRec(bindings=bindings, body=body):
            local = {name: ty for name, ty, _ in bindings}
            local_sym = {
                ty.name: resolve_symbol(ty, symbols=symbols) for _, ty, _ in bindings if isinstance(ty, L4.Symbol)
            }
            l3_bindings = [
                process_types(
                    id=ide,
                    type=ty,
                    expression=ex,
                    context={**context, **local},
                    symbols={**symbols, **local_sym},
                    fresh=fresh,
                )
                for ide, ty, ex in bindings
            ]
            return L3.LetRec(
                bindings=l3_bindings,
                body=process_expression(
                    expression=body, context={**context, **local}, symbols={**symbols, **local_sym}, fresh=fresh
                ),
            )
        case L4.Let(bindings=bindings, body=body):
            local = {name: ty for name, ty, _ in bindings}
            local_sym = {
                ty.name: resolve_symbol(ty, symbols=symbols) for _, ty, _ in bindings if isinstance(ty, L4.Symbol)
            }
            l3_bindings = [
                process_types(
                    id=ide,
                    type=ty,
                    expression=ex,
                    context=context,
                    symbols={**symbols, **local_sym},
                    fresh=fresh,
                )
                for ide, ty, ex in bindings
            ]
            return L3.Let(
                bindings=l3_bindings,
                body=process_expression(
                    expression=body, context={**context, **local}, symbols={**symbols, **local_sym}, fresh=fresh
                ),
            )
        case L4.Operation(operator=operator, left=left, right=right):
            if operator in ("==", "<"):
                return L3.Branch(
                    operator=operator,
                    left=_process(left),
                    right=_process(right),
                    consequent=L3.Immediate(value=1),
                    otherwise=L3.Immediate(value=0),
                )
            return L3.Primitive(operator=operator, left=_process(left), right=_process(right))
        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            return L3.Branch(
                operator="==",
                left=L3.Immediate(value=1),
                right=_process(condition),
                consequent=_process(consequent),
                otherwise=_process(otherwise),
            )
        case L4.Empty():
            return L3.Immediate(value=0)
        case L4.Immediate(value=value):
            if value is None:
                return L3.Immediate(value=0)
            if isinstance(value, bool):
                return L3.Immediate(value=1 if value else 0)
            return L3.Immediate(value=value)
        case L4.Function(params=params, body=body):
            local = {ide: ty for ide, ty in params}
            l3_params = [ide for ide, _ in params]
            return L3.Abstract(
                parameters=l3_params,
                body=process_expression(expression=body, context={**context, **local}, symbols=symbols, fresh=fresh),
            )
        case L4.Reference(name=name):
            return L3.Reference(name=name)
        case L4.Call(target=target, arguments=arguments):
            return L3.Apply(target=_process(target), arguments=[_process(ex) for ex in arguments])
        case L4.HeapAllocate(val=val):
            id = fresh("heapallocate")
            val_id = fresh("heapallocateval")
            return L3.Let(
                bindings=[(val_id, _process(val)), (id, L3.Allocate(count=1))],
                body=L3.Begin(
                    effects=[L3.Store(base=L3.Reference(name=id), index=0, value=L3.Reference(name=val_id))],
                    value=L3.Reference(name=id),
                ),
            )
        case L4.Get(target=target, index=index):
            ret = _process(target)
            type_ = context[target.name]
            while isinstance(type_, (L4.Mutable, L4.Symbol)):
                if isinstance(type_, L4.Symbol):
                    type_ = resolve_symbol(type_, symbols=symbols)
                else:
                    ret = L3.Load(base=ret, index=0)
                    type_ = type_.oftype
            if isinstance(type_, L4.List) or isinstance(type_, L4.Pair):
                return L3.Load(base=ret, index=index)
            return ret
        case L4.Set(target=target, index=index, value=value):
            ret = _process(target)
            type_ = context[target.name]
            value_ = _process(value)
            while isinstance(type_, (L4.Mutable, L4.Symbol)):
                if isinstance(type_, L4.Symbol):
                    type_ = resolve_symbol(type_, symbols=symbols)
                else:
                    t = type_.oftype
                    if isinstance(t, L4.Symbol):
                        t = resolve_symbol(t, symbols=symbols)
                    if isinstance(t, L4.Mutable):
                        type_ = t
                        ret = L3.Load(base=ret, index=0)
                    elif isinstance(t, L4.List) or isinstance(t, L4.Pair):
                        return L3.Store(base=L3.Load(base=ret, index=0), index=index, value=value_)
                    else:
                        return L3.Store(base=ret, index=0, value=value_)
            raise ValueError("Trying to mutate an immutable in set")
        case L4.NewList(typeof=typeof, size=size):
            id = fresh("list")
            type_default_initial = L3.Immediate(value=0)
            return L3.Let(
                bindings=[(id, L3.Allocate(count=size))],
                body=L3.Begin(
                    effects=[
                        L3.Store(base=L3.Reference(name=id), index=i, value=type_default_initial) for i in range(size)
                    ],
                    value=L3.Reference(name=id),
                ),
            )
        case L4.NewPair(val1=val1, val2=val2, typeof=typeof):
            id = fresh("pair")
            type_ = resolve_type(typeof, symbols=symbols)
            if not isinstance(type_, L4.Pair):
                raise ValueError("NewPair type must be of type Pair")
            return L3.Let(
                bindings=[(id, L3.Allocate(count=2))],
                body=L3.Begin(
                    effects=[
                        L3.Store(base=L3.Reference(name=id), index=0, value=_process(val1)),
                        L3.Store(base=L3.Reference(name=id), index=1, value=_process(val2)),
                    ],
                    value=L3.Reference(name=id),
                ),
            )
        case L4.Capsule(expression=capsule_expression):
            return _process(expression=capsule_expression)
        case L4.While(condition=condition, run=run):
            id = fresh("while")
            run_if = L4.If(
                condition=condition,
                consequent=L4.Bunch(expressions=[run, L4.Call(target=L4.Reference(name=id), arguments=[])]),
                otherwise=L4.Empty(),
            )
            while_loop = L4.LetRec(
                bindings=[(id, L4.FuncType(parameters=[], result=L4.Void()), L4.Function(params=[], body=run_if))],
                body=L4.Call(target=L4.Reference(name=id), arguments=[]),
            )
            return _process(expression=while_loop)
        case L4.For(times=times, run=run):
            if isinstance(times, int):
                times = L4.Immediate(value=times)
            id = fresh("for")
            id_ctr = fresh("for_counter")
            check_times = L4.Operation(
                operator="<", left=L4.Immediate(value=0), right=L4.Get(target=L4.Reference(name=id_ctr), index=0)
            )
            decrement = L4.Set(
                target=L4.Reference(name=id_ctr),
                index=0,
                value=L4.Operation(
                    operator="-", left=L4.Get(target=L4.Reference(name=id_ctr), index=0), right=L4.Immediate(value=1)
                ),
            )
            bunch = L4.Bunch(expressions=[decrement, run, L4.Call(target=L4.Reference(name=id), arguments=[])])
            run_if = L4.If(condition=check_times, consequent=bunch, otherwise=L4.Empty())
            for_loop = L4.LetRec(
                bindings=[
                    (id_ctr, L4.Mutable(oftype=L4.Int()), times),
                    (id, L4.FuncType(parameters=[], result=L4.Void()), L4.Function(params=[], body=run_if)),
                ],
                body=L4.Call(target=L4.Reference(name=id), arguments=[]),
            )
            return _process(expression=for_loop)
        case L4.Bunch(expressions=expressions):
            if len(expressions) == 0:
                return _process(L4.Empty())
            return L3.Begin(effects=[_process(ex) for ex in expressions[:-1]], value=_process(expressions[-1]))
        case _:  # pragma: no cover
            return


def convert_to_l3(
    program: L4.Program,
) -> L3.Program:
    check_program(program=program)
    fresh = SequentialNameGenerator()
    match program:
        case L4.Program(definitions=definitions, body=body):  # pragma: no branch
            context = {name: ty for name, ty, _ in definitions}
            lazy_sym = {ty.name: ty.payload for _, ty, _ in definitions if isinstance(ty, L4.Symbol)}
            symbols = {ty.name: resolve_symbol(ty, lazy_sym) for _, ty, _ in definitions if isinstance(ty, L4.Symbol)}
            l3_bindings = [
                process_types(id=ide, type=ty, expression=ex, context=context, symbols=symbols, fresh=fresh)
                for ide, ty, ex in definitions
            ]
            return L3.Program(
                parameters=[],
                body=L3.Let(
                    bindings=l3_bindings,
                    body=process_expression(expression=body, context=context, symbols=symbols, fresh=fresh),
                ),
            )


def dummy_parse(code: str) -> L4.Program:
    return L4.Program(definitions=[(code, L4.Void(), L4.Empty)], body=L4.Empty())


def check_expression(
    expression: L4.Expression,
    context: Context,
    symbols: Symbols,
) -> L4.Type:
    recur = partial(check_expression, context=context, symbols=symbols)
    type_equal = partial(assert_type_equality, symbols=symbols)

    match expression:
        case L4.Let(bindings=bindings, body=body):
            counts = Counter(name for name, _, _ in bindings)
            duplicates = {name: count for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate binders: {duplicates}")
            for _, ty, ex in bindings:
                type_equal(ty, recur(ex))
            local = {name: ty for name, ty, _ in bindings}
            local_sym = {
                ty.name: resolve_symbol(ty, symbols=symbols) for _, ty, _ in bindings if isinstance(ty, L4.Symbol)
            }
            return recur(body, context={**context, **local}, symbols={**symbols, **local_sym})

        case L4.LetRec(bindings=bindings, body=body):
            counts = Counter(name for name, _, _ in bindings)
            duplicates = {name: count for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate binders: {duplicates}")
            local = {name: ty for name, ty, _ in bindings}
            local_sym = {
                ty.name: resolve_symbol(ty, symbols=symbols) for _, ty, _ in bindings if isinstance(ty, L4.Symbol)
            }
            for _, ty, ex in bindings:
                assert_type_equality(
                    ty,
                    recur(ex, context={**context, **local}, symbols={**symbols, **local_sym}),
                    symbols={**symbols, **local_sym},
                )
            return recur(body, context={**context, **local}, symbols={**symbols, **local_sym})

        case L4.Reference(name=name):
            if name not in context:
                raise ValueError(f"unknown variable: {name}")
            return context[name]

        case L4.Immediate(value=value):
            if isinstance(value, bool):
                return L4.Bool()
            elif value is None:
                return L4.Void()
            return L4.Int()
        case L4.Operation(operator=operator, left=left, right=right):
            type_ = recur(left)
            type_equal(type_, recur(right))
            if operator in ("==", "<"):
                return L4.Bool()
            return type_
        case L4.If(condition=condition, consequent=consequent, otherwise=otherwise):
            type_equal(L4.Bool(), recur(condition))
            type_ = recur(consequent)
            type_equal(type_, recur(otherwise))
            return type_
        case L4.Empty():
            return L4.Void()
        case L4.Function(params=params, body=body):
            counts = Counter(name for name, _ in params)
            duplicates = {name: count for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate binders: {duplicates}")
            local = {ide: ty for ide, ty in params}
            return L4.FuncType(parameters=[ty for _, ty in params], result=recur(body, context={**context, **local}))
        case L4.Call(target=target, arguments=arguments):
            type_ = resolve_type(recur(target), symbols=symbols)
            if isinstance(type_, L4.FuncType):
                if len(arguments) != len(type_.parameters):
                    raise ValueError("Call argument and parameter length mismatch")
                for arg, par in zip(arguments, type_.parameters):
                    type_equal(par, recur(arg))
                return type_.result
            raise ValueError("Trying to call a non function")
        case L4.HeapAllocate(val=val):
            return L4.Mutable(oftype=recur(val))
        case L4.Get(target=target, index=index):
            if target.name not in context:
                raise ValueError(f"unknown variable: {target.name} in get")
            type_ = context[target.name]
            while isinstance(type_, (L4.Mutable, L4.Symbol)):
                if isinstance(type_, L4.Mutable):
                    type_ = type_.oftype
                else:
                    type_ = resolve_symbol(type_, symbols=symbols)
            if isinstance(type_, L4.List):
                return type_.typeof
            if isinstance(type_, L4.Pair):
                if index > 1:
                    raise ValueError("Pair value index must be 0 or 1 in get")
                return type_.type1 if index == 0 else type_.type2
            if index != 0:
                raise ValueError("Scalar value index must be 0 in get")
            return type_
        case L4.Set(target=target, index=index, value=value):
            if target.name not in context:
                raise ValueError(f"unknown variable: {target.name} in set")
            type_ = context[target.name]
            while isinstance(type_, (L4.Mutable, L4.Symbol)):
                if isinstance(type_, L4.Symbol):
                    type_ = resolve_symbol(type_, symbols=symbols)
                    continue
                t = type_.oftype
                if isinstance(t, L4.Symbol):
                    t = resolve_symbol(t, symbols=symbols)
                if not isinstance(t, L4.Mutable):
                    v = recur(value)
                    if isinstance(t, L4.List):
                        type_equal(v, t.typeof)
                    elif isinstance(t, L4.Pair):
                        if index > 1:
                            raise ValueError("Pair value index must be 0 or 1 in set")
                        type_equal(v, (t.type1 if index == 0 else t.type2))
                    else:
                        type_equal(v, t)
                        if index != 0:
                            raise ValueError("Scalar value index must be 0 in set")
                    return L4.Void()
                type_ = t
            raise ValueError(f"Set trying to mutate an immutable {target}")
        case L4.NewList(typeof=typeof, size=_):
            return L4.List(typeof=typeof)
        case L4.NewPair(val1=val1, val2=val2, typeof=typeof):
            type_ = resolve_type(typeof, symbols=symbols)
            if not isinstance(type_, L4.Pair):
                raise ValueError("NewPair type must be of type Pair")
            type_equal(recur(val1), type_.type1)
            type_equal(recur(val2), type_.type2)
            return typeof
        case L4.Capsule(typeof=typeof, expression=capsule_expression):
            type_equal(recur(capsule_expression), typeof)
            return typeof
        case L4.While(condition=condition, run=run):
            type_equal(recur(condition), L4.Bool())
            type_equal(recur(run), L4.Void())
            return L4.Void()
        case L4.For(times=times, run=run):
            if isinstance(times, int):
                times = L4.Immediate(value=times)
            type_equal(recur(times), L4.Int())
            type_equal(recur(run), L4.Void())
            return L4.Void()
        case L4.Bunch(expressions=expressions):
            if not expressions:
                return L4.Void()
            for ex in expressions[:-1]:
                recur(ex)
            return recur(expressions[-1])
        case _:  # pragma: no cover
            return


def check_program(program: L4.Program) -> None:
    match program:
        case L4.Program(definitions=definitions, body=body):  # pragma: no branch
            counts = Counter(name for name, _, _ in definitions)
            duplicates = {name for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate parameters: {duplicates}")
            local = {name: ty for name, ty, _ in definitions}
            lazy_sym = {ty.name: ty.payload for _, ty, _ in definitions if isinstance(ty, L4.Symbol)}
            local_symbols = {
                ty.name: resolve_symbol(ty, lazy_sym) for _, ty, _ in definitions if isinstance(ty, L4.Symbol)
            }
            for _, ty, ex in definitions:
                assert_type_equality(
                    ty, check_expression(expression=ex, context=local, symbols=local_symbols), symbols=local_symbols
                )
            check_expression(body, context=local, symbols=local_symbols)
