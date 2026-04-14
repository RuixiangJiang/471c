from __future__ import annotations

from collections.abc import Iterable, Sequence
import importlib

from .syntax import (
    Abstract,
    Allocate,
    Apply,
    Branch,
    Copy,
    Halt,
    Immediate,
    Load,
    Primitive,
    Program,
    Statement,
    Store,
)

L0 = importlib.import_module("L0.syntax")


class _FreshNames:
    def __init__(self) -> None:
        self._counter = 0

    def __call__(self, hint: str) -> str:
        value = f"{hint}$close{self._counter}"
        self._counter += 1
        return value


class _LiftedProcedures:
    def __init__(self) -> None:
        self.items: list[L0.Procedure] = []

    def append(self, value: L0.Procedure) -> None:
        self.items.append(value)

    def extend(self, values: Iterable[L0.Procedure]) -> None:
        self.items.extend(values)


def _copy(destination: str, source: str, then: L0.Statement) -> L0.Copy:
    return L0.Copy(destination=destination, source=source, then=then)


def _immediate(destination: str, value: int, then: L0.Statement) -> L0.Immediate:
    return L0.Immediate(destination=destination, value=value, then=then)


def _primitive(destination: str, operator: str, left: str, right: str, then: L0.Statement) -> L0.Primitive:
    return L0.Primitive(
        destination=destination,
        operator=operator,
        left=left,
        right=right,
        then=then,
    )


def _branch(operator: str, left: str, right: str, then: L0.Statement, otherwise: L0.Statement) -> L0.Branch:
    return L0.Branch(
        operator=operator,
        left=left,
        right=right,
        then=then,
        otherwise=otherwise,
    )


def _allocate(destination: str, count: int, then: L0.Statement) -> L0.Allocate:
    return L0.Allocate(destination=destination, count=count, then=then)


def _load(destination: str, base: str, index: int, then: L0.Statement) -> L0.Load:
    return L0.Load(destination=destination, base=base, index=index, then=then)


def _store(base: str, index: int, value: str, then: L0.Statement) -> L0.Store:
    return L0.Store(base=base, index=index, value=value, then=then)


def _address(destination: str, name: str, then: L0.Statement) -> L0.Address:
    return L0.Address(destination=destination, name=name, then=then)


def _call(target: str, arguments: Sequence[str]) -> L0.Call:
    return L0.Call(target=target, arguments=arguments)


def _halt(value: str) -> L0.Halt:
    return L0.Halt(value=value)


def _procedure(name: str, parameters: Sequence[str], body: L0.Statement) -> L0.Procedure:
    return L0.Procedure(name=name, parameters=parameters, body=body)


def _program(procedures: Sequence[L0.Procedure]) -> L0.Program:
    return L0.Program(procedures=procedures)


def _merge(left: Iterable[str], right: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []

    for value in [*left, *right]:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return tuple(result)


def _without(values: Iterable[str], removed: Iterable[str]) -> tuple[str, ...]:
    blocked = set(removed)
    return tuple(value for value in values if value not in blocked)


def free_variables(statement: Statement) -> tuple[str, ...]:
    match statement:
        case Copy(destination=destination, source=source, then=then):
            return _merge((source,), _without(free_variables(then), (destination,)))

        case Abstract(destination=destination, parameters=parameters, body=body, then=then):
            body_free = _without(free_variables(body), (destination, *parameters))
            then_free = _without(free_variables(then), (destination,))
            return _merge(body_free, then_free)

        case Apply(target=target, arguments=arguments):
            return _merge((target,), arguments)

        case Immediate(destination=destination, value=_, then=then):
            return _without(free_variables(then), (destination,))

        case Primitive(destination=destination, operator=_, left=left, right=right, then=then):
            return _merge((left, right), _without(free_variables(then), (destination,)))

        case Branch(operator=_, left=left, right=right, then=then, otherwise=otherwise):
            return _merge((left, right), _merge(free_variables(then), free_variables(otherwise)))

        case Allocate(destination=destination, count=_, then=then):
            return _without(free_variables(then), (destination,))

        case Load(destination=destination, base=base, index=_, then=then):
            return _merge((base,), _without(free_variables(then), (destination,)))

        case Store(base=base, index=_, value=value, then=then):
            return _merge((base, value), free_variables(then))

        case Halt(value=value):
            return (value,)

        case _:
            raise TypeError(f"Unhandled L1 statement in free_variables: {statement!r}")


def _prepend_capture_loads(statement: L0.Statement, closure_parameter: str, captures: Sequence[str]) -> L0.Statement:
    result = statement

    for index, capture in reversed(list(enumerate(captures, start=1))):
        result = _load(destination=capture, base=closure_parameter, index=index, then=result)

    return result


def _make_closure(
    destination: str,
    procedure_name: str,
    captures: Sequence[str],
    then: L0.Statement,
    fresh: _FreshNames,
) -> L0.Statement:
    address_name = fresh(f"{destination}$addr")
    result = then

    for index, capture in reversed(list(enumerate(captures, start=1))):
        result = _store(base=destination, index=index, value=capture, then=result)

    result = _store(base=destination, index=0, value=address_name, then=result)
    result = _address(destination=address_name, name=procedure_name, then=result)
    result = _allocate(destination=destination, count=len(captures) + 1, then=result)
    return result


def _close_statement(statement: Statement, lifted: _LiftedProcedures, fresh: _FreshNames) -> L0.Statement:
    match statement:
        case Copy(destination=destination, source=source, then=then):
            return _copy(destination=destination, source=source, then=_close_statement(then, lifted, fresh))

        case Abstract(destination=destination, parameters=parameters, body=body, then=then):
            captures = _without(free_variables(body), (destination, *parameters))
            procedure_name = fresh(destination)
            closure_parameter = fresh(f"{destination}$env")

            nested = _LiftedProcedures()
            closed_body = _close_statement(body, nested, fresh)
            closed_body = _prepend_capture_loads(closed_body, closure_parameter, captures)
            closed_body = _copy(destination=destination, source=closure_parameter, then=closed_body)

            lifted.append(
                _procedure(
                    name=procedure_name,
                    parameters=(closure_parameter, *parameters),
                    body=closed_body,
                )
            )
            lifted.extend(nested.items)

            return _make_closure(
                destination=destination,
                procedure_name=procedure_name,
                captures=captures,
                then=_close_statement(then, lifted, fresh),
                fresh=fresh,
            )

        case Apply(target=target, arguments=arguments):
            code_pointer = fresh(f"{target}$code")
            return _load(
                destination=code_pointer,
                base=target,
                index=0,
                then=_call(target=code_pointer, arguments=(target, *arguments)),
            )

        case Immediate(destination=destination, value=value, then=then):
            return _immediate(destination=destination, value=value, then=_close_statement(then, lifted, fresh))

        case Primitive(destination=destination, operator=operator, left=left, right=right, then=then):
            return _primitive(
                destination=destination,
                operator=operator,
                left=left,
                right=right,
                then=_close_statement(then, lifted, fresh),
            )

        case Branch(operator=operator, left=left, right=right, then=then, otherwise=otherwise):
            return _branch(
                operator=operator,
                left=left,
                right=right,
                then=_close_statement(then, lifted, fresh),
                otherwise=_close_statement(otherwise, lifted, fresh),
            )

        case Allocate(destination=destination, count=count, then=then):
            return _allocate(destination=destination, count=count, then=_close_statement(then, lifted, fresh))

        case Load(destination=destination, base=base, index=index, then=then):
            return _load(destination=destination, base=base, index=index, then=_close_statement(then, lifted, fresh))

        case Store(base=base, index=index, value=value, then=then):
            return _store(base=base, index=index, value=value, then=_close_statement(then, lifted, fresh))

        case Halt(value=value):
            return _halt(value=value)

        case _:
            raise TypeError(f"Unhandled L1 statement in close conversion: {statement!r}")


def close_program(program: Program) -> L0.Program:
    match program:
        case Program(parameters=parameters, body=body):
            fresh = _FreshNames()
            lifted = _LiftedProcedures()
            closed_body = _close_statement(body, lifted, fresh)
            lifted.append(_procedure(name="l0", parameters=parameters, body=closed_body))
            return _program(procedures=tuple(lifted.items))

        case _:
            raise TypeError(f"Unhandled L1 program in close_program: {program!r}")


close = close_program