from __future__ import annotations

from collections.abc import Mapping

from .syntax import (
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
    Term,
)


Constant = Immediate | Reference


def _is_constant(term: Term) -> bool:
    return isinstance(term, (Immediate, Reference))


def _has_effect(term: Term) -> bool:
    match term:
        case Immediate() | Reference() | Abstract() | Allocate():
            return False

        case Primitive(left=left, right=right):
            return _has_effect(left) or _has_effect(right)

        case Branch(left=left, right=right, consequent=consequent, otherwise=otherwise):
            return (
                _has_effect(left)
                or _has_effect(right)
                or _has_effect(consequent)
                or _has_effect(otherwise)
            )

        case Load(base=base):
            return _has_effect(base)

        case Let(bindings=bindings, body=body):
            return any(_has_effect(value) for _, value in bindings) or _has_effect(body)

        case Begin(effects=effects, value=value):
            return any(_has_effect(effect) for effect in effects) or _has_effect(value) or bool(effects)

        case Apply() | Store():
            return True

    raise TypeError(f"unsupported term: {term!r}")


def _free_variables(term: Term) -> set[str]:
    match term:
        case Immediate() | Allocate():
            return set()

        case Reference(name=name):
            return {name}

        case Abstract(parameters=parameters, body=body):
            return _free_variables(body) - set(parameters)

        case Apply(target=target, arguments=arguments):
            used = _free_variables(target)
            for argument in arguments:
                used |= _free_variables(argument)
            return used

        case Primitive(left=left, right=right):
            return _free_variables(left) | _free_variables(right)

        case Branch(left=left, right=right, consequent=consequent, otherwise=otherwise):
            return (
                _free_variables(left)
                | _free_variables(right)
                | _free_variables(consequent)
                | _free_variables(otherwise)
            )

        case Load(base=base):
            return _free_variables(base)

        case Store(base=base, value=value):
            return _free_variables(base) | _free_variables(value)

        case Begin(effects=effects, value=value):
            used = _free_variables(value)
            for effect in effects:
                used |= _free_variables(effect)
            return used

        case Let(bindings=bindings, body=body):
            used = _free_variables(body)
            for name, value in reversed(bindings):
                used.discard(name)
                used |= _free_variables(value)
            return used

    raise TypeError(f"unsupported term: {term!r}")


def _extend_env(env: Mapping[str, Constant], name: str, value: Term) -> dict[str, Constant]:
    next_env = dict(env)
    if _is_constant(value):
        next_env[name] = value
    else:
        next_env.pop(name, None)
    return next_env


def _flatten_begin(effects: list[Term], value: Term) -> Term:
    flat_effects: list[Term] = []
    for effect in effects:
        if isinstance(effect, Begin):
            flat_effects.extend(effect.effects)
            flat_effects.append(effect.value)
        else:
            flat_effects.append(effect)

    if isinstance(value, Begin):
        flat_effects.extend(value.effects)
        value = value.value

    flat_effects = [effect for effect in flat_effects if _has_effect(effect)]

    if not flat_effects:
        return value

    return Begin(effects=flat_effects, value=value)


def _optimize_term(term: Term, env: Mapping[str, Constant]) -> Term:
    match term:
        case Immediate() | Allocate():
            return term

        case Reference(name=name):
            return env.get(name, term)

        case Abstract(parameters=parameters, body=body):
            next_env = dict(env)
            for parameter in parameters:
                next_env.pop(parameter, None)
            return Abstract(parameters=parameters, body=_optimize_term(body, next_env))

        case Apply(target=target, arguments=arguments):
            return Apply(
                target=_optimize_term(target, env),
                arguments=[_optimize_term(argument, env) for argument in arguments],
            )

        case Primitive(operator=operator, left=left, right=right):
            left = _optimize_term(left, env)
            right = _optimize_term(right, env)

            if isinstance(left, Immediate) and isinstance(right, Immediate):
                match operator:
                    case "+":
                        return Immediate(value=left.value + right.value)
                    case "-":
                        return Immediate(value=left.value - right.value)
                    case "*":
                        return Immediate(value=left.value * right.value)
                    case _:
                        raise ValueError(f"unsupported primitive operator: {operator}")

            return Primitive(operator=operator, left=left, right=right)

        case Branch(
            operator=operator,
            left=left,
            right=right,
            consequent=consequent,
            otherwise=otherwise,
        ):
            left = _optimize_term(left, env)
            right = _optimize_term(right, env)
            consequent = _optimize_term(consequent, env)
            otherwise = _optimize_term(otherwise, env)

            if isinstance(left, Immediate) and isinstance(right, Immediate):
                match operator:
                    case "<":
                        return consequent if left.value < right.value else otherwise
                    case "==":
                        return consequent if left.value == right.value else otherwise
                    case _:
                        raise ValueError(f"unsupported branch operator: {operator}")

            return Branch(
                operator=operator,
                left=left,
                right=right,
                consequent=consequent,
                otherwise=otherwise,
            )

        case Load(base=base, index=index):
            return Load(base=_optimize_term(base, env), index=index)

        case Store(base=base, index=index, value=value):
            return Store(
                base=_optimize_term(base, env),
                index=index,
                value=_optimize_term(value, env),
            )

        case Begin(effects=effects, value=value):
            optimized_effects = [_optimize_term(effect, env) for effect in effects]
            optimized_value = _optimize_term(value, env)
            return _flatten_begin(optimized_effects, optimized_value)

        case Let(bindings=bindings, body=body):
            optimized_bindings: list[tuple[str, Term]] = []
            next_env = dict(env)

            for name, value in bindings:
                optimized_value = _optimize_term(value, next_env)
                optimized_bindings.append((name, optimized_value))
                next_env = _extend_env(next_env, name, optimized_value)

            optimized_body = _optimize_term(body, next_env)

            used = _free_variables(optimized_body)
            kept_bindings_rev: list[tuple[str, Term]] = []
            hoisted_effects_rev: list[Term] = []

            for name, value in reversed(optimized_bindings):
                if name in used:
                    kept_bindings_rev.append((name, value))
                    used.discard(name)
                    used |= _free_variables(value)
                elif _has_effect(value):
                    hoisted_effects_rev.append(value)
                    used |= _free_variables(value)

            result: Term = optimized_body
            kept_bindings = list(reversed(kept_bindings_rev))
            if kept_bindings:
                result = Let(bindings=kept_bindings, body=result)

            hoisted_effects = list(reversed(hoisted_effects_rev))
            if hoisted_effects:
                result = _flatten_begin(hoisted_effects, result)

            return result

    raise TypeError(f"unsupported term: {term!r}")


def optimize_program(program: Program) -> Program:
    current = program

    while True:
        optimized = Program(
            parameters=current.parameters,
            body=_optimize_term(current.body, {}),
        )

        if optimized == current:
            return optimized

        current = optimized