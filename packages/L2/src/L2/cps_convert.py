from collections.abc import Callable, Sequence
from functools import partial

from L1 import syntax as L1

from L2 import syntax as L2


def cps_convert_term(
    term: L2.Term,
    k: Callable[[L1.Identifier], L1.Statement],
    fresh: Callable[[str], str],
) -> L1.Statement:
    _term = partial(cps_convert_term, fresh=fresh)
    _terms = partial(cps_convert_terms, fresh=fresh)

    match term:
        case L2.Let(bindings=bindings, body=body):
            def convert_bindings(
                bindings: Sequence[tuple[L2.Identifier, L2.Term]],
            ) -> L1.Statement:
                match bindings:
                    case []:
                        return _term(body, k)

                    case [(name, value), *rest]:
                        return _term(
                            value,
                            lambda source: L1.Copy(
                                destination=name,
                                source=source,
                                then=convert_bindings(rest),
                            ),
                        )

                    case _:  # pragma: no cover
                        raise ValueError(bindings)

            return convert_bindings(bindings)

        case L2.Reference(name=name):
            return k(name)

        case L2.Abstract(parameters=parameters, body=body):
            destination = fresh("t")
            continuation = fresh("k")
            return L1.Abstract(
                destination=destination,
                parameters=[*parameters, continuation],
                body=_term(
                    body,
                    lambda value: L1.Apply(
                        target=continuation,
                        arguments=[value],
                    ),
                ),
                then=k(destination),
            )

        case L2.Apply(target=target, arguments=arguments):
            continuation = fresh("k")
            result = fresh("t")
            return L1.Abstract(
                destination=continuation,
                parameters=[result],
                body=k(result),
                then=_term(
                    target,
                    lambda target: _terms(
                        arguments,
                        lambda arguments: L1.Apply(
                            target=target,
                            arguments=[*arguments, continuation],
                        ),
                    ),
                ),
            )

        case L2.Immediate(value=value):
            destination = fresh("t")
            return L1.Immediate(
                destination=destination,
                value=value,
                then=k(destination),
            )

        case L2.Primitive(operator=operator, left=left, right=right):
            destination = fresh("t")
            return _terms(
                [left, right],
                lambda values: L1.Primitive(
                    destination=destination,
                    operator=operator,
                    left=values[0],
                    right=values[1],
                    then=k(destination),
                ),
            )

        case L2.Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            join = fresh("j")
            result = fresh("t")
            return L1.Abstract(
                destination=join,
                parameters=[result],
                body=k(result),
                then=_terms(
                    [left, right],
                    lambda values: L1.Branch(
                        operator=operator,
                        left=values[0],
                        right=values[1],
                        then=_term(
                            consequent,
                            lambda value: L1.Apply(
                                target=join,
                                arguments=[value],
                            ),
                        ),
                        otherwise=_term(
                            otherwise,
                            lambda value: L1.Apply(
                                target=join,
                                arguments=[value],
                            ),
                        ),
                    ),
                ),
            )

        case L2.Allocate(count=count):
            destination = fresh("t")
            return L1.Allocate(
                destination=destination,
                count=count,
                then=k(destination),
            )

        case L2.Load(base=base, index=index):
            destination = fresh("t")
            return _term(
                base,
                lambda base: L1.Load(
                    destination=destination,
                    base=base,
                    index=index,
                    then=k(destination),
                ),
            )

        case L2.Store(base=base, index=index, value=value):
            result = fresh("t")
            return _terms(
                [base, value],
                lambda values: L1.Store(
                    base=values[0],
                    index=index,
                    value=values[1],
                    then=L1.Immediate(
                        destination=result,
                        value=0,
                        then=k(result),
                    ),
                ),
            )

        case L2.Begin(effects=effects, value=value):  # pragma: no branch
            def convert_effects(effects: Sequence[L2.Term]) -> L1.Statement:
                match effects:
                    case []:
                        return _term(value, k)

                    case [first, *rest]:
                        return _term(first, lambda _: convert_effects(rest))

                    case _:  # pragma: no cover
                        raise ValueError(effects)

            return convert_effects(effects)

        case _:
            raise TypeError(f"Unhandled L2 term in cps_convert_term: {term!r}")


def cps_convert_terms(
    terms: Sequence[L2.Term],
    k: Callable[[Sequence[L1.Identifier]], L1.Statement],
    fresh: Callable[[str], str],
) -> L1.Statement:
    _term = partial(cps_convert_term, fresh=fresh)
    _terms = partial(cps_convert_terms, fresh=fresh)

    match terms:
        case []:
            return k([])

        case [first, *rest]:
            return _term(first, lambda first: _terms(rest, lambda rest: k([first, *rest])))

        case _:  # pragma: no cover
            raise ValueError(terms)


def cps_convert_program(
    program: L2.Program,
    fresh: Callable[[str], str],
) -> L1.Program:
    _term = partial(cps_convert_term, fresh=fresh)

    match program:
        case L2.Program(parameters=parameters, body=body):  # pragma: no branch
            return L1.Program(
                parameters=parameters,
                body=_term(body, lambda value: L1.Halt(value=value)),
            )