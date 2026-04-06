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
            if len(bindings) == 0:
                return _term(body, k)
            binding, t = bindings[0]
            return _term(
                t,
                lambda x: L1.Copy(
                    destination=binding, source=x, then=_term(L2.Let(bindings=bindings[1:], body=body), k)
                ),
            )

        case L2.Reference(name=name):
            return k(name)

        case L2.Abstract(parameters=parameters, body=body):
            new_identifier = fresh("t")
            abstract_identifier = fresh("k")
            return L1.Abstract(
                destination=new_identifier,
                parameters=parameters + [abstract_identifier],
                body=_term(body, lambda bo: L1.Apply(target=abstract_identifier, arguments=[bo])),
                then=k(new_identifier),
            )

        case L2.Apply(target=target, arguments=arguments):
            new_identifier = fresh("t")
            abstract_identifier = fresh("k")
            return _term(
                target,
                lambda tar: _terms(
                    arguments,
                    lambda args: L1.Abstract(
                        destination=abstract_identifier,
                        parameters=[new_identifier],
                        body=k(new_identifier),
                        then=L1.Apply(
                            target=tar,
                            arguments=args + [abstract_identifier],
                        ),
                    ),
                ),
            )

        case L2.Immediate(value=value):
            new_identifier = fresh("t")
            return L1.Immediate(
                destination=new_identifier,
                value=value,
                then=k(new_identifier),
            )

        case L2.Primitive(operator=operator, left=left, right=right):
            new_identifier = fresh("t")
            return _term(
                left,
                lambda le: _term(
                    right,
                    lambda ri: L1.Primitive(
                        destination=new_identifier,
                        operator=operator,
                        left=le,
                        right=ri,
                        then=k(new_identifier),
                    ),
                ),
            )

        case L2.Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            main_id = fresh("j")
            new_identifier = fresh("t")
            return _term(
                left,
                lambda le: _term(
                    right,
                    lambda ri: L1.Abstract(
                        destination=main_id,
                        parameters=[new_identifier],
                        body=k(new_identifier),
                        then=L1.Branch(
                            operator=operator,
                            left=le,
                            right=ri,
                            then=_term(
                                consequent,
                                lambda con: L1.Apply(
                                    target=main_id,
                                    arguments=[
                                        con,
                                    ],
                                ),
                            ),
                            otherwise=_term(
                                otherwise,
                                lambda oth: L1.Apply(
                                    target=main_id,
                                    arguments=[oth],
                                ),
                            ),
                        ),
                    ),
                ),
            )

        case L2.Allocate(count=count):
            new_identifier = fresh("t")
            return L1.Allocate(destination=new_identifier, count=count, then=k(new_identifier))

        case L2.Load(base=base, index=index):
            new_identifier = fresh("t")
            return _term(
                base,
                lambda x: L1.Load(
                    destination=new_identifier,
                    base=x,
                    index=index,
                    then=k(new_identifier),
                ),
            )

        case L2.Store(base=base, index=index, value=value):
            immediate_id = fresh("t")
            return _term(
                base,
                lambda ba: _term(
                    value,
                    lambda val: L1.Store(
                        base=ba,
                        index=index,
                        value=val,
                        then=L1.Immediate(
                            destination=immediate_id,
                            value=0,
                            then=k(immediate_id),
                        ),
                    ),
                ),
            )

        case L2.Begin(effects=effects, value=value):  # pragma: no branch
            if len(effects) == 0:
                return _term(value, k)
            return _term(effects[0], lambda _: _term(L2.Begin(effects=effects[1:], value=value), k))


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
