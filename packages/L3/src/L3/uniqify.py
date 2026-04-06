from collections.abc import Callable, Mapping
from functools import partial

from util.sequential_name_generator import SequentialNameGenerator

from .syntax import (
    Abstract,
    Allocate,
    Apply,
    Begin,
    Branch,
    Immediate,
    Let,
    LetRec,
    Load,
    Primitive,
    Program,
    Reference,
    Store,
    Term,
)

type Context = Mapping[str, str]


def uniqify_term(
    term: Term,
    context: Context,
    fresh: Callable[[str], str],
) -> Term:
    _term = partial(uniqify_term, context=context, fresh=fresh)

    match term:
        case Let(bindings=bindings, body=body):
            new_bindings = []
            new_context = {**context}
            for ref, t in bindings:
                new_ref = fresh(ref)
                new_context = {**new_context, ref: new_ref}
                new_bindings.append((new_ref, _term(term=t, context=context, fresh=fresh)))
            return Let(bindings=new_bindings, body=_term(term=body, context=new_context, fresh=fresh))

        case LetRec(bindings=bindings, body=body):
            new_bindings = []
            new_context = {**context}
            for ref, t in bindings:
                new_ref = fresh(ref)
                new_context = {**new_context, ref: new_ref}
                new_bindings.append((new_ref, _term(term=t, context=new_context, fresh=fresh)))
            return LetRec(bindings=new_bindings, body=_term(term=body, context=new_context, fresh=fresh))

        case Reference(name=name):
            return Reference(name=context.get(name, name))

        case Abstract(parameters=parameters, body=body):
            new_context = {**context}
            new_parameters = []
            for ref in parameters:
                new_ref = fresh(ref)
                new_context = {**new_context, ref: new_ref}
                new_parameters.append(new_ref)
            return Abstract(parameters=new_parameters, body=_term(term=body, context=new_context, fresh=fresh))

        case Apply(target=target, arguments=arguments):
            new_arguments = [_term(term=t, context=context, fresh=fresh) for t in arguments]
            return Apply(target=_term(term=target, context=context, fresh=fresh), arguments=new_arguments)

        case Immediate():
            return term

        case Primitive(operator=operator, left=left, right=right):
            return Primitive(
                operator=operator,
                left=_term(term=left, context=context, fresh=fresh),
                right=_term(term=right, context=context, fresh=fresh),
            )

        case Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            return Branch(
                operator=operator,
                left=_term(term=left, context=context, fresh=fresh),
                right=_term(term=right, context=context, fresh=fresh),
                consequent=_term(term=consequent, context=context, fresh=fresh),
                otherwise=_term(term=otherwise, context=context, fresh=fresh),
            )

        case Allocate():
            return term

        case Load(base=base, index=index):
            return Load(base=_term(term=base, context=context, fresh=fresh), index=index)

        case Store(base=base, index=index, value=value):
            return Store(
                base=_term(term=base, context=context, fresh=fresh),
                index=index,
                value=_term(term=value, context=context, fresh=fresh),
            )

        case Begin(effects=effects, value=value):  # pragma: no branch
            new_effects = [_term(term=t, context=context, fresh=fresh) for t in effects]
            return Begin(effects=new_effects, value=_term(term=value, context=context, fresh=fresh))


def uniqify_program(
    program: Program,
) -> tuple[Callable[[str], str], Program]:
    fresh = SequentialNameGenerator()

    _term = partial(uniqify_term, fresh=fresh)

    match program:
        case Program(parameters=parameters, body=body):  # pragma: no branch
            local = {parameter: fresh(parameter) for parameter in parameters}
            return (
                fresh,
                Program(
                    parameters=[local[parameter] for parameter in parameters],
                    body=_term(body, local),
                ),
            )
