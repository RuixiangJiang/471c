# noqa: F841
from collections.abc import Mapping
from functools import partial

from L2 import syntax as L2

from . import syntax as L3

type Context = Mapping[L3.Identifier, None]


def eliminate_letrec_term(
    term: L3.Term,
    context: Context,
) -> L2.Term:
    recur = partial(eliminate_letrec_term, context=context)

    match term:
        case L3.Let(bindings=bindings, body=body):
            binding_ids = {identifier: None for identifier, _ in bindings}
            l2_context = {key: val for key, val in context.items() if key not in binding_ids}
            recur_with_no_binding_context = partial(eliminate_letrec_term, context=l2_context)
            l2_bindings_list = [(identifier, recur(binding)) for identifier, binding in bindings]
            return L2.Let(bindings=l2_bindings_list, body=recur_with_no_binding_context(body))

        case L3.LetRec(bindings=bindings, body=body):
            binding_as_context = {identifier: None for identifier, _ in bindings}
            l2_context = {**binding_as_context, **context}
            recur_with_updated_context = partial(eliminate_letrec_term, context=l2_context)
            l2_bindings_allocates = [(identifier, L2.Allocate(count=1)) for identifier, _ in bindings]
            l2_bindings_stores = [
                L2.Store(
                    base=L2.Reference(name=identifier),
                    index=0,
                    value=recur_with_updated_context(binding),
                )
                for identifier, binding in bindings
            ]
            return L2.Let(
                bindings=[*l2_bindings_allocates],
                body=L2.Begin(
                    effects=[*l2_bindings_stores],
                    value=recur_with_updated_context(body),
                ),
            )

        case L3.Reference(name=name):
            l2_reference = L2.Reference(name=name)
            if name in context:
                return L2.Load(base=l2_reference, index=0)
            return l2_reference

        case L3.Abstract(parameters=parameters, body=body):
            parameter_ids = {parameter: None for parameter in parameters}
            l2_context = {key: val for key, val in context.items() if key not in parameter_ids}
            recur_with_no_parameter_context = partial(eliminate_letrec_term, context=l2_context)
            return L2.Abstract(parameters=parameters, body=recur_with_no_parameter_context(body))

        case L3.Apply(target=target, arguments=arguments):
            return L2.Apply(target=recur(target), arguments=[recur(argument) for argument in arguments])

        case L3.Immediate(value=value):
            return L2.Immediate(value=value)

        case L3.Primitive(operator=operator, left=left, right=right):
            return L2.Primitive(operator=operator, left=recur(left), right=recur(right))

        case L3.Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            return L2.Branch(
                operator=operator,
                left=recur(left),
                right=recur(right),
                consequent=recur(consequent),
                otherwise=recur(otherwise),
            )

        case L3.Allocate(count=count):
            return L2.Allocate(count=count)

        case L3.Load(base=base, index=index):
            return L2.Load(
                base=recur(base),
                index=index,
            )

        case L3.Store(base=base, index=index, value=value):
            return L2.Store(
                base=recur(base),
                index=index,
                value=recur(value),
            )

        case L3.Begin(effects=effects, value=value):  # pragma: no branch
            return L2.Begin(
                effects=[recur(effect) for effect in effects],
                value=recur(value),
            )


def eliminate_letrec_program(
    program: L3.Program,
) -> L2.Program:
    match program:
        case L3.Program(parameters=parameters, body=body):  # pragma: no branch
            return L2.Program(
                parameters=parameters,
                body=eliminate_letrec_term(body, {}),
            )
