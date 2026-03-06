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
    """
    Eliminate L3 LetRec by rewriting recursive bindings into heap-allocated cells.

    `context` tracks the set of recursive variables currently in scope.
    Any `Reference(name)` where `name in context` becomes `Load(Reference(name), 0)`.
    """
    recur = partial(eliminate_letrec_term, context=context)

    match term:
        case L3.Let(bindings=bindings, body=body):
            # parallel let: RHS do not see same-let binders (handled by the checker).
            new_bindings: list[tuple[L3.Identifier, L2.Term]] = []
            for name, value in bindings:
                new_bindings.append((name, recur(value)))

            return L2.Let(
                bindings=new_bindings,
                body=eliminate_letrec_term(body, context=context),
            )

        case L3.LetRec(bindings=bindings, body=body):
            # letrec elimination:
            #   letrec x = v in b
            #     => let x = allocate(1) in begin store(x[0], v'); b'
            rec_names = [name for name, _ in bindings]
            rec_ctx: dict[L3.Identifier, None] = {**context, **dict.fromkeys(rec_names, None)}

            # 1) allocate a 1-slot cell for each binder
            alloc_bindings: list[tuple[L3.Identifier, L2.Term]] = [
                (name, L2.Allocate(count=1)) for name, _ in bindings
            ]

            # 2) store each RHS into its cell
            effects: list[L2.Term] = []
            for name, value in bindings:
                effects.append(
                    L2.Store(
                        base=L2.Reference(name=name),
                        index=0,
                        value=eliminate_letrec_term(value, context=rec_ctx),
                    )
                )

            # 3) translate body under recursive context
            new_body = eliminate_letrec_term(body, context=rec_ctx)

            return L2.Let(
                bindings=alloc_bindings,
                body=L2.Begin(effects=effects, value=new_body),
            )

        case L3.Reference(name=name):
            # recursive var => load(name[0]); otherwise plain reference
            if name in context:
                return L2.Load(
                    base=L2.Reference(name=name),
                    index=0,
                )
            return L2.Reference(name=name)

        case L3.Abstract(parameters=parameters, body=body):
            return L2.Abstract(
                parameters=parameters,
                body=eliminate_letrec_term(body, context=context),
            )

        case L3.Apply(target=target, arguments=arguments):
            return L2.Apply(
                target=recur(target),
                arguments=[recur(a) for a in arguments],
            )

        case L3.Immediate(value=value):
            return L2.Immediate(value=value)

        case L3.Primitive(operator=operator, left=left, right=right):
            return L2.Primitive(
                operator=operator,
                left=recur(left),
                right=recur(right),
            )

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
                effects=[recur(e) for e in effects],
                value=recur(value),
            )

    raise TypeError(f"Unhandled L3 term in eliminate_letrec_term: {term!r}")


def eliminate_letrec_program(
    program: L3.Program,
) -> L2.Program:
    match program:
        case L3.Program(parameters=parameters, body=body):  # pragma: no branch
            return L2.Program(
                parameters=parameters,
                body=eliminate_letrec_term(body, {}),
            )