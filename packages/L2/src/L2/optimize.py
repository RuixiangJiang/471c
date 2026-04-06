from collections.abc import Mapping, Sequence

from L2 import syntax as L2

type Context = Mapping[L2.Identifier, int | None]


def try_resolveable(term: L2.Term, context: Context) -> int | None:
    match term:
        case L2.Reference(name=name):
            if name in context and context[name] is not None:
                return context[name]
        case L2.Immediate(value=value):
            return value
        case L2.Primitive(operator=operator, left=left, right=right):
            le, ri = try_resolveable(term=left, context=context), try_resolveable(term=right, context=context)
            if le is not None and ri is not None:
                return le + ri if operator == "+" else le - ri if operator == "-" else le * ri
        case _:
            return None
    return None


def build_folding(term: L2.Term, context: Context) -> L2.Term:
    match term:
        case L2.Let(bindings=bindings, body=body):
            new_bindings: Sequence[tuple[L2.Identifier, L2.Term]] = []
            local = dict(context)
            for name, te in bindings:
                folded_te = build_folding(term=te, context=local)
                new_bindings.append((name, folded_te))
                local[name] = folded_te.value if isinstance(folded_te, L2.Immediate) else None
            return L2.Let(bindings=new_bindings, body=build_folding(term=body, context=local))
        case L2.Abstract(parameters=parameters, body=body):
            return L2.Abstract(
                parameters=parameters,
                body=build_folding(term=body, context={**context, **{p: None for p in parameters}}),
            )
        case L2.Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            left_res = try_resolveable(term=left, context=context)
            right_res = try_resolveable(term=right, context=context)
            if left_res is not None and right_res is not None:
                early_resolution = left_res < right_res if operator == "<" else left_res == right_res
                return build_folding(term=consequent if early_resolution else otherwise, context=context)
            return L2.Branch(
                operator=operator,
                left=build_folding(term=left, context=context),
                right=build_folding(term=right, context=context),
                consequent=build_folding(term=consequent, context=context),
                otherwise=build_folding(term=otherwise, context=context),
            )
        case L2.Primitive(operator=operator, left=left, right=right):
            value = try_resolveable(term=term, context=context)
            if value is not None:
                return L2.Immediate(value=value)
            return L2.Primitive(
                operator=operator,
                left=build_folding(term=left, context=context),
                right=build_folding(term=right, context=context),
            )
        case L2.Apply(target=target, arguments=arguments):
            return L2.Apply(
                target=build_folding(term=target, context=context),
                arguments=[build_folding(term=t, context=context) for t in arguments],
            )
        case L2.Allocate(count=_):
            return term
        case L2.Immediate(value=_):
            return term
        case L2.Reference(name=name):
            value = try_resolveable(term=term, context=context)
            if value is not None:
                return L2.Immediate(value=value)
            return term
        case L2.Load(base=base, index=index):
            return L2.Load(base=build_folding(term=base, context=context), index=index)
        case L2.Store(base=base, index=index, value=value):
            return L2.Store(
                base=build_folding(term=base, context=context),
                index=index,
                value=build_folding(term=value, context=context),
            )
        case L2.Begin(effects=effects, value=value):  # pragma: no branch
            return L2.Begin(
                effects=[build_folding(term=t, context=context) for t in effects],
                value=build_folding(term=value, context=context),
            )


def collect_uses(term: L2.Term) -> set[L2.Identifier]:
    match term:
        case L2.Let(bindings=bindings, body=body):
            uses = collect_uses(term=body)
            for name, t in bindings:
                uses = uses | collect_uses(term=t)
            return uses

        case L2.Reference(name=name):
            return {name}

        case L2.Abstract(parameters=parameters, body=body):
            return collect_uses(term=body) - set(parameters)

        case L2.Apply(target=target, arguments=arguments):
            uses = collect_uses(term=target)
            for te in arguments:
                uses = uses | collect_uses(term=te)
            return uses

        case L2.Immediate(value=_):
            return set()

        case L2.Primitive(operator=_, left=left, right=right):
            return collect_uses(term=left) | collect_uses(term=right)

        case L2.Branch(operator=_, left=left, right=right, consequent=consequent, otherwise=otherwise):
            return (
                collect_uses(term=left)
                | collect_uses(term=right)
                | collect_uses(term=consequent)
                | collect_uses(term=otherwise)
            )

        case L2.Allocate(count=_):
            return set()

        case L2.Load(base=base, index=_):
            return collect_uses(term=base)

        case L2.Store(base=base, index=_, value=value):
            return collect_uses(term=base) | collect_uses(term=value)

        case L2.Begin(effects=effects, value=value):  # pragma: no branch
            uses = collect_uses(term=value)
            for t in effects:
                uses = uses | collect_uses(term=t)
            return uses


def dead_code_elimination(term: L2.Term) -> L2.Term:
    match term:
        case L2.Let(bindings=bindings, body=body):
            new_body = dead_code_elimination(term=body)
            new_bindings: Sequence[tuple[L2.Identifier, L2.Term]] = []
            uses = collect_uses(term=body)
            for name, te in bindings:
                if name in uses:
                    new_bindings.append((name, dead_code_elimination(term=te)))
            return L2.Let(bindings=new_bindings, body=new_body)

        case L2.Reference(name=name):
            return term

        case L2.Abstract(parameters=parameters, body=body):
            return L2.Abstract(parameters=parameters, body=dead_code_elimination(term=body))

        case L2.Apply(target=target, arguments=arguments):
            new_target = dead_code_elimination(term=target)
            new_arguments: Sequence[L2.Term] = []
            for t in arguments:
                new_arguments.append(dead_code_elimination(term=t))
            return L2.Apply(
                target=new_target,
                arguments=new_arguments,
            )

        case L2.Immediate(value=_):
            return term

        case L2.Primitive(operator=operator, left=left, right=right):
            return L2.Primitive(
                operator=operator,
                left=dead_code_elimination(term=left),
                right=dead_code_elimination(term=right),
            )

        case L2.Branch(operator=operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            return L2.Branch(
                operator=operator,
                left=dead_code_elimination(term=left),
                right=dead_code_elimination(term=right),
                consequent=dead_code_elimination(term=consequent),
                otherwise=dead_code_elimination(term=otherwise),
            )

        case L2.Allocate(count=_):
            return term

        case L2.Load(base=base, index=index):
            return L2.Load(base=dead_code_elimination(term=base), index=index)

        case L2.Store(base=base, index=_index, value=value):
            return L2.Store(
                base=dead_code_elimination(term=base),
                index=_index,
                value=dead_code_elimination(term=value),
            )

        case L2.Begin(effects=effects, value=value):  # pragma: no branch
            return L2.Begin(effects=effects, value=dead_code_elimination(term=value))


def optimize_program(
    program: L2.Program,
) -> L2.Program:
    match program:
        case L2.Program(parameters=parameters, body=body):  # pragma: no branch
            context = {name: None for name in parameters}
            folded_body = body
            for _ in range(5):
                folded_body = build_folding(term=folded_body, context=context)
            optimized_body = dead_code_elimination(term=folded_body)
            return L2.Program(parameters=parameters, body=optimized_body)
