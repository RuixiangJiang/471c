from collections import Counter
from collections.abc import Mapping
from functools import partial

from .syntax import (
    Abstract,
    Allocate,
    Apply,
    Begin,
    Branch,
    Identifier,
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

type Context = Mapping[Identifier, None]


def check_term(
    term: Term,
    context: Context,
) -> None:
    """
    Perform a lightweight static semantic check over an L3 AST term.

    What it checks:
    - Unbound variable use:
        * Any `Reference(name=...)` must appear in `context`, otherwise raise ValueError.
    - Duplicate bindings / shadowing disallowed:
        * `Let` cannot bind the same identifier twice (and cannot reuse an identifier already in scope).
        * `LetRec` cannot bind the same identifier twice (and cannot reuse an identifier already in scope).
        * `Abstract` cannot have duplicate parameters (and cannot reuse an identifier already in scope).
      (This matches the intended lowering strategy where reusing names may break semantics.)

    What it does NOT check:
    - Types, arity, operator validity, memory bounds, etc.
      It only ensures that variable binding structure is well-formed.

    Traversal strategy:
    - Structural recursion: recursively visits sub-terms to ensure checks apply everywhere.
    - `recur` is a convenience wrapper that reuses the same `context`.
    """
    def recur(t: Term, *, ctx: Context = context) -> None:
        check_term(t, ctx)

    match term:
        case Let(bindings=bindings, body=body):
            counts = Counter(name for name, _ in bindings)
            duplicates = {name: count for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate binders: {duplicates}")

            # parallel let: each RHS sees only the incoming context
            for _, value in bindings:
                recur(value, ctx=context)

            local = dict.fromkeys([name for name, _ in bindings], None)
            recur(body, ctx={**context, **local})

        case LetRec(bindings=bindings, body=body):
            counts = Counter(name for name, _ in bindings)
            duplicates = {name: count for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate binders: {duplicates}")

            local = dict.fromkeys([name for name, _ in bindings], None)

            # letrec RHS sees all binders
            for _name, value in bindings:
                recur(value, ctx={**context, **local})

            recur(body, ctx={**context, **local})

        case Reference(name=name):
            if name not in context:
                raise ValueError(f"unknown variable: {name}")

        case Abstract(parameters=parameters, body=body):
            counts = Counter(parameters)
            duplicates = {name for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate parameters: {duplicates}")

            local = dict.fromkeys(parameters, None)
            recur(body, ctx={**context, **local})

        case Apply(target=target, arguments=arguments):
            recur(target, ctx=context)
            for argument in arguments:
                recur(argument, ctx=context)

        case Immediate(value=_value):
            return

        case Primitive(operator=_operator, left=left, right=right):
            recur(left, ctx=context)
            recur(right, ctx=context)

        case Branch(operator=_operator, left=left, right=right, consequent=consequent, otherwise=otherwise):
            recur(left, ctx=context)
            recur(right, ctx=context)
            recur(consequent, ctx=context)
            recur(otherwise, ctx=context)

        case Allocate(count=_count):
            return

        case Load(base=base, index=_index):
            recur(base, ctx=context)

        case Store(base=base, index=_index, value=value):
            recur(base, ctx=context)
            recur(value, ctx=context)

        case Begin(effects=effects, value=value):  # pragma: no branch
            for effect in effects:
                recur(effect, ctx=context)
            recur(value, ctx=context)


def check_program(
    program: Program,
) -> None:
    match program:
        case Program(parameters=parameters, body=body):  # pragma: no branch
            counts = Counter(parameters)
            duplicates = {name for name, count in counts.items() if count > 1}
            if duplicates:
                raise ValueError(f"duplicate parameters: {duplicates}")

            local = dict.fromkeys(parameters, None)
            check_term(body, context=local)