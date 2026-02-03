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
    Reference,
    Store,
    Term,
)

type Context = Mapping[Identifier, None]


def check_term(
    term: Term,
    context: Context,
) -> None:
    recur = partial(check_term, context=context)  # noqa: F841

    match term:
        case Let():
            pass

        case LetRec():
            pass

        case Reference():
            pass

        case Abstract():
            pass

        case Apply():
            pass

        case Immediate():
            pass

        case Primitive():
            pass

        case Branch():
            pass

        case Allocate():
            pass

        case Load():
            pass

        case Store():
            pass

        case Begin():  # pragma: no cover
            pass
