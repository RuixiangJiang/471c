from collections.abc import Sequence
from pathlib import Path

from lark import Lark, Token, Transformer
from lark.visitors import v_args

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


class AstTransformer(Transformer[Token, Program | Term]):
    def IDENTIFIER(self, t: Token) -> Identifier:
        return str(t)

    def INT(self, t: Token) -> int:
        return int(t)

    def parameters(self, parameters: Sequence[Identifier]) -> list[Identifier]:
        return list(parameters)

    def bindings(self, bindings: Sequence[tuple[Identifier, Term]]) -> list[tuple[Identifier, Term]]:
        return list(bindings)

    @v_args(inline=True)
    def binding(self, name: Identifier, value: Term) -> tuple[Identifier, Term]:
        return (name, value)

    @v_args(inline=True)
    def term(self, t: Term) -> Term:
        return t

    @v_args(inline=True)
    def program(self, _program: Token, parameters: Sequence[Identifier], body: Term) -> Program:
        return Program(parameters=list(parameters), body=body)

    @v_args(inline=True)
    def let(self, _let: Token, bindings: Sequence[tuple[Identifier, Term]], body: Term) -> Term:
        return Let(bindings=list(bindings), body=body)

    @v_args(inline=True)
    def letrec(self, _letrec: Token, bindings: Sequence[tuple[Identifier, Term]], body: Term) -> Term:
        return LetRec(bindings=list(bindings), body=body)

    @v_args(inline=True)
    def reference(self, name: Identifier) -> Term:
        return Reference(name=name)

    @v_args(inline=True)
    def abstract(self, _lambda: Token, parameters: Sequence[Identifier], body: Term) -> Term:
        return Abstract(parameters=list(parameters), body=body)

    @v_args(inline=True)
    def apply(self, target: Term, *arguments: Term) -> Term:
        return Apply(target=target, arguments=list(arguments))

    @v_args(inline=True)
    def immediate(self, value: int) -> Term:
        return Immediate(value=value)

    @v_args(inline=True)
    def primitive(self, operator: Token, left: Term, right: Term) -> Term:
        return Primitive(operator=str(operator), left=left, right=right)

    @v_args(inline=True)
    def condition(self, operator: Token, left: Term, right: Term) -> tuple[str, Term, Term]:
        return (str(operator), left, right)

    @v_args(inline=True)
    def branch(self, _if: Token, cond: tuple[str, Term, Term], consequent: Term, otherwise: Term) -> Term:
        operator, left, right = cond
        return Branch(
            operator=operator,
            left=left,
            right=right,
            consequent=consequent,
            otherwise=otherwise,
        )

    @v_args(inline=True)
    def allocate(self, _allocate: Token, count: int) -> Term:
        return Allocate(count=count)

    @v_args(inline=True)
    def load(self, _load: Token, base: Term, index: int) -> Term:
        return Load(base=base, index=index)

    @v_args(inline=True)
    def store(self, _store: Token, base: Term, index: int, value: Term) -> Term:
        return Store(base=base, index=index, value=value)

    @v_args(inline=True)
    def begin(self, _begin: Token, *terms: Term) -> Term:
        ts = list(terms)
        return Begin(effects=ts[:-1], value=ts[-1])


def parse_term(source: str) -> Term:
    grammar = Path(__file__).with_name("L3.lark").read_text()
    parser = Lark(grammar, start="term", parser="lalr")
    tree = parser.parse(source)
    return AstTransformer().transform(tree)


def parse_program(source: str) -> Program:
    grammar = Path(__file__).with_name("L3.lark").read_text()
    parser = Lark(grammar, start="program", parser="lalr")
    tree = parser.parse(source)
    return AstTransformer().transform(tree)