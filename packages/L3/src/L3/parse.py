from collections.abc import Sequence
from pathlib import Path
from typing import cast

from lark import Lark, Token, Transformer, Tree
from lark.visitors import v_args  # pyright: ignore[reportUnknownVariableType]

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
    Nat,
    Primitive,
    Program,
    Reference,
    Store,
    Term,
)


class AstTransformer(Transformer[Token, Program | Term]):
    @v_args(inline=True)
    def program(
        self,
        _program: Token,
        parameters: Sequence[Identifier],
        body: Term,
    ) -> Program:
        return Program(
            parameters=parameters,
            body=body,
        )

    def __default_token__(
        self,
        wild: Token,
    ) -> Token | Identifier | Immediate:
        if wild.type == "IDENTIFIER":
            return wild.value
        if wild.type == "INTEGER":
            return Immediate(value=int(wild.value))
        return wild

    @v_args(inline=True)
    def number(
        self,
        wild: Token,
    ) -> Immediate:
        return Immediate(value=int(wild.value))

    @v_args(inline=True)
    def nat(
        self,
        wild: Token,
    ) -> Nat:
        return int(wild.value)

    def parameters(
        self,
        parameters: Sequence[Identifier],
    ) -> Sequence[Identifier]:
        return parameters

    @v_args(inline=True)
    def term(
        self,
        term: Term,
    ) -> Term:
        if isinstance(term, str):
            return Reference(name=term)
        return term

    @v_args(inline=True)
    def let(
        self,
        _let: Token,
        bindings: Sequence[tuple[Identifier, Term]],
        body: Term,
    ) -> Term:
        return Let(
            bindings=bindings,
            body=body,
        )

    @v_args(inline=True)
    def letrec(
        self,
        _letrec: Token,
        bindings: Sequence[tuple[Identifier, Term]],
        body: Term,
    ) -> Term:
        return LetRec(
            bindings=bindings,
            body=body,
        )

    @v_args(inline=True)
    def abstract(self, _: Token, parameters: Sequence[Identifier], body: Term) -> Term:
        return Abstract(parameters=parameters, body=body)

    def apply(self, body: list[Tree[Term]]) -> Term:
        tree: Tree[Term] = body[0]
        terms: Sequence[Term] = [cast(Term, child) for child in tree.children]
        target, arguments = terms[0], terms[1:]
        return Apply(target=target, arguments=arguments)

    @v_args(inline=True)
    def primitive(self, operator: Token, left: Term, right: Term) -> Term:
        return Primitive(operator=operator.value, left=left, right=right)

    @v_args(inline=True)
    def branch(self, _: Token, operator: Token, left: Term, right: Term, consequent: Term, otherwise: Term) -> Term:
        return Branch(operator=operator.value, left=left, right=right, consequent=consequent, otherwise=otherwise)

    @v_args(inline=True)
    def allocate(self, _: Token, count: Nat) -> Term:
        return Allocate(count=count)

    @v_args(inline=True)
    def load(self, _: Token, base: Term, index: Nat) -> Term:
        return Load(base=base, index=index)

    @v_args(inline=True)
    def store(self, _: Token, base: Term, index: Nat, value: Term) -> Term:
        return Store(base=base, index=index, value=value)

    def begin(self, body: tuple[Token, Tree[Sequence[Term]]]) -> Term:
        tree: Tree[Sequence[Term]] = body[1]
        terms: Sequence[Term] = [cast(Term, child) for child in tree.children]
        effects, value = terms[0:-1], terms[-1]
        return Begin(effects=effects, value=value)

    def bindings(
        self,
        bindings: Sequence[tuple[Identifier, Term]],
    ) -> Sequence[tuple[Identifier, Term]]:
        return bindings

    @v_args(inline=True)
    def binding(
        self,
        name: Identifier,
        value: Term,
    ) -> tuple[Identifier, Term]:
        return name, value


def parse_term(source: str) -> Term:
    grammar = Path(__file__).with_name("L3.lark").read_text()
    parser = Lark(grammar, start="term")
    tree = parser.parse(source)  # pyright: ignore[reportUnknownMemberType]
    return AstTransformer().transform(tree)  # pyright: ignore[reportReturnType]


def parse_program(source: str) -> Program:
    grammar = Path(__file__).with_name("L3.lark").read_text()
    parser = Lark(grammar, start="program")
    tree = parser.parse(source)  # pyright: ignore[reportUnknownMemberType]
    return AstTransformer().transform(tree)  # pyright: ignore[reportReturnType]
