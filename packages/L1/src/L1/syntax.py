from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

type Identifier = Annotated[str, Field(min_length=1)]
type Nat = Annotated[int, Field(ge=0)]


class Program(BaseModel, frozen=True):
    tag: Literal["l1"] = "l1"
    parameters: Sequence[Identifier]
    body: Statement


type Statement = Annotated[
    Copy | Abstract | Apply | Immediate | Primitive | Branch | Allocate | Load | Store | Halt,
    Field(discriminator="tag"),
]


class Copy(BaseModel, frozen=True):
    tag: Literal["copy"] = "copy"
    destination: Identifier
    source: Identifier
    then: Statement


class Abstract(BaseModel, frozen=True):
    tag: Literal["abstract"] = "abstract"
    destination: Identifier
    parameters: Sequence[Identifier]
    body: Statement
    then: Statement


class Apply(BaseModel, frozen=True):
    tag: Literal["apply"] = "apply"
    target: Identifier
    arguments: Sequence[Identifier]


class Immediate(BaseModel, frozen=True):
    tag: Literal["immediate"] = "immediate"
    destination: Identifier
    value: int
    then: Statement


class Primitive(BaseModel, frozen=True):
    tag: Literal["primitive"] = "primitive"
    destination: Identifier
    operator: Literal["+", "-", "*"]
    left: Identifier
    right: Identifier
    then: Statement


class Branch(BaseModel, frozen=True):
    tag: Literal["branch"] = "branch"
    operator: Literal["<", "=="]
    left: Identifier
    right: Identifier
    then: Statement
    otherwise: Statement


class Allocate(BaseModel, frozen=True):
    tag: Literal["allocate"] = "allocate"
    destination: Identifier
    count: Nat
    then: Statement


class Load(BaseModel, frozen=True):
    tag: Literal["load"] = "load"
    destination: Identifier
    base: Identifier
    index: Nat
    then: Statement


class Store(BaseModel, frozen=True):
    tag: Literal["store"] = "store"
    base: Identifier
    index: Nat
    value: Identifier
    then: Statement


class Halt(BaseModel, frozen=True):
    tag: Literal["halt"] = "halt"
    value: Identifier
