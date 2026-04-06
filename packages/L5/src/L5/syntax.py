from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from L4 import syntax as L4

type Identifier = L4.Identifier
type VName = L4.VName
type Positive = L4.Positive
type Nat = L4.Nat
type Type = L4.Type

type Expression = Annotated[
    L4.Function
    | L4.If
    | L4.Reference
    | L4.Immediate
    | L4.Let
    | L4.LetRec
    | L4.Operation
    | L4.Call
    | L4.Empty
    | L4.NewList
    | L4.NewPair
    | L4.HeapAllocate
    | L4.Get
    | L4.Set
    | L4.Capsule
    | L4.While
    | L4.For
    | L4.Bunch
    | ShortCircuit
    | Switch
    | Break
    | Continue
    | Foreach,
    Field(discriminator="tag"),
]


class Program(BaseModel, frozen=True):
    tag: Literal["l5"] = "l5"
    definitions: Sequence[tuple[Identifier, Type, Expression]]
    body: Expression


class ShortCircuit(BaseModel, frozen=True):
    tag: Literal["shortcircuit"] = "shortcircuit"
    operator: Literal["&&", "||"]
    left: Expression
    right: Expression


class SwitchCase(BaseModel, frozen=True):
    tag: Literal["switchcase"] = "switchcase"
    value: bool | int
    body: Expression


class Switch(BaseModel, frozen=True):
    tag: Literal["switch"] = "switch"
    scrutinee: Expression
    cases: Sequence[SwitchCase]
    default: Expression


class Break(BaseModel, frozen=True):
    tag: Literal["break"] = "break"


class Continue(BaseModel, frozen=True):
    tag: Literal["continue"] = "continue"


class Foreach(BaseModel, frozen=True):
    """
    Static-unrolled foreach.

    Because L4.Get only accepts a Nat constant index, this version requires:
      - target must be a Reference
      - count must be a compile-time Positive integer

    Example:
      Foreach(
          binder="x",
          typeof=L4.Int(),
          target=L4.Reference(name="xs"),
          count=4,
          run=...
      )
    """
    tag: Literal["foreach"] = "foreach"
    binder: Identifier
    typeof: Type
    target: L4.Reference
    count: Nat
    run: Expression


Program.model_rebuild()
ShortCircuit.model_rebuild()
SwitchCase.model_rebuild()
Switch.model_rebuild()
Break.model_rebuild()
Continue.model_rebuild()
Foreach.model_rebuild()