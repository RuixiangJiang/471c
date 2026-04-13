from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from L4 import syntax as L4

type Identifier = L4.Identifier
type VName = L4.VName
type Positive = L4.Positive
type Nat = L4.Nat

type Type = Annotated[
    L4.Mutable
    | L4.Int
    | L4.Bool
    | L4.FuncType
    | L4.List
    | L4.Pair
    | L4.Symbol
    | L4.Void
    | ClassType,
    Field(discriminator="tag"),
]

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
    | Foreach
    | This
    | NewObject
    | FieldAccess
    | FieldAssign
    | MethodCall,
    Field(discriminator="tag"),
]


class ClassType(BaseModel, frozen=True):
    tag: Literal["classtype"] = "classtype"
    name: Identifier


class FieldDef(BaseModel, frozen=True):
    tag: Literal["fielddef"] = "fielddef"
    name: Identifier
    typeof: Type


class MethodDef(BaseModel, frozen=True):
    tag: Literal["methoddef"] = "methoddef"
    name: Identifier
    parameters: Sequence[tuple[Identifier, Type]]
    returns: Type
    body: Expression


class ClassDef(BaseModel, frozen=True):
    tag: Literal["classdef"] = "classdef"
    name: Identifier
    parent: Identifier | None = None
    fields: Sequence[FieldDef]
    methods: Sequence[MethodDef]


class Program(BaseModel, frozen=True):
    tag: Literal["l5"] = "l5"
    classes: Sequence[ClassDef]
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
    tag: Literal["foreach"] = "foreach"
    binder: Identifier
    typeof: Type
    target: L4.Reference
    count: Nat
    run: Expression


class This(BaseModel, frozen=True):
    tag: Literal["this"] = "this"


class NewObject(BaseModel, frozen=True):
    tag: Literal["newobject"] = "newobject"
    name: Identifier
    arguments: Sequence[Expression]


class FieldAccess(BaseModel, frozen=True):
    tag: Literal["fieldaccess"] = "fieldaccess"
    target: Expression
    field: Identifier


class FieldAssign(BaseModel, frozen=True):
    tag: Literal["fieldassign"] = "fieldassign"
    target: Expression
    field: Identifier
    value: Expression


class MethodCall(BaseModel, frozen=True):
    tag: Literal["methodcall"] = "methodcall"
    target: Expression
    method: Identifier
    arguments: Sequence[Expression]


ClassType.model_rebuild()
FieldDef.model_rebuild()
MethodDef.model_rebuild()
ClassDef.model_rebuild()
Program.model_rebuild()
ShortCircuit.model_rebuild()
SwitchCase.model_rebuild()
Switch.model_rebuild()
Break.model_rebuild()
Continue.model_rebuild()
Foreach.model_rebuild()
This.model_rebuild()
NewObject.model_rebuild()
FieldAccess.model_rebuild()
FieldAssign.model_rebuild()
MethodCall.model_rebuild()