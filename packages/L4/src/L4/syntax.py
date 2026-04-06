from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

type Identifier = Annotated[str, Field(min_length=1)]
type VName = Annotated[str, Field(min_length=1, max_length=32)]
type Positive = Annotated[int, Field(ge=1)]
type Nat = Annotated[int, Field(ge=0)]

type Type = Annotated[Mutable | Int | Bool | FuncType | List | Pair | Symbol | Void, Field(discriminator="tag")]

type Expression = Annotated[
    Function
    | If
    | Reference
    | Immediate
    | Let
    | LetRec
    | Operation
    | Call
    | Empty
    | NewList
    | NewPair
    | HeapAllocate
    | Get
    | Set
    | Capsule
    | While
    | For
    | Bunch,
    Field(discriminator="tag"),
]


class Program(BaseModel, frozen=True):
    tag: Literal["l4"] = "l4"
    definitions: Sequence[tuple[Identifier, Type, Expression]]
    body: Expression


class Int(BaseModel, frozen=True):
    tag: Literal["int"] = "int"


class Bool(BaseModel, frozen=True):
    tag: Literal["bool"] = "bool"


class Void(BaseModel, frozen=True):
    tag: Literal["void"] = "void"


class Symbol(BaseModel, frozen=True):
    tag: Literal["symbol"] = "symbol"
    name: VName
    payload: Type


class FuncType(BaseModel, frozen=True):
    tag: Literal["functype"] = "functype"
    parameters: Sequence[Type]
    result: Type


class List(BaseModel, frozen=True):
    tag: Literal["list"] = "list"
    typeof: Type


class Pair(BaseModel, frozen=True):
    tag: Literal["pair"] = "pair"
    type1: Type
    type2: Type


class Mutable(BaseModel, frozen=True):
    tag: Literal["mutable"] = "mutable"
    oftype: Type


class Immediate(BaseModel, frozen=True):
    tag: Literal["immediate"] = "immediate"
    value: bool | int | None


class Reference(BaseModel, frozen=True):
    tag: Literal["reference"] = "reference"
    name: Identifier


class HeapAllocate(BaseModel, frozen=True):
    tag: Literal["heapallocate"] = "heapallocate"
    val: Expression


class If(BaseModel, frozen=True):
    tag: Literal["if"] = "if"
    condition: Expression
    consequent: Expression
    otherwise: Expression


class Function(BaseModel, frozen=True):
    tag: Literal["function"] = "function"
    params: Sequence[tuple[Identifier, Type]]
    body: Expression


class Call(BaseModel, frozen=True):
    tag: Literal["call"] = "call"
    target: Expression
    arguments: Sequence[Expression]


class Operation(BaseModel, frozen=True):
    tag: Literal["operation"] = "operation"
    operator: Literal["+", "-", "*", "==", "<"]
    left: Expression
    right: Expression


class Let(BaseModel, frozen=True):
    tag: Literal["let"] = "let"
    bindings: Sequence[tuple[Identifier, Type, Expression]]
    body: Expression


class LetRec(BaseModel, frozen=True):
    tag: Literal["letrec"] = "letrec"
    bindings: Sequence[tuple[Identifier, Type, Expression]]
    body: Expression


class Empty(BaseModel, frozen=True):
    tag: Literal["empty"] = "empty"


class NewList(BaseModel, frozen=True):
    tag: Literal["newlist"] = "newlist"
    size: Positive
    typeof: Type


class NewPair(BaseModel, frozen=True):
    tag: Literal["newpair"] = "newpair"
    val1: Expression
    val2: Expression
    typeof: Type


class Get(BaseModel, frozen=True):
    tag: Literal["get"] = "get"
    target: Reference
    index: Nat


class Set(BaseModel, frozen=True):
    tag: Literal["set"] = "set"
    target: Reference
    index: Nat
    value: Expression


class Capsule(BaseModel, frozen=True):
    tag: Literal["capsule"] = "capsule"
    typeof: Type
    expression: Expression


class While(BaseModel, frozen=True):
    tag: Literal["while"] = "while"
    condition: Expression
    run: Expression


class For(BaseModel, frozen=True):
    tag: Literal["for"] = "for"
    times: Positive | Expression
    run: Expression


class Bunch(BaseModel, frozen=True):
    tag: Literal["bunch"] = "bunch"
    expressions: Sequence[Expression]


Function.model_rebuild()
If.model_rebuild()
Reference.model_rebuild()
Immediate.model_rebuild()
Let.model_rebuild()
LetRec.model_rebuild()
Operation.model_rebuild()
Call.model_rebuild()
Empty.model_rebuild()
NewList.model_rebuild()
NewPair.model_rebuild()
Pair.model_rebuild()
HeapAllocate.model_rebuild()
Get.model_rebuild()
Set.model_rebuild()
Mutable.model_rebuild()
Int.model_rebuild()
Bool.model_rebuild()
FuncType.model_rebuild()
List.model_rebuild()
Void.model_rebuild()
Program.model_rebuild()
Capsule.model_rebuild()
While.model_rebuild()
For.model_rebuild()
Bunch.model_rebuild()
Symbol.model_rebuild()
