from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from L4 import syntax as L4

from . import syntax as L5


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class Token:
    value: str
    line: int
    col: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1

    while i < len(source):
        ch = source[i]

        if ch in " \t\r":
            i += 1
            col += 1
            continue

        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue

        if ch == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
                col += 1
            continue

        if ch in "()":
            tokens.append(Token(ch, line, col))
            i += 1
            col += 1
            continue

        start = i
        start_col = col
        while i < len(source) and source[i] not in " \t\r\n();":
            i += 1
            col += 1

        tokens.append(Token(source[start:i], line, start_col))

    return tokens


def parse_sexp(source: str) -> Any:
    tokens = tokenize(source)
    pos = 0

    def parse_one() -> Any:
        nonlocal pos

        if pos >= len(tokens):
            raise ParseError("unexpected end of input")

        tok = tokens[pos]

        if tok.value == "(":
            pos += 1
            items: list[Any] = []

            while True:
                if pos >= len(tokens):
                    raise ParseError(f"unclosed '(' at line {tok.line}, col {tok.col}")

                if tokens[pos].value == ")":
                    pos += 1
                    return items

                items.append(parse_one())

        if tok.value == ")":
            raise ParseError(f"unexpected ')' at line {tok.line}, col {tok.col}")

        pos += 1
        return parse_atom(tok.value)

    result = parse_one()

    if pos != len(tokens):
        tok = tokens[pos]
        raise ParseError(f"unexpected token {tok.value!r} at line {tok.line}, col {tok.col}")

    return result


def parse_atom(value: str) -> bool | int | None | str:
    if value == "true":
        return True

    if value == "false":
        return False

    if value in {"nil", "none"}:
        return None

    if value.isdigit() or (value.startswith("-") and len(value) > 1 and value[1:].isdigit()):
        return int(value)

    return value


def parse_file(path: str | Path) -> L5.Program:
    return parse_program(Path(path).read_text())


def parse_program(source: str) -> L5.Program:
    return sexp_to_program(parse_sexp(source))


def parse_expression(source: str) -> L5.Expression:
    return parse_expr(parse_sexp(source))


def expect_list(value: Any, ctx: str) -> list[Any]:
    if not isinstance(value, list):
        raise ParseError(f"expected list in {ctx}, got {value!r}")
    return value


def expect_symbol(value: Any, ctx: str) -> str:
    if not isinstance(value, str):
        raise ParseError(f"expected symbol in {ctx}, got {value!r}")
    if value == "":
        raise ParseError(f"empty symbol in {ctx}")
    return value


def expect_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParseError(f"expected integer in {ctx}, got {value!r}")
    return value


def expect_nat(value: Any, ctx: str) -> int:
    result = expect_int(value, ctx)
    if result < 0:
        raise ParseError(f"expected natural number in {ctx}, got {value!r}")
    return result


def expect_positive(value: Any, ctx: str) -> int:
    result = expect_int(value, ctx)
    if result < 1:
        raise ParseError(f"expected positive integer in {ctx}, got {value!r}")
    return result


def expect_len(xs: list[Any], length: int, ctx: str) -> None:
    if len(xs) != length:
        raise ParseError(f"{ctx} expects {length} items, got {len(xs)}")


def make_if(condition: L5.Expression, consequent: L5.Expression, otherwise: L5.Expression) -> L4.If:
    return L4.If.model_construct(
        tag="if",
        condition=condition,
        consequent=consequent,
        otherwise=otherwise,
    )


def make_function(params: list[tuple[str, L5.Type]], body: L5.Expression) -> L4.Function:
    return L4.Function.model_construct(
        tag="function",
        params=params,
        body=body,
    )


def make_call(target: L5.Expression, arguments: list[L5.Expression]) -> L4.Call:
    return L4.Call.model_construct(
        tag="call",
        target=target,
        arguments=arguments,
    )


def make_operation(operator: str, left: L5.Expression, right: L5.Expression) -> L4.Operation:
    return L4.Operation.model_construct(
        tag="operation",
        operator=operator,
        left=left,
        right=right,
    )


def make_let(bindings: list[tuple[str, L5.Type, L5.Expression]], body: L5.Expression) -> L4.Let:
    return L4.Let.model_construct(
        tag="let",
        bindings=bindings,
        body=body,
    )


def make_letrec(bindings: list[tuple[str, L5.Type, L5.Expression]], body: L5.Expression) -> L4.LetRec:
    return L4.LetRec.model_construct(
        tag="letrec",
        bindings=bindings,
        body=body,
    )


def make_bunch(expressions: list[L5.Expression]) -> L4.Bunch:
    return L4.Bunch.model_construct(
        tag="bunch",
        expressions=expressions,
    )


def make_heap_allocate(val: L5.Expression) -> L4.HeapAllocate:
    return L4.HeapAllocate.model_construct(
        tag="heapallocate",
        val=val,
    )


def make_new_pair(val1: L5.Expression, val2: L5.Expression, typeof: L5.Type) -> L4.NewPair:
    return L4.NewPair.model_construct(
        tag="newpair",
        val1=val1,
        val2=val2,
        typeof=typeof,
    )


def make_set(target: L4.Reference, index: int, value: L5.Expression) -> L4.Set:
    return L4.Set.model_construct(
        tag="set",
        target=target,
        index=index,
        value=value,
    )


def make_capsule(typeof: L5.Type, expression: L5.Expression) -> L4.Capsule:
    return L4.Capsule.model_construct(
        tag="capsule",
        typeof=typeof,
        expression=expression,
    )


def make_while(condition: L5.Expression, run: L5.Expression) -> L4.While:
    return L4.While.model_construct(
        tag="while",
        condition=condition,
        run=run,
    )


def make_for(times: int | L5.Expression, run: L5.Expression) -> L4.For:
    return L4.For.model_construct(
        tag="for",
        times=times,
        run=run,
    )


def sexp_to_program(sexp: Any) -> L5.Program:
    xs = expect_list(sexp, "program")
    if not xs:
        raise ParseError("empty program")

    tag = expect_symbol(xs[0], "program tag")
    if tag != "l5":
        raise ParseError(f"expected program tag 'l5', got {tag!r}")

    if len(xs) < 2:
        raise ParseError("program must have at least a body")

    if len(xs) == 2:
        return L5.Program(classes=[], definitions=[], body=parse_expr(xs[1]))

    classes: list[L5.ClassDef] = []
    definitions: list[tuple[str, L5.Type, L5.Expression]] = []

    for clause in xs[1:-1]:
        clause_items = expect_list(clause, "program clause")
        if not clause_items:
            raise ParseError("empty program clause")

        clause_tag = expect_symbol(clause_items[0], "program clause tag")

        if clause_tag == "classes":
            classes.extend(parse_class_def(item) for item in clause_items[1:])
        elif clause_tag == "definitions":
            definitions.extend(parse_definition(item) for item in clause_items[1:])
        else:
            raise ParseError(f"unknown program clause {clause_tag!r}")

    return L5.Program(
        classes=classes,
        definitions=definitions,
        body=parse_expr(xs[-1]),
    )


def parse_class_def(value: Any) -> L5.ClassDef:
    xs = expect_list(value, "class definition")
    if len(xs) < 2:
        raise ParseError("class definition must have at least a class tag and name")

    if expect_symbol(xs[0], "class definition tag") != "class":
        raise ParseError("class definition must start with 'class'")

    name = expect_symbol(xs[1], "class name")
    parent: str | None = None
    fields: list[L5.FieldDef] = []
    methods: list[L5.MethodDef] = []

    i = 2
    if i < len(xs) and xs[i] == "extends":
        if i + 1 >= len(xs):
            raise ParseError("'extends' must be followed by a parent class name")
        parent = expect_symbol(xs[i + 1], "parent class name")
        i += 2

    while i < len(xs):
        section = expect_list(xs[i], "class section")
        if not section:
            raise ParseError("empty class section")

        section_tag = expect_symbol(section[0], "class section tag")

        if section_tag == "extends":
            expect_len(section, 2, "extends section")
            parent = expect_symbol(section[1], "parent class name")
        elif section_tag == "fields":
            fields.extend(parse_field_def(item) for item in section[1:])
        elif section_tag == "methods":
            methods.extend(parse_method_def(item) for item in section[1:])
        else:
            raise ParseError(f"unknown class section {section_tag!r}")

        i += 1

    return L5.ClassDef(name=name, parent=parent, fields=fields, methods=methods)


def parse_field_def(value: Any) -> L5.FieldDef:
    xs = expect_list(value, "field definition")
    if not xs:
        raise ParseError("empty field definition")

    if xs[0] == "field":
        expect_len(xs, 3, "field definition")
        name = expect_symbol(xs[1], "field name")
        typeof = parse_type(xs[2])
    else:
        expect_len(xs, 2, "field definition")
        name = expect_symbol(xs[0], "field name")
        typeof = parse_type(xs[1])

    return L5.FieldDef(name=name, typeof=typeof)


def parse_method_def(value: Any) -> L5.MethodDef:
    xs = expect_list(value, "method definition")
    expect_len(xs, 5, "method definition")

    if expect_symbol(xs[0], "method definition tag") != "method":
        raise ParseError("method definition must start with 'method'")

    return L5.MethodDef(
        name=expect_symbol(xs[1], "method name"),
        parameters=parse_typed_params(xs[2]),
        returns=parse_type(xs[3]),
        body=parse_expr(xs[4]),
    )


def parse_definition(value: Any) -> tuple[str, L5.Type, L5.Expression]:
    xs = expect_list(value, "definition")
    if not xs:
        raise ParseError("empty definition")

    if xs[0] == "def":
        expect_len(xs, 4, "definition")
        return (
            expect_symbol(xs[1], "definition name"),
            parse_type(xs[2]),
            parse_expr(xs[3]),
        )

    expect_len(xs, 3, "definition")
    return (
        expect_symbol(xs[0], "definition name"),
        parse_type(xs[1]),
        parse_expr(xs[2]),
    )


def parse_typed_params(value: Any) -> list[tuple[str, L5.Type]]:
    params = expect_list(value, "typed parameters")
    result: list[tuple[str, L5.Type]] = []

    for param in params:
        item = expect_list(param, "typed parameter")
        expect_len(item, 2, "typed parameter")
        result.append((expect_symbol(item[0], "parameter name"), parse_type(item[1])))

    return result


def parse_type(value: Any) -> L5.Type:
    if isinstance(value, str):
        name = value.lower()

        if name == "int":
            return L4.Int()

        if name == "bool":
            return L4.Bool()

        if name == "void":
            return L4.Void()

        raise ParseError(f"unknown atomic type {value!r}")

    xs = expect_list(value, "type")
    if not xs:
        raise ParseError("empty type")

    head = expect_symbol(xs[0], "type tag")

    if head in {"class", "classtype", "class-type"}:
        expect_len(xs, 2, "class type")
        return L5.ClassType(name=expect_symbol(xs[1], "class type name"))

    if head == "mutable":
        expect_len(xs, 2, "mutable type")
        return L4.Mutable(oftype=parse_type(xs[1]))

    if head == "list":
        expect_len(xs, 2, "list type")
        return L4.List(typeof=parse_type(xs[1]))

    if head == "pair":
        expect_len(xs, 3, "pair type")
        return L4.Pair(type1=parse_type(xs[1]), type2=parse_type(xs[2]))

    if head in {"->", "func", "functype", "function"}:
        expect_len(xs, 3, "function type")
        parameter_types = expect_list(xs[1], "function parameter types")
        return L4.FuncType(
            parameters=[parse_type(item) for item in parameter_types],
            result=parse_type(xs[2]),
        )

    if head == "symbol":
        expect_len(xs, 3, "symbol type")
        return L4.Symbol(
            name=expect_symbol(xs[1], "symbol type name"),
            payload=parse_type(xs[2]),
        )

    raise ParseError(f"unknown type tag {head!r}")


def parse_expr(value: Any) -> L5.Expression:
    if isinstance(value, bool):
        return L4.Immediate(value=value)

    if isinstance(value, int):
        return L4.Immediate(value=value)

    if value is None:
        return L4.Immediate(value=None)

    if isinstance(value, str):
        if value == "empty":
            return L4.Empty()

        if value == "this":
            return L5.This()

        return L4.Reference(name=value)

    xs = expect_list(value, "expression")
    if not xs:
        raise ParseError("empty expression")

    head = xs[0]

    if not isinstance(head, str):
        return make_call(
            target=parse_expr(head),
            arguments=[parse_expr(item) for item in xs[1:]],
        )

    if head == "if":
        expect_len(xs, 4, "if expression")
        return make_if(
            condition=parse_expr(xs[1]),
            consequent=parse_expr(xs[2]),
            otherwise=parse_expr(xs[3]),
        )

    if head == "let":
        expect_len(xs, 3, "let expression")
        return make_let(bindings=parse_bindings(xs[1]), body=parse_expr(xs[2]))

    if head == "letrec":
        expect_len(xs, 3, "letrec expression")
        return make_letrec(bindings=parse_bindings(xs[1]), body=parse_expr(xs[2]))

    if head in {"\\", "lambda", "λ", "function"}:
        expect_len(xs, 3, "function expression")
        return make_function(params=parse_typed_params(xs[1]), body=parse_expr(xs[2]))

    if head in {"+", "-", "*", "==", "<"}:
        expect_len(xs, 3, f"operator {head!r}")
        return make_operation(operator=head, left=parse_expr(xs[1]), right=parse_expr(xs[2]))

    if head in {"&&", "||"}:
        expect_len(xs, 3, f"short-circuit operator {head!r}")
        return L5.ShortCircuit(operator=head, left=parse_expr(xs[1]), right=parse_expr(xs[2]))

    if head in {"begin", "bunch"}:
        return make_bunch(expressions=[parse_expr(item) for item in xs[1:]])

    if head == "switch":
        if len(xs) < 3:
            raise ParseError("switch expression expects a scrutinee and at least a default branch")

        scrutinee = parse_expr(xs[1])
        cases: list[L5.SwitchCase] = []
        default: L5.Expression | None = None

        for item in xs[2:]:
            clause = expect_list(item, "switch clause")
            if not clause:
                raise ParseError("empty switch clause")

            clause_tag = expect_symbol(clause[0], "switch clause tag")

            if clause_tag == "case":
                expect_len(clause, 3, "switch case")
                case_value = clause[1]

                if not isinstance(case_value, bool) and not isinstance(case_value, int):
                    raise ParseError("switch case value must be int or bool")

                cases.append(L5.SwitchCase(value=case_value, body=parse_expr(clause[2])))

            elif clause_tag == "default":
                expect_len(clause, 2, "switch default")
                default = parse_expr(clause[1])

            else:
                raise ParseError(f"unknown switch clause {clause_tag!r}")

        if default is None:
            raise ParseError("switch expression requires a default branch")

        return L5.Switch(scrutinee=scrutinee, cases=cases, default=default)

    if head == "break":
        expect_len(xs, 1, "break expression")
        return L5.Break()

    if head == "continue":
        expect_len(xs, 1, "continue expression")
        return L5.Continue()

    if head == "foreach":
        expect_len(xs, 5, "foreach expression")
        binder = expect_list(xs[1], "foreach binder")
        expect_len(binder, 2, "foreach binder")

        return L5.Foreach(
            binder=expect_symbol(binder[0], "foreach binder name"),
            typeof=parse_type(binder[1]),
            target=parse_reference_expr(xs[2], "foreach target"),
            count=expect_nat(xs[3], "foreach count"),
            run=parse_expr(xs[4]),
        )

    if head == "this":
        expect_len(xs, 1, "this expression")
        return L5.This()

    if head in {"new", "new-object", "newobject"}:
        if len(xs) < 2:
            raise ParseError("new expression expects a class name")

        return L5.NewObject(
            name=expect_symbol(xs[1], "class name"),
            arguments=[parse_expr(item) for item in xs[2:]],
        )

    if head in {".", "field", "field-access"}:
        expect_len(xs, 3, "field access")
        return L5.FieldAccess(
            target=parse_expr(xs[1]),
            field=expect_symbol(xs[2], "field name"),
        )

    if head in {"set-field!", "field-set!", "field-assign"}:
        expect_len(xs, 4, "field assignment")
        return L5.FieldAssign(
            target=parse_expr(xs[1]),
            field=expect_symbol(xs[2], "field name"),
            value=parse_expr(xs[3]),
        )

    if head in {"call-method", "method-call", ":"}:
        if len(xs) < 3:
            raise ParseError("method call expects target and method name")

        return L5.MethodCall(
            target=parse_expr(xs[1]),
            method=expect_symbol(xs[2], "method name"),
            arguments=[parse_expr(item) for item in xs[3:]],
        )

    if head == "empty":
        expect_len(xs, 1, "empty expression")
        return L4.Empty()

    if head in {"new-list", "newlist"}:
        expect_len(xs, 3, "new-list expression")
        return L4.NewList(size=expect_positive(xs[1], "new-list size"), typeof=parse_type(xs[2]))

    if head in {"new-pair", "newpair"}:
        expect_len(xs, 4, "new-pair expression")
        return make_new_pair(
            val1=parse_expr(xs[1]),
            val2=parse_expr(xs[2]),
            typeof=parse_type(xs[3]),
        )

    if head in {"heap-allocate", "heapallocate"}:
        expect_len(xs, 2, "heap-allocate expression")
        return make_heap_allocate(val=parse_expr(xs[1]))

    if head == "get":
        expect_len(xs, 3, "get expression")
        return L4.Get(
            target=parse_reference_expr(xs[1], "get target"),
            index=expect_nat(xs[2], "get index"),
        )

    if head == "set":
        expect_len(xs, 4, "set expression")
        return make_set(
            target=parse_reference_expr(xs[1], "set target"),
            index=expect_nat(xs[2], "set index"),
            value=parse_expr(xs[3]),
        )

    if head == "capsule":
        expect_len(xs, 3, "capsule expression")
        return make_capsule(typeof=parse_type(xs[1]), expression=parse_expr(xs[2]))

    if head == "while":
        expect_len(xs, 3, "while expression")
        return make_while(condition=parse_expr(xs[1]), run=parse_expr(xs[2]))

    if head == "for":
        expect_len(xs, 3, "for expression")
        times = xs[1] if isinstance(xs[1], int) and not isinstance(xs[1], bool) else parse_expr(xs[1])
        return make_for(times=times, run=parse_expr(xs[2]))

    if head == "reference":
        expect_len(xs, 2, "reference expression")
        return L4.Reference(name=expect_symbol(xs[1], "reference name"))

    if head == "immediate":
        expect_len(xs, 2, "immediate expression")
        item = xs[1]

        if not (isinstance(item, bool) or isinstance(item, int) or item is None):
            raise ParseError(f"immediate value must be bool, int, or nil, got {item!r}")

        return L4.Immediate(value=item)

    if isinstance(head, str) and head in {"/", "!=", "<=", ">", ">="}:
        raise ParseError(f"unsupported operator {head!r}")

    return make_call(
        target=parse_expr(head),
        arguments=[parse_expr(item) for item in xs[1:]],
    )


def parse_reference_expr(value: Any, ctx: str) -> L4.Reference:
    expr = parse_expr(value)

    if not isinstance(expr, L4.Reference):
        raise ParseError(f"{ctx} must be a reference, got {expr!r}")

    return expr


def parse_bindings(value: Any) -> list[tuple[str, L5.Type, L5.Expression]]:
    raw_bindings = expect_list(value, "bindings")
    bindings: list[tuple[str, L5.Type, L5.Expression]] = []

    for raw in raw_bindings:
        item = expect_list(raw, "binding")
        expect_len(item, 3, "binding")
        bindings.append(
            (
                expect_symbol(item[0], "binding name"),
                parse_type(item[1]),
                parse_expr(item[2]),
            )
        )

    return bindings