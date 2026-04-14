from __future__ import annotations

import pytest

from L0.syntax import (
    Address,
    Allocate as Allocate0,
    Branch as Branch0,
    Call,
    Copy as Copy0,
    Halt as Halt0,
    Immediate as Immediate0,
    Load as Load0,
    Primitive as Primitive0,
    Procedure,
    Program as Program0,
    Store as Store0,
)
from L1 import close as close_module
from L1.close import close_program, free_variables
from L1.syntax import (
    Abstract,
    Allocate,
    Apply,
    Branch,
    Copy,
    Halt,
    Immediate,
    Load,
    Primitive,
    Program,
    Store,
)


def test_free_variables_excludes_abstract_bindings():
    statement = Immediate(
        destination="y",
        value=1,
        then=Abstract(
            destination="f",
            parameters=("x",),
            body=Primitive(
                destination="z",
                operator="+",
                left="x",
                right="y",
                then=Halt(value="z"),
            ),
            then=Halt(value="f"),
        ),
    )

    assert free_variables(statement) == ()


def test_free_variables_apply_deduplicates_names():
    statement = Apply(target="f", arguments=("f", "x", "f", "x"))
    assert free_variables(statement) == ("f", "x")


def test_free_variables_covers_remaining_statement_forms():
    statement = Copy(
        destination="tmp",
        source="src",
        then=Branch(
            operator="<",
            left="lhs",
            right="rhs",
            then=Allocate(
                destination="arr",
                count=2,
                then=Load(
                    destination="elt",
                    base="arr",
                    index=0,
                    then=Store(
                        base="arr",
                        index=1,
                        value="src",
                        then=Halt(value="elt"),
                    ),
                ),
            ),
            otherwise=Immediate(
                destination="k",
                value=0,
                then=Primitive(
                    destination="sum",
                    operator="+",
                    left="k",
                    right="rhs",
                    then=Halt(value="sum"),
                ),
            ),
        ),
    )

    assert free_variables(statement) == ("src", "lhs", "rhs")


def test_free_variables_invalid_statement_raises_type_error():
    with pytest.raises(TypeError, match="Unhandled L1 statement in free_variables"):
        free_variables(object())  # type: ignore[arg-type]


def test_close_lifts_simple_function():
    program = Program(
        parameters=(),
        body=Abstract(
            destination="f",
            parameters=("x",),
            body=Halt(value="x"),
            then=Halt(value="f"),
        ),
    )

    actual = close_program(program)

    assert isinstance(actual, Program0)
    assert len(actual.procedures) == 2

    lifted, entry = actual.procedures

    assert isinstance(lifted, Procedure)
    assert lifted.name.startswith("f$close")
    assert lifted.parameters[1:] == ("x",)

    assert isinstance(lifted.body, Copy0)
    assert lifted.body.destination == "f"
    assert lifted.body.source == lifted.parameters[0]
    assert lifted.body.then == Halt0(value="x")

    assert entry.name == "l0"
    assert entry.parameters == ()

    assert isinstance(entry.body, Allocate0)
    assert entry.body.destination == "f"
    assert entry.body.count == 1

    assert isinstance(entry.body.then, Address)
    assert entry.body.then.name == lifted.name

    assert isinstance(entry.body.then.then, Store0)
    assert entry.body.then.then.base == "f"
    assert entry.body.then.then.index == 0
    assert entry.body.then.then.then == Halt0(value="f")


def test_close_captures_free_variables_in_environment():
    program = Program(
        parameters=(),
        body=Immediate(
            destination="y",
            value=1,
            then=Abstract(
                destination="f",
                parameters=("x",),
                body=Primitive(
                    destination="z",
                    operator="+",
                    left="x",
                    right="y",
                    then=Halt(value="z"),
                ),
                then=Halt(value="f"),
            ),
        ),
    )

    actual = close_program(program)

    lifted, entry = actual.procedures

    assert lifted.name.startswith("f$close")

    assert isinstance(lifted.body, Copy0)
    assert lifted.body.destination == "f"
    assert lifted.body.source == lifted.parameters[0]

    assert isinstance(lifted.body.then, Load0)
    assert lifted.body.then.destination == "y"
    assert lifted.body.then.base == lifted.parameters[0]
    assert lifted.body.then.index == 1

    assert isinstance(entry.body, Immediate0)
    assert entry.body.destination == "y"
    assert entry.body.value == 1

    alloc = entry.body.then
    assert isinstance(alloc, Allocate0)
    assert alloc.destination == "f"
    assert alloc.count == 2

    addr = alloc.then
    assert isinstance(addr, Address)
    assert addr.name == lifted.name

    store_code = addr.then
    assert isinstance(store_code, Store0)
    assert store_code.base == "f"
    assert store_code.index == 0

    store_capture = store_code.then
    assert isinstance(store_capture, Store0)
    assert store_capture.base == "f"
    assert store_capture.index == 1
    assert store_capture.value == "y"
    assert store_capture.then == Halt0(value="f")


def test_close_lifts_nested_functions():
    program = Program(
        parameters=(),
        body=Immediate(
            destination="a",
            value=7,
            then=Abstract(
                destination="outer",
                parameters=("x",),
                body=Abstract(
                    destination="inner",
                    parameters=("y",),
                    body=Primitive(
                        destination="sum",
                        operator="+",
                        left="a",
                        right="y",
                        then=Halt(value="sum"),
                    ),
                    then=Halt(value="inner"),
                ),
                then=Halt(value="outer"),
            ),
        ),
    )

    actual = close_program(program)

    assert len(actual.procedures) == 3
    outer_proc, inner_proc, entry = actual.procedures

    assert outer_proc.name.startswith("outer$close")
    assert inner_proc.name.startswith("inner$close")
    assert entry.name == "l0"

    assert isinstance(outer_proc.body, Copy0)
    assert outer_proc.body.destination == "outer"
    assert outer_proc.body.source == outer_proc.parameters[0]

    assert isinstance(outer_proc.body.then, Load0)
    assert outer_proc.body.then.destination == "a"
    assert outer_proc.body.then.base == outer_proc.parameters[0]
    assert outer_proc.body.then.index == 1

    assert isinstance(inner_proc.body, Copy0)
    assert inner_proc.body.destination == "inner"
    assert inner_proc.body.source == inner_proc.parameters[0]

    assert isinstance(inner_proc.body.then, Load0)
    assert inner_proc.body.then.destination == "a"
    assert inner_proc.body.then.base == inner_proc.parameters[0]
    assert inner_proc.body.then.index == 1


def test_close_lowers_apply_to_load_then_call():
    program = Program(
        parameters=(),
        body=Abstract(
            destination="id",
            parameters=("x",),
            body=Halt(value="x"),
            then=Apply(target="id", arguments=("id",)),
        ),
    )

    actual = close_program(program)
    entry = actual.procedures[-1]

    assert isinstance(entry.body, Allocate0)

    addr = entry.body.then
    assert isinstance(addr, Address)

    store_code = addr.then
    assert isinstance(store_code, Store0)
    assert store_code.base == "id"
    assert store_code.index == 0

    load_code = store_code.then
    assert isinstance(load_code, Load0)
    assert load_code.base == "id"
    assert load_code.index == 0

    call = load_code.then
    assert isinstance(call, Call)
    assert call.arguments[0] == "id"
    assert call.arguments[1:] == ("id",)


def test_close_recursive_function_rebinds_self_from_environment():
    program = Program(
        parameters=(),
        body=Abstract(
            destination="loop",
            parameters=("x",),
            body=Apply(target="loop", arguments=("x",)),
            then=Halt(value="loop"),
        ),
    )

    actual = close_program(program)

    lifted, entry = actual.procedures

    assert lifted.name.startswith("loop$close")
    assert lifted.parameters[1:] == ("x",)

    assert isinstance(lifted.body, Copy0)
    assert lifted.body.destination == "loop"
    assert lifted.body.source == lifted.parameters[0]

    assert isinstance(lifted.body.then, Load0)
    assert lifted.body.then.base == "loop"
    assert lifted.body.then.index == 0

    assert isinstance(lifted.body.then.then, Call)
    assert lifted.body.then.then.arguments[0] == "loop"
    assert lifted.body.then.then.arguments[1:] == ("x",)

    assert entry.name == "l0"


def test_close_program_preserves_first_order_statements():
    program = Program(
        parameters=("p",),
        body=Copy(
            destination="x",
            source="p",
            then=Immediate(
                destination="y",
                value=10,
                then=Primitive(
                    destination="z",
                    operator="*",
                    left="x",
                    right="y",
                    then=Branch(
                        operator="==",
                        left="z",
                        right="y",
                        then=Allocate(
                            destination="arr",
                            count=2,
                            then=Load(
                                destination="v",
                                base="arr",
                                index=0,
                                then=Store(
                                    base="arr",
                                    index=1,
                                    value="v",
                                    then=Halt(value="v"),
                                ),
                            ),
                        ),
                        otherwise=Halt(value="x"),
                    ),
                ),
            ),
        ),
    )

    actual = close_program(program)

    assert len(actual.procedures) == 1
    entry = actual.procedures[0]
    assert entry.name == "l0"
    assert entry.parameters == ("p",)

    body = entry.body
    assert isinstance(body, Copy0)
    assert body.destination == "x"
    assert body.source == "p"

    body = body.then
    assert isinstance(body, Immediate0)
    assert body.destination == "y"
    assert body.value == 10

    body = body.then
    assert isinstance(body, Primitive0)
    assert body.destination == "z"
    assert body.operator == "*"
    assert body.left == "x"
    assert body.right == "y"

    body = body.then
    assert isinstance(body, Branch0)
    assert body.operator == "=="
    assert body.left == "z"
    assert body.right == "y"

    then_branch = body.then
    assert isinstance(then_branch, Allocate0)
    assert then_branch.destination == "arr"
    assert then_branch.count == 2

    then_branch = then_branch.then
    assert isinstance(then_branch, Load0)
    assert then_branch.destination == "v"
    assert then_branch.base == "arr"
    assert then_branch.index == 0

    then_branch = then_branch.then
    assert isinstance(then_branch, Store0)
    assert then_branch.base == "arr"
    assert then_branch.index == 1
    assert then_branch.value == "v"
    assert then_branch.then == Halt0(value="v")

    assert body.otherwise == Halt0(value="x")


def test_close_statement_invalid_statement_raises_type_error():
    fresh = close_module._FreshNames()
    lifted = close_module._LiftedProcedures()

    with pytest.raises(TypeError, match="Unhandled L1 statement in close conversion"):
        close_module._close_statement(object(), lifted, fresh)  # type: ignore[arg-type]


def test_close_program_invalid_program_raises_type_error():
    with pytest.raises(TypeError, match="Unhandled L1 program in close_program"):
        close_program(object())  # type: ignore[arg-type]