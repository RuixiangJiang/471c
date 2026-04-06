from L4 import syntax as L4
from L5 import syntax as L5


def test_class_type():
    actual = L5.ClassType(name="Point")
    assert actual.tag == "classtype"
    assert actual.name == "Point"


def test_field_def():
    actual = L5.FieldDef(name="x", typeof=L4.Int())
    assert actual.tag == "fielddef"
    assert actual.name == "x"
    assert actual.typeof == L4.Int()


def test_method_def():
    actual = L5.MethodDef(
        name="sum",
        parameters=[("dx", L4.Int())],
        returns=L4.Int(),
        body=L4.Reference(name="dx"),
    )
    assert actual.tag == "methoddef"
    assert actual.name == "sum"
    assert actual.parameters == [("dx", L4.Int())]
    assert actual.returns == L4.Int()
    assert actual.body == L4.Reference(name="dx")


def test_class_def():
    actual = L5.ClassDef(
        name="Point",
        fields=[
            L5.FieldDef(name="x", typeof=L4.Int()),
            L5.FieldDef(name="y", typeof=L4.Int()),
        ],
        methods=[
            L5.MethodDef(
                name="sum",
                parameters=[],
                returns=L4.Int(),
                body=L4.Immediate(value=0),
            )
        ],
    )
    assert actual.tag == "classdef"
    assert actual.name == "Point"
    assert len(actual.fields) == 2
    assert len(actual.methods) == 1


def test_program():
    actual = L5.Program(
        classes=[],
        definitions=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )
    assert actual.tag == "l5"
    assert actual.classes == []
    assert actual.definitions == [("x", L4.Int(), L4.Immediate(value=1))]
    assert actual.body == L4.Reference(name="x")


def test_short_circuit():
    actual = L5.ShortCircuit(
        operator="&&",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )
    assert actual.tag == "shortcircuit"
    assert actual.operator == "&&"


def test_switch_case_and_switch():
    case = L5.SwitchCase(value=1, body=L4.Immediate(value=10))
    actual = L5.Switch(
        scrutinee=L4.Reference(name="x"),
        cases=[case],
        default=L4.Immediate(value=0),
    )
    assert case.tag == "switchcase"
    assert actual.tag == "switch"
    assert actual.cases == [case]


def test_break_continue():
    assert L5.Break().tag == "break"
    assert L5.Continue().tag == "continue"


def test_foreach():
    actual = L5.Foreach(
        binder="x",
        typeof=L4.Int(),
        target=L4.Reference(name="xs"),
        count=3,
        run=L4.Reference(name="x"),
    )
    assert actual.tag == "foreach"
    assert actual.binder == "x"
    assert actual.typeof == L4.Int()
    assert actual.target == L4.Reference(name="xs")
    assert actual.count == 3
    assert actual.run == L4.Reference(name="x")


def test_this():
    actual = L5.This()
    assert actual.tag == "this"


def test_new_object():
    actual = L5.NewObject(
        name="Point",
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )
    assert actual.tag == "newobject"
    assert actual.name == "Point"
    assert actual.arguments == [L4.Immediate(value=1), L4.Immediate(value=2)]


def test_field_access():
    actual = L5.FieldAccess(
        target=L4.Reference(name="p"),
        field="x",
    )
    assert actual.tag == "fieldaccess"
    assert actual.target == L4.Reference(name="p")
    assert actual.field == "x"


def test_field_assign():
    actual = L5.FieldAssign(
        target=L4.Reference(name="p"),
        field="x",
        value=L4.Immediate(value=5),
    )
    assert actual.tag == "fieldassign"
    assert actual.target == L4.Reference(name="p")
    assert actual.field == "x"
    assert actual.value == L4.Immediate(value=5)


def test_method_call():
    actual = L5.MethodCall(
        target=L4.Reference(name="p"),
        method="sum",
        arguments=[L4.Immediate(value=1)],
    )
    assert actual.tag == "methodcall"
    assert actual.target == L4.Reference(name="p")
    assert actual.method == "sum"
    assert actual.arguments == [L4.Immediate(value=1)]