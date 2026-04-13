import L5
from L4 import syntax as L4
from L5 import syntax as L5Syntax


def test_init_exports():
    assert callable(L5.convert_to_l3)
    assert callable(L5.convert_to_l4)
    assert callable(L5.dummy_parse)


def test_class_type():
    actual = L5Syntax.ClassType(name="Point")
    assert actual.tag == "classtype"
    assert actual.name == "Point"


def test_field_def():
    actual = L5Syntax.FieldDef(name="x", typeof=L4.Int())
    assert actual.tag == "fielddef"
    assert actual.name == "x"
    assert actual.typeof == L4.Int()


def test_method_def():
    actual = L5Syntax.MethodDef(
        name="getX",
        parameters=[],
        returns=L4.Int(),
        body=L5Syntax.FieldAccess(target=L5Syntax.This(), field="x"),
    )
    assert actual.tag == "methoddef"
    assert actual.name == "getX"
    assert actual.parameters == []
    assert actual.returns == L4.Int()


def test_class_def_without_parent():
    actual = L5Syntax.ClassDef(
        name="Point",
        fields=[L5Syntax.FieldDef(name="x", typeof=L4.Int())],
        methods=[],
    )
    assert actual.tag == "classdef"
    assert actual.name == "Point"
    assert actual.parent is None


def test_class_def_with_parent():
    actual = L5Syntax.ClassDef(
        name="ColoredPoint",
        parent="Point",
        fields=[L5Syntax.FieldDef(name="color", typeof=L4.Int())],
        methods=[],
    )
    assert actual.tag == "classdef"
    assert actual.parent == "Point"


def test_program():
    actual = L5Syntax.Program(
        classes=[],
        definitions=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )
    assert actual.tag == "l5"
    assert actual.classes == []
    assert actual.definitions == [("x", L4.Int(), L4.Immediate(value=1))]
    assert actual.body == L4.Reference(name="x")


def test_short_circuit():
    actual = L5Syntax.ShortCircuit(
        operator="&&",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )
    assert actual.tag == "shortcircuit"
    assert actual.operator == "&&"


def test_switch_case_and_switch():
    case = L5Syntax.SwitchCase(value=1, body=L4.Immediate(value=10))
    actual = L5Syntax.Switch(
        scrutinee=L4.Reference(name="x"),
        cases=[case],
        default=L4.Immediate(value=0),
    )
    assert case.tag == "switchcase"
    assert actual.tag == "switch"
    assert actual.cases == [case]


def test_break_continue():
    assert L5Syntax.Break().tag == "break"
    assert L5Syntax.Continue().tag == "continue"


def test_foreach():
    actual = L5Syntax.Foreach(
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


def test_this():
    actual = L5Syntax.This()
    assert actual.tag == "this"


def test_new_object():
    actual = L5Syntax.NewObject(
        name="Point",
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )
    assert actual.tag == "newobject"
    assert actual.name == "Point"
    assert actual.arguments == [L4.Immediate(value=1), L4.Immediate(value=2)]


def test_field_access():
    actual = L5Syntax.FieldAccess(target=L4.Reference(name="p"), field="x")
    assert actual.tag == "fieldaccess"
    assert actual.target == L4.Reference(name="p")
    assert actual.field == "x"


def test_field_assign():
    actual = L5Syntax.FieldAssign(
        target=L4.Reference(name="p"),
        field="x",
        value=L4.Immediate(value=5),
    )
    assert actual.tag == "fieldassign"
    assert actual.target == L4.Reference(name="p")
    assert actual.field == "x"
    assert actual.value == L4.Immediate(value=5)


def test_method_call():
    actual = L5Syntax.MethodCall(
        target=L4.Reference(name="p"),
        method="getX",
        arguments=[],
    )
    assert actual.tag == "methodcall"
    assert actual.target == L4.Reference(name="p")
    assert actual.method == "getX"
    assert actual.arguments == []