import pytest
from L4 import syntax as L4

from L5 import syntax as L5
from L5.inheritance_convert import (
    ClassInfo,
    FieldInfo,
    MethodInfo,
    collect_classes,
    field_index,
    field_info,
    method_info,
)


def make_point_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="Point",
        fields=[L5.FieldDef(name="x", typeof=L4.Int())],
        methods=[
            L5.MethodDef(
                name="getX",
                parameters=[],
                returns=L4.Int(),
                body=L5.FieldAccess(target=L5.This(), field="x"),
            )
        ],
    )


def make_colored_point_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="ColoredPoint",
        parent="Point",
        fields=[L5.FieldDef(name="color", typeof=L4.Int())],
        methods=[
            L5.MethodDef(
                name="getColor",
                parameters=[],
                returns=L4.Int(),
                body=L5.FieldAccess(target=L5.This(), field="color"),
            )
        ],
    )


def make_override_child() -> L5.ClassDef:
    return L5.ClassDef(
        name="Child",
        parent="Point",
        fields=[],
        methods=[
            L5.MethodDef(
                name="getX",
                parameters=[],
                returns=L4.Int(),
                body=L5.FieldAccess(target=L5.This(), field="x"),
            )
        ],
    )


def test_collect_classes_without_inheritance():
    env = collect_classes([make_point_class()])
    assert "Point" in env
    point = env["Point"]
    assert point.name == "Point"
    assert point.parent is None
    assert [f.name for f in point.all_fields] == ["x"]
    assert set(point.methods.keys()) == {"getX"}


def test_collect_classes_with_inheritance():
    env = collect_classes([make_point_class(), make_colored_point_class()])
    child = env["ColoredPoint"]

    assert child.parent == "Point"
    assert [f.name for f in child.all_fields] == ["x", "color"]
    assert set(child.methods.keys()) == {"getX", "getColor"}

    inherited = child.methods["getX"]
    own = child.methods["getColor"]
    assert inherited.owner == "Point"
    assert own.owner == "ColoredPoint"


def test_collect_classes_with_override():
    env = collect_classes([make_point_class(), make_override_child()])
    child = env["Child"]
    resolved = child.methods["getX"]
    assert resolved.owner == "Child"


def test_field_and_method_lookup_helpers():
    env = collect_classes([make_point_class(), make_colored_point_class()])
    child = env["ColoredPoint"]

    assert field_index(child, "x") == 0
    assert field_index(child, "color") == 1
    assert field_info(child, "x") == FieldInfo(name="x", typeof=L4.Int(), owner="Point")
    assert method_info(child, "getX").owner == "Point"
    assert method_info(child, "getColor").owner == "ColoredPoint"

    with pytest.raises(TypeError, match="has no field"):
        field_index(child, "y")

    with pytest.raises(TypeError, match="has no field"):
        field_info(child, "y")

    with pytest.raises(TypeError, match="has no method"):
        method_info(child, "missing")


def test_collect_classes_duplicate_class():
    cls = make_point_class()
    with pytest.raises(TypeError, match="duplicate class definition"):
        collect_classes([cls, cls])


def test_collect_classes_unknown_parent():
    cls = L5.ClassDef(
        name="Child",
        parent="Missing",
        fields=[],
        methods=[],
    )
    with pytest.raises(TypeError, match="unknown parent class"):
        collect_classes([cls])


def test_collect_classes_cyclic_inheritance():
    a = L5.ClassDef(name="A", parent="B", fields=[], methods=[])
    b = L5.ClassDef(name="B", parent="A", fields=[], methods=[])
    with pytest.raises(TypeError, match="cyclic inheritance"):
        collect_classes([a, b])


def test_collect_classes_duplicate_field():
    cls = L5.ClassDef(
        name="Point",
        fields=[
            L5.FieldDef(name="x", typeof=L4.Int()),
            L5.FieldDef(name="x", typeof=L4.Int()),
        ],
        methods=[],
    )
    with pytest.raises(TypeError, match="duplicate field"):
        collect_classes([cls])


def test_collect_classes_inherited_field_conflict():
    parent = make_point_class()
    child = L5.ClassDef(
        name="Child",
        parent="Point",
        fields=[L5.FieldDef(name="x", typeof=L4.Int())],
        methods=[],
    )
    with pytest.raises(TypeError, match="conflicts with inherited field"):
        collect_classes([parent, child])


def test_collect_classes_duplicate_method():
    cls = L5.ClassDef(
        name="Point",
        fields=[],
        methods=[
            L5.MethodDef(name="f", parameters=[], returns=L4.Void(), body=L4.Empty()),
            L5.MethodDef(name="f", parameters=[], returns=L4.Void(), body=L4.Empty()),
        ],
    )
    with pytest.raises(TypeError, match="duplicate method"):
        collect_classes([cls])


def test_collect_classes_incompatible_override():
    parent = make_point_class()
    child = L5.ClassDef(
        name="Child",
        parent="Point",
        fields=[],
        methods=[
            L5.MethodDef(
                name="getX",
                parameters=[],
                returns=L4.Bool(),
                body=L4.Immediate(value=True),
            )
        ],
    )
    with pytest.raises(TypeError, match="incompatible signature"):
        collect_classes([parent, child])