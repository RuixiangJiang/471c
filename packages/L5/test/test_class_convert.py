import pytest
from L4 import syntax as L4

from L5 import syntax as L5
from L5.class_convert import (
    access_from_ref,
    build_payload_expression,
    class_payload_type,
    class_rep_type,
    field_path,
    infer_type,
    lower_field_access,
    lower_field_assign,
    lower_method_call,
    lower_method_definition,
    lower_new_object,
    lower_type,
    rebuild_from_ref,
    same_type,
    type_error,
    wrap_let,
    wrap_with_reference,
)
from L5.inheritance_convert import MethodInfo, collect_classes
from L5.minor_convert import SequentialNameGenerator


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
            ),
            L5.MethodDef(
                name="setX",
                parameters=[("v", L4.Int())],
                returns=L4.Void(),
                body=L5.FieldAssign(
                    target=L5.This(),
                    field="x",
                    value=L4.Reference(name="v"),
                ),
            ),
        ],
    )


def make_child_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="Child",
        parent="Point",
        fields=[L5.FieldDef(name="y", typeof=L4.Int())],
        methods=[
            L5.MethodDef(
                name="getY",
                parameters=[],
                returns=L4.Int(),
                body=L5.FieldAccess(target=L5.This(), field="y"),
            ),
        ],
    )


def make_override_child() -> L5.ClassDef:
    return L5.ClassDef(
        name="OverrideChild",
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


def make_empty_class() -> L5.ClassDef:
    return L5.ClassDef(name="Empty", fields=[], methods=[])


def make_box_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="Box",
        fields=[],
        methods=[
            L5.MethodDef(
                name="noop",
                parameters=[],
                returns=L4.Void(),
                body=L4.Empty(),
            )
        ],
    )


def make_bad_method_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="BadMethod",
        fields=[],
        methods=[
            L5.MethodDef(
                name="bad",
                parameters=[],
                returns=L4.Int(),
                body=L4.Immediate(value=False),
            )
        ],
    )


def make_env():
    return collect_classes(
        [
            make_point_class(),
            make_child_class(),
            make_override_child(),
            make_empty_class(),
            make_box_class(),
        ]
    )


def recursive_lower(expression, **kwargs):
    fresh = kwargs["fresh"]
    context = kwargs["context"]
    class_env = kwargs["class_env"]
    current_class = kwargs["current_class"]

    if isinstance(expression, L5.NewObject):
        return lower_new_object(
            expression,
            fresh=fresh,
            context=context,
            class_env=class_env,
            current_class=current_class,
            lower_expr=recursive_lower,
        )
    return expression


def test_type_error_and_same_type():
    actual = type_error("boom")
    assert isinstance(actual, TypeError)
    assert str(actual) == "boom"

    assert same_type(L4.Int(), L4.Int())
    assert not same_type(L4.Int(), L4.Bool())


def test_class_payload_and_rep_type():
    env = make_env()

    point = env["Point"]
    child = env["Child"]
    empty = env["Empty"]

    assert class_payload_type(point, env) == L4.Int()
    assert class_rep_type(point, env) == L4.Mutable(oftype=L4.Int())

    assert class_payload_type(child, env) == L4.Pair(type1=L4.Int(), type2=L4.Int())
    assert class_rep_type(child, env) == L4.Mutable(
        oftype=L4.Pair(type1=L4.Int(), type2=L4.Int())
    )

    assert class_payload_type(empty, env) == L4.Void()
    assert class_rep_type(empty, env) == L4.Mutable(oftype=L4.Void())


def test_lower_type():
    env = make_env()

    assert lower_type(L4.Int(), env) == L4.Int()
    assert lower_type(L4.Bool(), env) == L4.Bool()
    assert lower_type(L4.Void(), env) == L4.Void()
    symbol = L4.Symbol.model_construct(tag="symbol", name="S", payload="S")
    assert lower_type(symbol, env) == symbol
    assert lower_type(L4.Mutable(oftype=L4.Int()), env) == L4.Mutable(oftype=L4.Int())
    assert lower_type(L4.List(typeof=L4.Int()), env) == L4.List(typeof=L4.Int())
    assert lower_type(L4.Pair(type1=L4.Int(), type2=L4.Int()), env) == L4.Pair(
        type1=L4.Int(), type2=L4.Int()
    )
    assert lower_type(
        L4.FuncType(parameters=[L4.Int()], result=L4.Bool()),
        env,
    ) == L4.FuncType(parameters=[L4.Int()], result=L4.Bool())

    assert lower_type(L5.ClassType(name="Point"), env) == L4.Mutable(oftype=L4.Int())

    with pytest.raises(TypeError, match="unknown class type"):
        lower_type(L5.ClassType(name="Missing"), env)

    with pytest.raises(TypeError, match="cannot lower type"):
        lower_type(object(), env)  # type: ignore[arg-type]


def test_wrap_let():
    actual = wrap_let(
        "x",
        L4.Int(),
        L4.Immediate(value=1),
        L4.Reference(name="x"),
    )
    assert actual == L4.Let(
        bindings=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )


def test_field_path():
    assert field_path(0, 1) == []
    assert field_path(0, 2) == [0]
    assert field_path(1, 2) == [1]
    assert field_path(0, 3) == [0]
    assert field_path(1, 3) == [1, 0]
    assert field_path(2, 3) == [1, 1]

    with pytest.raises(TypeError, match="empty class"):
        field_path(0, 0)

    with pytest.raises(TypeError, match="invalid field index"):
        field_path(-1, 2)

    with pytest.raises(TypeError, match="invalid field index"):
        field_path(3, 3)


def test_access_from_ref():
    fresh = SequentialNameGenerator()

    actual = access_from_ref(
        L4.Reference(name="root"),
        L4.Pair(type1=L4.Int(), type2=L4.Int()),
        [1],
        fresh=fresh,
    )
    assert isinstance(actual, L4.Let)

    actual = access_from_ref(
        L4.Reference(name="root"),
        L4.Int(),
        [],
        fresh=fresh,
    )
    assert actual == L4.Reference(name="root")

    with pytest.raises(TypeError, match="cannot descend"):
        access_from_ref(
            L4.Reference(name="x"),
            L4.Int(),
            [0],
            fresh=fresh,
        )


def test_rebuild_from_ref():
    fresh = SequentialNameGenerator()
    pair_type = L4.Pair(type1=L4.Int(), type2=L4.Int())

    base = rebuild_from_ref(
        L4.Reference(name="root"),
        L4.Int(),
        [],
        L4.Immediate(value=5),
        fresh=fresh,
    )
    assert base == L4.Immediate(value=5)

    actual = rebuild_from_ref(
        L4.Reference(name="root"),
        pair_type,
        [0],
        L4.Immediate(value=9),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)

    actual = rebuild_from_ref(
        L4.Reference(name="root"),
        pair_type,
        [1],
        L4.Immediate(value=9),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)

    nested_left_type = L4.Pair(
        type1=L4.Pair(type1=L4.Int(), type2=L4.Int()),
        type2=L4.Int(),
    )
    nested_left = rebuild_from_ref(
        L4.Reference(name="root"),
        nested_left_type,
        [0, 1],
        L4.Immediate(value=7),
        fresh=fresh,
    )
    assert isinstance(nested_left, L4.NewPair)

    nested_right_type = L4.Pair(
        type1=L4.Int(),
        type2=L4.Pair(type1=L4.Int(), type2=L4.Int()),
    )
    nested_right = rebuild_from_ref(
        L4.Reference(name="root"),
        nested_right_type,
        [1, 0],
        L4.Immediate(value=11),
        fresh=fresh,
    )
    assert isinstance(nested_right, L4.NewPair)

    with pytest.raises(TypeError, match="pair path step must be 0 or 1"):
        rebuild_from_ref(
            L4.Reference(name="root"),
            pair_type,
            [2],
            L4.Immediate(value=0),
            fresh=fresh,
        )

    with pytest.raises(TypeError, match="cannot rebuild"):
        rebuild_from_ref(
            L4.Reference(name="root"),
            L4.Int(),
            [0],
            L4.Immediate(value=0),
            fresh=fresh,
        )


def test_wrap_with_reference():
    actual = wrap_with_reference(
        L4.Reference(name="obj"),
        L4.Int(),
        fresh=SequentialNameGenerator(),
        prefix="tmp",
        body_builder=lambda ref: L4.Get(target=ref, index=0),
    )
    assert actual == L4.Get(target=L4.Reference(name="obj"), index=0)

    actual = wrap_with_reference(
        L4.Immediate(value=1),
        L4.Int(),
        fresh=SequentialNameGenerator(),
        prefix="tmp",
        body_builder=lambda ref: ref,
    )
    assert isinstance(actual, L4.Let)


def test_build_payload_expression():
    env = make_env()
    point = env["Point"]
    child = env["Child"]
    empty = env["Empty"]

    point_payload = build_payload_expression(
        point,
        [L4.Immediate(value=1)],
        class_env=env,
    )
    assert point_payload == L4.Immediate(value=1)

    child_payload = build_payload_expression(
        child,
        [L4.Immediate(value=1), L4.Immediate(value=2)],
        class_env=env,
    )
    assert child_payload == L4.NewPair(
        val1=L4.Immediate(value=1),
        val2=L4.Immediate(value=2),
        typeof=L4.Pair(type1=L4.Int(), type2=L4.Int()),
    )

    assert build_payload_expression(empty, [], class_env=env) == L4.Empty()

    with pytest.raises(TypeError, match="constructor argument count mismatch"):
        build_payload_expression(child, [L4.Immediate(value=1)], class_env=env)


def test_infer_type_this():
    env = make_env()
    assert infer_type(L5.This(), context={}, class_env=env, current_class="Point") == L5.ClassType(
        name="Point"
    )

    with pytest.raises(TypeError, match="this used outside"):
        infer_type(L5.This(), context={}, class_env=env, current_class=None)


def test_infer_type_new_object():
    env = make_env()

    assert infer_type(
        L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
        context={},
        class_env=env,
        current_class=None,
    ) == L5.ClassType(name="Point")

    assert infer_type(
        L5.NewObject(
            name="Child",
            arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L5.ClassType(name="Child")

    assert infer_type(
        L5.NewObject(name="Empty", arguments=[]),
        context={},
        class_env=env,
        current_class=None,
    ) == L5.ClassType(name="Empty")

    with pytest.raises(TypeError, match="unknown class"):
        infer_type(
            L5.NewObject(name="Missing", arguments=[]),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="expects 2 arguments"):
        infer_type(
            L5.NewObject(name="Child", arguments=[L4.Immediate(value=1)]),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="constructor argument"):
        infer_type(
            L5.NewObject(
                name="Child",
                arguments=[L4.Immediate(value=True), L4.Immediate(value=2)],
            ),
            context={},
            class_env=env,
            current_class=None,
        )


def test_infer_type_field_access_assign_and_method_call():
    env = make_env()
    point_ctx = {"p": L5.ClassType(name="Point")}
    child_ctx = {"c": L5.ClassType(name="Child")}
    override_ctx = {"o": L5.ClassType(name="OverrideChild")}

    assert infer_type(
        L5.FieldAccess(target=L4.Reference(name="p"), field="x"),
        context=point_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.FieldAccess(target=L4.Reference(name="c"), field="x"),
        context=child_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.FieldAccess(target=L4.Reference(name="c"), field="y"),
        context=child_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.FieldAssign(
            target=L4.Reference(name="p"),
            field="x",
            value=L4.Immediate(value=5),
        ),
        context=point_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L5.MethodCall(
            target=L4.Reference(name="p"),
            method="getX",
            arguments=[],
        ),
        context=point_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.MethodCall(
            target=L4.Reference(name="c"),
            method="getX",
            arguments=[],
        ),
        context=child_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.MethodCall(
            target=L4.Reference(name="o"),
            method="getX",
            arguments=[],
        ),
        context=override_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L5.MethodCall(
            target=L4.Reference(name="p"),
            method="setX",
            arguments=[L4.Immediate(value=3)],
        ),
        context=point_ctx,
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="field access requires class target"):
        infer_type(
            L5.FieldAccess(target=L4.Immediate(value=1), field="x"),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="field assignment requires class target"):
        infer_type(
            L5.FieldAssign(
                target=L4.Immediate(value=1),
                field="x",
                value=L4.Immediate(value=5),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="field assignment expected"):
        infer_type(
            L5.FieldAssign(
                target=L4.Reference(name="p"),
                field="x",
                value=L4.Immediate(value=False),
            ),
            context=point_ctx,
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="method call requires class target"):
        infer_type(
            L5.MethodCall(target=L4.Immediate(value=1), method="getX", arguments=[]),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="expects 1 arguments"):
        infer_type(
            L5.MethodCall(
                target=L4.Reference(name="p"),
                method="setX",
                arguments=[],
            ),
            context=point_ctx,
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="method argument expected"):
        infer_type(
            L5.MethodCall(
                target=L4.Reference(name="p"),
                method="setX",
                arguments=[L4.Immediate(value=False)],
            ),
            context=point_ctx,
            class_env=env,
            current_class=None,
        )


def test_infer_type_misc():
    env = make_env()

    assert infer_type(
        L5.ShortCircuit(
            operator="&&",
            left=L4.Immediate(value=True),
            right=L4.Immediate(value=False),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    with pytest.raises(TypeError, match="short-circuit operators require bool operands"):
        infer_type(
            L5.ShortCircuit(
                operator="&&",
                left=L4.Immediate(value=1),
                right=L4.Immediate(value=False),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L5.Switch(
            scrutinee=L4.Immediate(value=1),
            cases=[L5.SwitchCase(value=1, body=L4.Immediate(value=10))],
            default=L4.Immediate(value=0),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="switch scrutinee must be int or bool"):
        infer_type(
            L5.Switch(
                scrutinee=L4.Empty(),
                cases=[],
                default=L4.Immediate(value=0),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="all switch branches must have the same type"):
        infer_type(
            L5.Switch(
                scrutinee=L4.Immediate(value=1),
                cases=[L5.SwitchCase(value=1, body=L4.Immediate(value=True))],
                default=L4.Immediate(value=0),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(L5.Break(), context={}, class_env=env, current_class=None) == L4.Void()
    assert infer_type(L5.Continue(), context={}, class_env=env, current_class=None) == L4.Void()

    assert infer_type(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=2,
            run=L4.Reference(name="x"),
        ),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="foreach target must be a list reference"):
        infer_type(
            L5.Foreach(
                binder="x",
                typeof=L4.Int(),
                target=L4.Reference(name="ys"),
                count=1,
                run=L4.Reference(name="x"),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="foreach binder type does not match"):
        infer_type(
            L5.Foreach(
                binder="x",
                typeof=L4.Bool(),
                target=L4.Reference(name="xs"),
                count=1,
                run=L4.Reference(name="x"),
            ),
            context={"xs": L4.List(typeof=L4.Int())},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Reference(name="x"),
        context={"x": L4.Int()},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="unknown reference"):
        infer_type(
            L4.Reference(name="missing"),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Immediate(value=True),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    assert infer_type(
        L4.Immediate(value=1),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L4.Immediate(value=None),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.Operation(operator="+", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L4.Operation(operator="<", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    assert infer_type(
        L4.Operation(operator="==", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    with pytest.raises(TypeError, match="requires int operands"):
        infer_type(
            L4.Operation(operator="+", left=L4.Immediate(value=True), right=L4.Immediate(value=2)),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="== requires operands"):
        infer_type(
            L4.Operation(operator="==", left=L4.Immediate(value=True), right=L4.Immediate(value=2)),
            context={},
            class_env=env,
            current_class=None,
        )

    bogus_op = L4.Operation.model_construct(  # type: ignore[arg-type]
        tag="operation",
        operator="??",
        left=L4.Immediate(value=1),
        right=L4.Immediate(value=2),
    )
    with pytest.raises(TypeError, match="unknown operator"):
        infer_type(
            bogus_op,
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.If(
            condition=L4.Immediate(value=True),
            consequent=L4.Immediate(value=1),
            otherwise=L4.Immediate(value=0),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="if condition must be bool"):
        infer_type(
            L4.If(
                condition=L4.Immediate(value=1),
                consequent=L4.Immediate(value=1),
                otherwise=L4.Immediate(value=0),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="if branches must have the same type"):
        infer_type(
            L4.If(
                condition=L4.Immediate(value=True),
                consequent=L4.Immediate(value=1),
                otherwise=L4.Immediate(value=False),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Let(
            bindings=[("x", L4.Int(), L4.Immediate(value=1))],
            body=L4.Reference(name="x"),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="let binding"):
        infer_type(
            L4.Let(
                bindings=[("x", L4.Int(), L4.Immediate(value=False))],
                body=L4.Reference(name="x"),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.LetRec(
            bindings=[("f", L4.FuncType(parameters=[], result=L4.Void()), L4.Empty())],
            body=L4.Reference(name="f"),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.FuncType(parameters=[], result=L4.Void())

    with pytest.raises(TypeError, match="cannot infer a bare function expression"):
        infer_type(
            L4.Function(params=[], body=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="bare call inference is not implemented"):
        infer_type(
            L4.Call(target=L4.Reference(name="f"), arguments=[]),
            context={"f": L4.FuncType(parameters=[], result=L4.Void())},
            class_env=env,
            current_class=None,
        )

    assert infer_type(L4.Empty(), context={}, class_env=env, current_class=None) == L4.Void()
    assert infer_type(
        L4.NewList(size=3, typeof=L4.Int()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.List(typeof=L4.Int())

    pair_type = L4.Pair(type1=L4.Int(), type2=L4.Bool())
    assert infer_type(
        L4.NewPair(
            val1=L4.Immediate(value=1),
            val2=L4.Immediate(value=False),
            typeof=pair_type,
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == pair_type

    assert infer_type(
        L4.HeapAllocate(val=L4.Immediate(value=1)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Mutable(oftype=L4.Int())

    assert infer_type(
        L4.Get(target=L4.Reference(name="m"), index=0),
        context={"m": L4.Mutable(oftype=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L4.Get(target=L4.Reference(name="p"), index=0),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert infer_type(
        L4.Get(target=L4.Reference(name="p"), index=1),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    assert infer_type(
        L4.Get(target=L4.Reference(name="xs"), index=9),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="unknown reference"):
        infer_type(
            L4.Get(target=L4.Reference(name="missing"), index=0),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="mutable get only supports index 0"):
        infer_type(
            L4.Get(target=L4.Reference(name="m"), index=1),
            context={"m": L4.Mutable(oftype=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="pair get index must be 0 or 1"):
        infer_type(
            L4.Get(target=L4.Reference(name="p"), index=2),
            context={"p": pair_type},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="get is not supported"):
        infer_type(
            L4.Get(target=L4.Reference(name="x"), index=0),
            context={"x": L4.Int()},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Set(target=L4.Reference(name="m"), index=0, value=L4.Immediate(value=1)),
        context={"m": L4.Mutable(oftype=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.Set(target=L4.Reference(name="p"), index=0, value=L4.Immediate(value=1)),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.Set(target=L4.Reference(name="p"), index=1, value=L4.Immediate(value=False)),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.Set(target=L4.Reference(name="xs"), index=0, value=L4.Immediate(value=1)),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="unknown reference"):
        infer_type(
            L4.Set(
                target=L4.Reference(name="missing"),
                index=0,
                value=L4.Immediate(value=1),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="invalid mutable set"):
        infer_type(
            L4.Set(target=L4.Reference(name="m"), index=1, value=L4.Immediate(value=1)),
            context={"m": L4.Mutable(oftype=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="pair set index must be 0 or 1"):
        infer_type(
            L4.Set(target=L4.Reference(name="p"), index=2, value=L4.Immediate(value=1)),
            context={"p": pair_type},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="invalid pair set"):
        infer_type(
            L4.Set(target=L4.Reference(name="p"), index=1, value=L4.Immediate(value=1)),
            context={"p": pair_type},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="invalid list set"):
        infer_type(
            L4.Set(target=L4.Reference(name="xs"), index=0, value=L4.Immediate(value=False)),
            context={"xs": L4.List(typeof=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="set is not supported"):
        infer_type(
            L4.Set(target=L4.Reference(name="x"), index=0, value=L4.Immediate(value=1)),
            context={"x": L4.Int()},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=1)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="capsule expected"):
        infer_type(
            L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=False)),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.While(condition=L4.Immediate(value=True), run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="while condition must be bool"):
        infer_type(
            L4.While(condition=L4.Immediate(value=1), run=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.For(times=3, run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.For(times=L4.Immediate(value=3), run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="for times must be int"):
        infer_type(
            L4.For(times=L4.Immediate(value=False), run=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    assert infer_type(
        L4.Bunch(expressions=[]),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert infer_type(
        L4.Bunch(expressions=[L4.Empty(), L4.Immediate(value=1)]),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="cannot infer type for"):
        infer_type(
            object(),  # type: ignore[arg-type]
            context={},
            class_env=env,
            current_class=None,
        )


def test_lower_new_object():
    env = make_env()
    fresh = SequentialNameGenerator()

    actual = lower_new_object(
        L5.NewObject(name="Child", arguments=[L4.Immediate(value=1), L4.Immediate(value=2)]),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        lower_expr=recursive_lower,
    )
    assert isinstance(actual, L4.HeapAllocate)

    actual = lower_new_object(
        L5.NewObject(name="Empty", arguments=[]),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        lower_expr=recursive_lower,
    )
    assert actual == L4.HeapAllocate(val=L4.Empty())

    with pytest.raises(TypeError, match="unknown class"):
        lower_new_object(
            L5.NewObject(name="Missing", arguments=[]),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            lower_expr=recursive_lower,
        )


def test_lower_field_access_and_assign():
    env = make_env()
    fresh = SequentialNameGenerator()
    ctx = {"c": L5.ClassType(name="Child"), "p": L5.ClassType(name="Point")}

    access = lower_field_access(
        L5.FieldAccess(target=L4.Reference(name="c"), field="x"),
        fresh=fresh,
        context=ctx,
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert isinstance(access, (L4.Let, L4.Reference))

    access_nonref = lower_field_access(
        L5.FieldAccess(
            target=L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
            field="x",
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert isinstance(access_nonref, L4.Let)

    assign = lower_field_assign(
        L5.FieldAssign(
            target=L4.Reference(name="c"),
            field="y",
            value=L4.Immediate(value=7),
        ),
        fresh=fresh,
        context=ctx,
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert isinstance(assign, (L4.Let, L4.Set))

    assign_nonref = lower_field_assign(
        L5.FieldAssign(
            target=L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
            field="x",
            value=L4.Immediate(value=9),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert isinstance(assign_nonref, L4.Let)

    with pytest.raises(TypeError, match="field access requires class target"):
        lower_field_access(
            L5.FieldAccess(target=L4.Immediate(value=1), field="x"),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            infer_expr=infer_type,
            lower_expr=recursive_lower,
        )

    with pytest.raises(TypeError, match="field assignment requires class target"):
        lower_field_assign(
            L5.FieldAssign(
                target=L4.Immediate(value=1),
                field="x",
                value=L4.Immediate(value=2),
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            infer_expr=infer_type,
            lower_expr=recursive_lower,
        )


def test_lower_method_call_inherited_and_overridden():
    env = make_env()
    fresh = SequentialNameGenerator()

    inherited_call = lower_method_call(
        L5.MethodCall(target=L4.Reference(name="c"), method="getX", arguments=[]),
        fresh=fresh,
        context={"c": L5.ClassType(name="Child")},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert inherited_call == L4.Call(
        target=L4.Reference(name="Point_getX"),
        arguments=[L4.Reference(name="c")],
    )

    own_call = lower_method_call(
        L5.MethodCall(target=L4.Reference(name="c"), method="getY", arguments=[]),
        fresh=fresh,
        context={"c": L5.ClassType(name="Child")},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert own_call == L4.Call(
        target=L4.Reference(name="Child_getY"),
        arguments=[L4.Reference(name="c")],
    )

    overridden_call = lower_method_call(
        L5.MethodCall(target=L4.Reference(name="o"), method="getX", arguments=[]),
        fresh=fresh,
        context={"o": L5.ClassType(name="OverrideChild")},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert overridden_call == L4.Call(
        target=L4.Reference(name="OverrideChild_getX"),
        arguments=[L4.Reference(name="o")],
    )

    nonref_call = lower_method_call(
        L5.MethodCall(
            target=L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
            method="getX",
            arguments=[],
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert isinstance(nonref_call, L4.Let)

    with pytest.raises(TypeError, match="method call requires class target"):
        lower_method_call(
            L5.MethodCall(target=L4.Immediate(value=1), method="getX", arguments=[]),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            infer_expr=infer_type,
            lower_expr=recursive_lower,
        )


def test_lower_method_definition():
    env = make_env()

    box_method = MethodInfo(
        name="noop",
        parameters=[],
        returns=L4.Void(),
        body=L4.Empty(),
        owner="Box",
    )
    actual = lower_method_definition(
        "Box",
        box_method,
        fresh=SequentialNameGenerator(),
        class_env=env,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )
    assert actual[0] == "Box_noop"
    assert isinstance(actual[1], L4.FuncType)
    assert isinstance(actual[2], L4.Function)

    bad_env = collect_classes([make_bad_method_class()])
    bad_method = MethodInfo(
        name="bad",
        parameters=[],
        returns=L4.Int(),
        body=L4.Immediate(value=False),
        owner="BadMethod",
    )
    with pytest.raises(TypeError, match="expected return"):
        lower_method_definition(
            "BadMethod",
            bad_method,
            fresh=SequentialNameGenerator(),
            class_env=bad_env,
            infer_expr=infer_type,
            lower_expr=recursive_lower,
        )


def test_lower_method_definition_with_parameters_hits_context_extension():
    env = make_env()

    method = MethodInfo(
        name="idInt",
        parameters=[("v", L4.Int())],
        returns=L4.Int(),
        body=L4.Reference(name="v"),
        owner="Point",
    )

    actual = lower_method_definition(
        "Point",
        method,
        fresh=SequentialNameGenerator(),
        class_env=env,
        infer_expr=infer_type,
        lower_expr=recursive_lower,
    )

    assert actual[0] == "Point_idInt"
    assert isinstance(actual[1], L4.FuncType)
    assert actual[1] == L4.FuncType(
        parameters=[L4.Mutable(oftype=L4.Int()), L4.Int()],
        result=L4.Int(),
    )
    assert isinstance(actual[2], L4.Function)
    assert actual[2].params == [
        ("this", L4.Mutable(oftype=L4.Int())),
        ("v", L4.Int()),
    ]
    assert actual[2].body == L4.Reference(name="v")