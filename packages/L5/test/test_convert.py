import pytest
from L3.syntax import Program as L3Program
from L4 import syntax as L4

from L5 import convert_to_l3, convert_to_l4, dummy_parse
from L5 import syntax as L5
from L5.convert import (
    ClassInfo,
    FieldInfo,
    LoopContext,
    MethodInfo,
    SequentialNameGenerator,
    _access_from_ref,
    _build_payload_expression,
    _class_payload_type,
    _class_rep_type,
    _collect_classes,
    _field_index,
    _field_info,
    _field_path,
    _guarded_void,
    _infer_type,
    _lower_expression,
    _lower_type,
    _method_info,
    _rebuild_from_ref,
    _same_type,
    _type_error,
    _wrap_let,
    _wrap_with_reference,
)


def make_point_class() -> L5.ClassDef:
    return L5.ClassDef(
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
                body=L5.FieldAccess(target=L5.This(), field="x"),
            ),
            L5.MethodDef(
                name="move_x",
                parameters=[("dx", L4.Int())],
                returns=L4.Void(),
                body=L5.FieldAssign(
                    target=L5.This(),
                    field="x",
                    value=L4.Reference(name="dx"),
                ),
            ),
        ],
    )


def make_point_env():
    return _collect_classes([make_point_class()])


def test_dummy_parse_returns_minimal_program():
    actual = dummy_parse("hello")
    assert isinstance(actual, L5.Program)
    assert actual.classes == []
    assert actual.definitions == [("hello", L4.Void(), L4.Empty())]
    assert actual.body == L4.Empty()


def test_sequential_name_generator_counts_per_prefix():
    fresh = SequentialNameGenerator()
    assert fresh("x") == "x0"
    assert fresh("x") == "x1"
    assert fresh("y") == "y0"


def test_same_type_and_type_error():
    assert _same_type(L4.Int(), L4.Int())
    assert not _same_type(L4.Int(), L4.Bool())
    actual = _type_error("boom")
    assert isinstance(actual, TypeError)
    assert str(actual) == "boom"


def test_collect_classes():
    env = make_point_env()
    assert "Point" in env
    point = env["Point"]
    assert point.name == "Point"
    assert [f.name for f in point.fields] == ["x", "y"]
    assert set(point.methods.keys()) == {"sum", "move_x"}


def test_collect_classes_duplicate_class():
    cls = make_point_class()
    with pytest.raises(TypeError, match="duplicate class definition"):
        _collect_classes([cls, cls])


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
        _collect_classes([cls])


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
        _collect_classes([cls])


def test_field_lookup_helpers():
    env = make_point_env()
    point = env["Point"]

    assert _field_index(point, "x") == 0
    assert _field_index(point, "y") == 1
    assert _field_info(point, "x") == FieldInfo(name="x", typeof=L4.Int())
    assert _method_info(point, "sum").name == "sum"

    with pytest.raises(TypeError, match="has no field"):
        _field_index(point, "z")
    with pytest.raises(TypeError, match="has no field"):
        _field_info(point, "z")
    with pytest.raises(TypeError, match="has no method"):
        _method_info(point, "nope")


def test_lower_type_for_regular_and_class_types():
    env = make_point_env()

    assert _lower_type(L4.Int(), env) == L4.Int()
    assert _lower_type(L4.Bool(), env) == L4.Bool()
    assert _lower_type(L4.Void(), env) == L4.Void()

    actual = _lower_type(L5.ClassType(name="Point"), env)
    assert actual == L4.Mutable(
        oftype=L4.Pair(
            type1=L4.Int(),
            type2=L4.Int(),
        )
    )

    with pytest.raises(TypeError, match="unknown class type"):
        _lower_type(L5.ClassType(name="Missing"), env)

    bogus = object()
    with pytest.raises(TypeError, match="cannot lower type"):
        _lower_type(bogus, env)  # type: ignore[arg-type]


def test_class_payload_and_rep_type():
    env = make_point_env()
    point = env["Point"]

    assert _class_payload_type(point, env) == L4.Pair(type1=L4.Int(), type2=L4.Int())
    assert _class_rep_type(point, env) == L4.Mutable(
        oftype=L4.Pair(type1=L4.Int(), type2=L4.Int())
    )

    empty = ClassInfo(name="Empty", fields=[], methods={})
    assert _class_payload_type(empty, env) == L4.Void()
    assert _class_rep_type(empty, env) == L4.Mutable(oftype=L4.Void())


def test_wrap_let():
    actual = _wrap_let(
        "x",
        L4.Int(),
        L4.Immediate(value=1),
        L4.Reference(name="x"),
    )
    assert actual == L4.Let(
        bindings=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )


def test_guarded_void_without_loop_returns_same_expression():
    expr = L4.Immediate(value=1)
    assert _guarded_void(expr, loop=None) == expr


def test_guarded_void_with_loop_wraps_in_if():
    expr = L4.Empty()
    loop = LoopContext(break_flag="b", continue_flag="c")

    actual = _guarded_void(expr, loop=loop)

    assert isinstance(actual, L4.If)
    assert actual.consequent == L4.Empty()
    assert actual.otherwise == expr


def test_field_path():
    assert _field_path(0, 1) == []
    assert _field_path(0, 2) == [0]
    assert _field_path(1, 2) == [1, 0]
    assert _field_path(0, 3) == [0]
    assert _field_path(1, 3) == [1, 0]
    assert _field_path(2, 3) == [1, 1, 0]

    with pytest.raises(TypeError, match="empty class"):
        _field_path(0, 0)
    with pytest.raises(TypeError, match="invalid field index"):
        _field_path(1, 1)


def test_access_from_ref():
    fresh = SequentialNameGenerator()
    actual = _access_from_ref(
        L4.Reference(name="root"),
        L4.Pair(type1=L4.Int(), type2=L4.Int()),
        [0],
        fresh=fresh,
    )
    assert isinstance(actual, L4.Let)

    with pytest.raises(TypeError, match="cannot descend"):
        _access_from_ref(L4.Reference(name="x"), L4.Int(), [0], fresh=fresh)


def test_rebuild_from_ref():
    fresh = SequentialNameGenerator()
    actual = _rebuild_from_ref(
        L4.Reference(name="root"),
        L4.Pair(type1=L4.Int(), type2=L4.Int()),
        [0],
        L4.Immediate(value=99),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)

    with pytest.raises(TypeError, match="pair path step must be 0 or 1"):
        _rebuild_from_ref(
            L4.Reference(name="root"),
            L4.Pair(type1=L4.Int(), type2=L4.Int()),
            [2],
            L4.Immediate(value=0),
            fresh=fresh,
        )

    with pytest.raises(TypeError, match="cannot rebuild"):
        _rebuild_from_ref(
            L4.Reference(name="root"),
            L4.Int(),
            [0],
            L4.Immediate(value=0),
            fresh=fresh,
        )


def test_wrap_with_reference_reference_case():
    fresh = SequentialNameGenerator()
    actual = _wrap_with_reference(
        L4.Reference(name="obj"),
        L4.Int(),
        fresh=fresh,
        prefix="tmp",
        body_builder=lambda ref: L4.Get(target=ref, index=0),
    )
    assert actual == L4.Get(target=L4.Reference(name="obj"), index=0)


def test_wrap_with_reference_non_reference_case():
    fresh = SequentialNameGenerator()
    actual = _wrap_with_reference(
        L4.Immediate(value=1),
        L4.Int(),
        fresh=fresh,
        prefix="tmp",
        body_builder=lambda ref: ref,
    )
    assert isinstance(actual, L4.Let)
    assert actual.bindings[0][0] == "tmp0"


def test_build_payload_expression():
    env = make_point_env()
    point = env["Point"]

    actual = _build_payload_expression(
        point,
        [L4.Immediate(value=1), L4.Immediate(value=2)],
        class_env=env,
    )
    assert actual == L4.NewPair(
        val1=L4.Immediate(value=1),
        val2=L4.Immediate(value=2),
        typeof=L4.Pair(type1=L4.Int(), type2=L4.Int()),
    )

    empty = ClassInfo(name="Empty", fields=[], methods={})
    assert _build_payload_expression(empty, [], class_env=env) == L4.Empty()

    with pytest.raises(TypeError, match="constructor argument count mismatch"):
        _build_payload_expression(point, [L4.Immediate(value=1)], class_env=env)


def test_infer_type_for_this():
    env = make_point_env()
    assert _infer_type(L5.This(), context={}, class_env=env, current_class="Point") == L5.ClassType(
        name="Point"
    )
    with pytest.raises(TypeError, match="this used outside"):
        _infer_type(L5.This(), context={}, class_env=env, current_class=None)


def test_infer_type_for_new_object():
    env = make_point_env()
    actual = _infer_type(
        L5.NewObject(
            name="Point",
            arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
        ),
        context={},
        class_env=env,
        current_class=None,
    )
    assert actual == L5.ClassType(name="Point")

    with pytest.raises(TypeError, match="unknown class"):
        _infer_type(
            L5.NewObject(name="Missing", arguments=[]),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="expects 2 arguments"):
        _infer_type(
            L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="constructor argument"):
        _infer_type(
            L5.NewObject(
                name="Point",
                arguments=[L4.Immediate(value=True), L4.Immediate(value=2)],
            ),
            context={},
            class_env=env,
            current_class=None,
        )


def test_infer_type_for_field_access_and_assign():
    env = make_point_env()
    context = {"p": L5.ClassType(name="Point")}

    assert _infer_type(
        L5.FieldAccess(target=L4.Reference(name="p"), field="x"),
        context=context,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert _infer_type(
        L5.FieldAssign(
            target=L4.Reference(name="p"),
            field="x",
            value=L4.Immediate(value=5),
        ),
        context=context,
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="field access requires class target"):
        _infer_type(
            L5.FieldAccess(target=L4.Immediate(value=1), field="x"),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="field assignment requires class target"):
        _infer_type(
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
        _infer_type(
            L5.FieldAssign(
                target=L4.Reference(name="p"),
                field="x",
                value=L4.Immediate(value=False),
            ),
            context=context,
            class_env=env,
            current_class=None,
        )


def test_infer_type_for_method_call():
    env = make_point_env()
    context = {"p": L5.ClassType(name="Point")}

    assert _infer_type(
        L5.MethodCall(
            target=L4.Reference(name="p"),
            method="sum",
            arguments=[],
        ),
        context=context,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert _infer_type(
        L5.MethodCall(
            target=L4.Reference(name="p"),
            method="move_x",
            arguments=[L4.Immediate(value=3)],
        ),
        context=context,
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="method call requires class target"):
        _infer_type(
            L5.MethodCall(
                target=L4.Immediate(value=1),
                method="sum",
                arguments=[],
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="expects 1 arguments"):
        _infer_type(
            L5.MethodCall(
                target=L4.Reference(name="p"),
                method="move_x",
                arguments=[],
            ),
            context=context,
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="method argument expected"):
        _infer_type(
            L5.MethodCall(
                target=L4.Reference(name="p"),
                method="move_x",
                arguments=[L4.Immediate(value=False)],
            ),
            context=context,
            class_env=env,
            current_class=None,
        )


def test_infer_type_short_circuit_and_switch():
    env = make_point_env()

    assert _infer_type(
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
        _infer_type(
            L5.ShortCircuit(
                operator="&&",
                left=L4.Immediate(value=1),
                right=L4.Immediate(value=False),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
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
        _infer_type(
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
        _infer_type(
            L5.Switch(
                scrutinee=L4.Immediate(value=1),
                cases=[L5.SwitchCase(value=1, body=L4.Immediate(value=True))],
                default=L4.Immediate(value=0),
            ),
            context={},
            class_env=env,
            current_class=None,
        )


def test_infer_type_foreach():
    env = make_point_env()
    context = {"xs": L4.List(typeof=L4.Int())}

    assert _infer_type(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=2,
            run=L4.Reference(name="x"),
        ),
        context=context,
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="foreach target must be a list reference"):
        _infer_type(
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
        _infer_type(
            L5.Foreach(
                binder="x",
                typeof=L4.Bool(),
                target=L4.Reference(name="xs"),
                count=1,
                run=L4.Reference(name="x"),
            ),
            context=context,
            class_env=env,
            current_class=None,
        )


def test_infer_type_reference_immediate_operation_if():
    env = make_point_env()
    context = {"x": L4.Int(), "b": L4.Bool()}

    assert _infer_type(L4.Reference(name="x"), context=context, class_env=env, current_class=None) == L4.Int()

    with pytest.raises(TypeError, match="unknown reference"):
        _infer_type(L4.Reference(name="missing"), context={}, class_env=env, current_class=None)

    assert _infer_type(L4.Immediate(value=True), context={}, class_env=env, current_class=None) == L4.Bool()
    assert _infer_type(L4.Immediate(value=1), context={}, class_env=env, current_class=None) == L4.Int()
    assert _infer_type(L4.Immediate(value=None), context={}, class_env=env, current_class=None) == L4.Void()

    assert _infer_type(
        L4.Operation(operator="+", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert _infer_type(
        L4.Operation(operator="<", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    assert _infer_type(
        L4.Operation(operator="==", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    with pytest.raises(TypeError, match="requires int operands"):
        _infer_type(
            L4.Operation(operator="+", left=L4.Immediate(value=True), right=L4.Immediate(value=2)),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="== requires operands"):
        _infer_type(
            L4.Operation(operator="==", left=L4.Immediate(value=True), right=L4.Immediate(value=2)),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="requires int operands"):
        _infer_type(
            L4.Operation(operator="*", left=L4.Immediate(value=True), right=L4.Immediate(value=2)),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
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
        _infer_type(
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
        _infer_type(
            L4.If(
                condition=L4.Immediate(value=True),
                consequent=L4.Immediate(value=1),
                otherwise=L4.Immediate(value=False),
            ),
            context={},
            class_env=env,
            current_class=None,
        )


def test_infer_type_let_letrec_and_function_call_errors():
    env = make_point_env()

    assert _infer_type(
        L4.Let(
            bindings=[("x", L4.Int(), L4.Immediate(value=1))],
            body=L4.Reference(name="x"),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="let binding"):
        _infer_type(
            L4.Let(
                bindings=[("x", L4.Int(), L4.Immediate(value=False))],
                body=L4.Reference(name="x"),
            ),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.LetRec(
            bindings=[("f", L4.FuncType(parameters=[], result=L4.Void()), L4.Empty())],
            body=L4.Reference(name="f"),
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.FuncType(parameters=[], result=L4.Void())

    with pytest.raises(TypeError, match="cannot infer a bare function expression"):
        _infer_type(
            L4.Function(params=[], body=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="bare call inference is not implemented"):
        _infer_type(
            L4.Call(target=L4.Reference(name="f"), arguments=[]),
            context={"f": L4.FuncType(parameters=[], result=L4.Void())},
            class_env=env,
            current_class=None,
        )


def test_infer_type_empty_newlist_newpair_heap_get_set_capsule_loops_bunch():
    env = make_point_env()

    assert _infer_type(L4.Empty(), context={}, class_env=env, current_class=None) == L4.Void()
    assert _infer_type(
        L4.NewList(size=3, typeof=L4.Int()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.List(typeof=L4.Int())

    pair_type = L4.Pair(type1=L4.Int(), type2=L4.Bool())
    assert _infer_type(
        L4.NewPair(
            val1=L4.Immediate(value=1),
            val2=L4.Immediate(value=False),
            typeof=pair_type,
        ),
        context={},
        class_env=env,
        current_class=None,
    ) == pair_type

    assert _infer_type(
        L4.HeapAllocate(val=L4.Immediate(value=1)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Mutable(oftype=L4.Int())

    assert _infer_type(
        L4.Get(target=L4.Reference(name="m"), index=0),
        context={"m": L4.Mutable(oftype=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert _infer_type(
        L4.Get(target=L4.Reference(name="p"), index=0),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    assert _infer_type(
        L4.Get(target=L4.Reference(name="p"), index=1),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Bool()

    assert _infer_type(
        L4.Get(target=L4.Reference(name="xs"), index=9),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="unknown reference"):
        _infer_type(
            L4.Get(target=L4.Reference(name="missing"), index=0),
            context={},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="mutable get only supports index 0"):
        _infer_type(
            L4.Get(target=L4.Reference(name="m"), index=1),
            context={"m": L4.Mutable(oftype=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="pair get index must be 0 or 1"):
        _infer_type(
            L4.Get(target=L4.Reference(name="p"), index=2),
            context={"p": pair_type},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="get is not supported"):
        _infer_type(
            L4.Get(target=L4.Reference(name="x"), index=0),
            context={"x": L4.Int()},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.Set(target=L4.Reference(name="m"), index=0, value=L4.Immediate(value=1)),
        context={"m": L4.Mutable(oftype=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert _infer_type(
        L4.Set(target=L4.Reference(name="p"), index=0, value=L4.Immediate(value=1)),
        context={"p": pair_type},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert _infer_type(
        L4.Set(target=L4.Reference(name="xs"), index=0, value=L4.Immediate(value=1)),
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="invalid mutable set"):
        _infer_type(
            L4.Set(target=L4.Reference(name="m"), index=1, value=L4.Immediate(value=1)),
            context={"m": L4.Mutable(oftype=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="invalid pair set"):
        _infer_type(
            L4.Set(target=L4.Reference(name="p"), index=1, value=L4.Immediate(value=1)),
            context={"p": pair_type},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="invalid list set"):
        _infer_type(
            L4.Set(target=L4.Reference(name="xs"), index=0, value=L4.Immediate(value=False)),
            context={"xs": L4.List(typeof=L4.Int())},
            class_env=env,
            current_class=None,
        )

    with pytest.raises(TypeError, match="set is not supported"):
        _infer_type(
            L4.Set(target=L4.Reference(name="x"), index=0, value=L4.Immediate(value=1)),
            context={"x": L4.Int()},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=1)),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="capsule expected"):
        _infer_type(
            L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=False)),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.While(condition=L4.Immediate(value=True), run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="while condition must be bool"):
        _infer_type(
            L4.While(condition=L4.Immediate(value=1), run=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.For(times=3, run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert _infer_type(
        L4.For(times=L4.Immediate(value=3), run=L4.Empty()),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    with pytest.raises(TypeError, match="for times must be int"):
        _infer_type(
            L4.For(times=L4.Immediate(value=False), run=L4.Empty()),
            context={},
            class_env=env,
            current_class=None,
        )

    assert _infer_type(
        L4.Bunch(expressions=[]),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Void()

    assert _infer_type(
        L4.Bunch(expressions=[L4.Empty(), L4.Immediate(value=1)]),
        context={},
        class_env=env,
        current_class=None,
    ) == L4.Int()

    with pytest.raises(TypeError, match="cannot infer type for"):
        _infer_type(object(), context={}, class_env=env, current_class=None)  # type: ignore[arg-type]


def test_lower_expression_this_newobject_fieldaccess_fieldassign_methodcall():
    env = make_point_env()
    fresh = SequentialNameGenerator()
    context = {"p": L5.ClassType(name="Point")}

    assert _lower_expression(
        L5.This(),
        fresh=fresh,
        context={},
        class_env=env,
        current_class="Point",
        loop=None,
    ) == L4.Reference(name="this")

    with pytest.raises(ValueError, match="this used outside"):
        _lower_expression(
            L5.This(),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )

    new_obj = _lower_expression(
        L5.NewObject(
            name="Point",
            arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert new_obj == L4.HeapAllocate(
        val=L4.NewPair(
            val1=L4.Immediate(value=1),
            val2=L4.Immediate(value=2),
            typeof=L4.Pair(type1=L4.Int(), type2=L4.Int()),
        )
    )

    field_access = _lower_expression(
        L5.FieldAccess(target=L4.Reference(name="p"), field="x"),
        fresh=fresh,
        context=context,
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(field_access, L4.Let)

    field_assign = _lower_expression(
        L5.FieldAssign(
            target=L4.Reference(name="p"),
            field="x",
            value=L4.Immediate(value=7),
        ),
        fresh=fresh,
        context=context,
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(field_assign, L4.Let)

    method_call = _lower_expression(
        L5.MethodCall(
            target=L4.Reference(name="p"),
            method="sum",
            arguments=[],
        ),
        fresh=fresh,
        context=context,
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert method_call == L4.Call(
        target=L4.Reference(name="Point_sum"),
        arguments=[L4.Reference(name="p")],
    )


def test_lower_expression_short_circuit_switch_break_continue_foreach():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L5.ShortCircuit(
            operator="&&",
            left=L4.Immediate(value=True),
            right=L4.Immediate(value=False),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert actual == L4.If(
        condition=L4.Immediate(value=True),
        consequent=L4.Immediate(value=False),
        otherwise=L4.Immediate(value=False),
    )

    actual = _lower_expression(
        L5.Switch(
            scrutinee=L4.Immediate(value=True),
            cases=[L5.SwitchCase(value=True, body=L4.Immediate(value=1))],
            default=L4.Immediate(value=0),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(actual, L4.Let)

    with pytest.raises(ValueError, match="break used outside"):
        _lower_expression(
            L5.Break(),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )

    with pytest.raises(ValueError, match="continue used outside"):
        _lower_expression(
            L5.Continue(),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )

    loop = LoopContext(break_flag="b", continue_flag="c")
    assert _lower_expression(
        L5.Break(),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=loop,
    ) == L4.Set(target=L4.Reference(name="b"), index=0, value=L4.Immediate(value=True))

    assert _lower_expression(
        L5.Continue(),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=loop,
    ) == L4.Set(target=L4.Reference(name="c"), index=0, value=L4.Immediate(value=True))

    foreach = _lower_expression(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=0,
            run=L4.Reference(name="x"),
        ),
        fresh=fresh,
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(foreach, L4.Let)
    assert foreach.body == L4.Empty()


def test_lower_expression_loops_and_common_l4_nodes():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    while_ex = _lower_expression(
        L4.While(condition=L4.Immediate(value=True), run=L4.Empty()),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(while_ex, L4.LetRec)

    for_ex = _lower_expression(
        L4.For(times=3, run=L4.Empty()),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(for_ex, L4.LetRec)

    assert _lower_expression(
        L4.Reference(name="x"),
        fresh=fresh,
        context={"x": L4.Int()},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.Reference(name="x")

    assert _lower_expression(
        L4.Immediate(value=1),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.Immediate(value=1)

    assert _lower_expression(
        L4.Empty(),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.Empty()

    assert _lower_expression(
        L4.Get(target=L4.Reference(name="x"), index=0),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.Get(target=L4.Reference(name="x"), index=0)

    assert _lower_expression(
        L4.NewList(size=3, typeof=L4.Int()),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.NewList(size=3, typeof=L4.Int())

    assert isinstance(
        _lower_expression(
            L4.Bunch(expressions=[]),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Empty,
    )

    assert isinstance(
        _lower_expression(
            L4.Let(bindings=[("x", L4.Int(), L4.Immediate(value=1))], body=L4.Reference(name="x")),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Let,
    )

    assert isinstance(
        _lower_expression(
            L4.LetRec(
                bindings=[("f", L4.FuncType(parameters=[], result=L4.Void()), L4.Empty())],
                body=L4.Reference(name="f"),
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.LetRec,
    )

    assert isinstance(
        _lower_expression(
            L4.Function(params=[("x", L4.Int())], body=L4.Reference(name="x")),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Function,
    )

    assert isinstance(
        _lower_expression(
            L4.If(
                condition=L4.Immediate(value=True),
                consequent=L4.Immediate(value=1),
                otherwise=L4.Immediate(value=0),
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.If,
    )

    assert isinstance(
        _lower_expression(
            L4.Operation(operator="+", left=L4.Immediate(value=1), right=L4.Immediate(value=2)),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Operation,
    )

    assert isinstance(
        _lower_expression(
            L4.Call(target=L4.Reference(name="f"), arguments=[]),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Call,
    )

    assert isinstance(
        _lower_expression(
            L4.HeapAllocate(val=L4.Immediate(value=1)),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.HeapAllocate,
    )

    assert isinstance(
        _lower_expression(
            L4.NewPair(
                val1=L4.Immediate(value=1),
                val2=L4.Immediate(value=2),
                typeof=L4.Pair(type1=L4.Int(), type2=L4.Int()),
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.NewPair,
    )

    assert isinstance(
        _lower_expression(
            L4.Set(target=L4.Reference(name="m"), index=0, value=L4.Immediate(value=1)),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Set,
    )

    assert isinstance(
        _lower_expression(
            L4.Capsule(typeof=L4.Int(), expression=L4.Immediate(value=1)),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        ),
        L4.Capsule,
    )

    with pytest.raises(TypeError, match="unhandled L5 expression"):
        _lower_expression(
            object(),  # type: ignore[arg-type]
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_convert_to_l4_generates_class_methods_and_body():
    point = make_point_class()
    program = L5.Program(
        classes=[point],
        definitions=[
            (
                "p",
                L5.ClassType(name="Point"),
                L5.NewObject(
                    name="Point",
                    arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
                ),
            )
        ],
        body=L5.MethodCall(
            target=L4.Reference(name="p"),
            method="sum",
            arguments=[],
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual, L4.Program)
    assert len(actual.definitions) == 3

    names = [name for name, _, _ in actual.definitions]
    assert "Point_sum" in names
    assert "Point_move_x" in names
    assert "p" in names

    body = actual.body
    assert body == L4.Call(
        target=L4.Reference(name="Point_sum"),
        arguments=[L4.Reference(name="p")],
    )


def test_convert_to_l4_definition_type_mismatch():
    program = L5.Program(
        classes=[],
        definitions=[("x", L4.Int(), L4.Immediate(value=False))],
        body=L4.Empty(),
    )
    with pytest.raises(TypeError, match="definition 'x' expected"):
        convert_to_l4(program)


def test_convert_to_l4_method_return_mismatch():
    bad = L5.ClassDef(
        name="Bad",
        fields=[],
        methods=[
            L5.MethodDef(
                name="f",
                parameters=[],
                returns=L4.Int(),
                body=L4.Immediate(value=False),
            )
        ],
    )
    program = L5.Program(classes=[bad], definitions=[], body=L4.Empty())
    with pytest.raises(TypeError, match="expected return"):
        convert_to_l4(program)


def test_convert_to_l4_duplicate_generated_method_name():
    c1 = L5.ClassDef(
        name="A",
        fields=[],
        methods=[L5.MethodDef(name="f", parameters=[], returns=L4.Void(), body=L4.Empty())],
    )
    c2 = L5.ClassDef(
        name="A",
        fields=[],
        methods=[L5.MethodDef(name="f", parameters=[], returns=L4.Void(), body=L4.Empty())],
    )
    program = L5.Program(classes=[c1, c2], definitions=[], body=L4.Empty())
    with pytest.raises(TypeError, match="duplicate class definition"):
        convert_to_l4(program)


def test_convert_to_l3_returns_l3_program():
    program = L5.Program(classes=[], definitions=[], body=L4.Empty())
    actual = convert_to_l3(program)
    assert isinstance(actual, L3Program)


def test_infer_type_break_and_continue_return_void():
    env = make_point_env()
    assert _infer_type(L5.Break(), context={}, class_env=env, current_class=None) == L4.Void()
    assert _infer_type(L5.Continue(), context={}, class_env=env, current_class=None) == L4.Void()


def test_infer_type_unknown_operator_via_model_construct():
    env = make_point_env()
    bogus = L4.Operation.model_construct(  # type: ignore[arg-type]
        tag="operation",
        operator="??",
        left=L4.Immediate(value=1),
        right=L4.Immediate(value=2),
    )
    with pytest.raises(TypeError, match="unknown operator"):
        _infer_type(bogus, context={}, class_env=env, current_class=None)


def test_lower_short_circuit_invalid_operator():
    env = make_point_env()
    fresh = SequentialNameGenerator()
    bogus = L5.ShortCircuit.model_construct(  # type: ignore[arg-type]
        tag="shortcircuit",
        operator="??",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )
    with pytest.raises(ValueError, match="unknown short-circuit operator"):
        _lower_expression(
            bogus,
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_lower_expression_new_object_unknown_class():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    with pytest.raises(TypeError, match="unknown class"):
        _lower_expression(
            L5.NewObject(name="Missing", arguments=[]),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_lower_expression_field_access_and_assign_non_class_target():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    with pytest.raises(TypeError, match="field access requires class target"):
        _lower_expression(
            L5.FieldAccess(target=L4.Immediate(value=1), field="x"),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )

    with pytest.raises(TypeError, match="field assignment requires class target"):
        _lower_expression(
            L5.FieldAssign(
                target=L4.Immediate(value=1),
                field="x",
                value=L4.Immediate(value=2),
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_lower_expression_method_call_non_class_target():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    with pytest.raises(TypeError, match="method call requires class target"):
        _lower_expression(
            L5.MethodCall(
                target=L4.Immediate(value=1),
                method="sum",
                arguments=[],
            ),
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_lower_expression_field_access_assign_and_method_call_with_non_reference_target():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    # non-reference target forces _wrap_with_reference to take the Let branch
    obj_expr = L5.NewObject(
        name="Point",
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )

    access = _lower_expression(
        L5.FieldAccess(target=obj_expr, field="x"),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(access, L4.Let)

    assign = _lower_expression(
        L5.FieldAssign(
            target=obj_expr,
            field="x",
            value=L4.Immediate(value=9),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(assign, L4.Let)

    call = _lower_expression(
        L5.MethodCall(
            target=obj_expr,
            method="sum",
            arguments=[],
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(call, L4.Let)


def test_lower_expression_foreach_nonempty_shape():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L5.Foreach(
            binder="x",
            typeof=L4.Int(),
            target=L4.Reference(name="xs"),
            count=2,
            run=L4.Reference(name="x"),
        ),
        fresh=fresh,
        context={"xs": L4.List(typeof=L4.Int())},
        class_env=env,
        current_class=None,
        loop=None,
    )

    assert isinstance(actual, L4.Let)
    assert isinstance(actual.body, L4.Bunch)
    assert len(actual.body.expressions) == 2


def test_lower_expression_bunch_with_loop_guarding():
    env = make_point_env()
    fresh = SequentialNameGenerator()
    loop = LoopContext(break_flag="breakX", continue_flag="continueX")

    actual = _lower_expression(
        L4.Bunch(
            expressions=[
                L4.Immediate(value=1),
                L4.Immediate(value=2),
                L4.Immediate(value=3),
            ]
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=loop,
    )

    assert isinstance(actual, L4.Bunch)
    assert actual.expressions[0] == L4.Immediate(value=1)
    assert isinstance(actual.expressions[1], L4.If)
    assert isinstance(actual.expressions[2], L4.If)


def test_lower_expression_bare_l4_call_with_arguments():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L4.Call(
            target=L4.Reference(name="f"),
            arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert actual == L4.Call(
        target=L4.Reference(name="f"),
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )


def test_convert_to_l4_with_definition_referencing_class_method_result():
    point = make_point_class()
    program = L5.Program(
        classes=[point],
        definitions=[
            (
                "p",
                L5.ClassType(name="Point"),
                L5.NewObject(
                    name="Point",
                    arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
                ),
            ),
            (
                "v",
                L4.Int(),
                L5.MethodCall(
                    target=L4.Reference(name="p"),
                    method="sum",
                    arguments=[],
                ),
            ),
        ],
        body=L4.Reference(name="v"),
    )

    actual = convert_to_l4(program)
    assert isinstance(actual, L4.Program)
    assert actual.body == L4.Reference(name="v")


def test_convert_to_l4_unknown_class_type_in_definition():
    program = L5.Program(
        classes=[],
        definitions=[
            (
                "p",
                L5.ClassType(name="Missing"),
                L4.Empty(),
            )
        ],
        body=L4.Empty(),
    )

    with pytest.raises(TypeError, match="definition 'p' expected"):
        convert_to_l4(program)


def test_infer_type_set_unknown_reference():
    env = make_point_env()

    with pytest.raises(TypeError, match="unknown reference 'missing'"):
        _infer_type(
            L4.Set(
                target=L4.Reference(name="missing"),
                index=0,
                value=L4.Immediate(value=1),
            ),
            context={},
            class_env=env,
            current_class=None,
        )


def test_rebuild_from_ref_base_case_and_left_nested_case():
    fresh = SequentialNameGenerator()

    # hit line 625: if not path: return new_value
    base = _rebuild_from_ref(
        L4.Reference(name="root"),
        L4.Int(),
        [],
        L4.Immediate(value=9),
        fresh=fresh,
    )
    assert base == L4.Immediate(value=9)

    # hit lines 634-635: step == 0 and rest
    nested_type = L4.Pair(
        type1=L4.Pair(type1=L4.Int(), type2=L4.Int()),
        type2=L4.Int(),
    )
    actual = _rebuild_from_ref(
        L4.Reference(name="root"),
        nested_type,
        [0, 0],
        L4.Immediate(value=7),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)


def test_rebuild_from_ref_right_nested_case():
    fresh = SequentialNameGenerator()

    nested_type = L4.Pair(
        type1=L4.Int(),
        type2=L4.Pair(type1=L4.Int(), type2=L4.Int()),
    )
    actual = _rebuild_from_ref(
        L4.Reference(name="root"),
        nested_type,
        [1, 0],
        L4.Immediate(value=11),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)


def test_rebuild_from_ref_right_direct_case():
    fresh = SequentialNameGenerator()

    pair_type = L4.Pair(type1=L4.Int(), type2=L4.Int())
    actual = _rebuild_from_ref(
        L4.Reference(name="root"),
        pair_type,
        [1],
        L4.Immediate(value=22),
        fresh=fresh,
    )
    assert isinstance(actual, L4.NewPair)


def test_lower_expression_short_circuit_or_branch():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L5.ShortCircuit(
            operator="||",
            left=L4.Immediate(value=False),
            right=L4.Immediate(value=True),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )

    assert actual == L4.If(
        condition=L4.Immediate(value=False),
        consequent=L4.Immediate(value=True),
        otherwise=L4.Immediate(value=True),
    )


def test_lower_expression_for_with_expression_times_branch():
    env = make_point_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L4.For(
            times=L4.Reference(name="n"),
            run=L4.Empty(),
        ),
        fresh=fresh,
        context={"n": L4.Int()},
        class_env=env,
        current_class=None,
        loop=None,
    )

    assert isinstance(actual, L4.LetRec)


def test_convert_to_l4_duplicate_generated_method_name_via_monkeypatch(monkeypatch):
    c1 = L5.ClassDef(
        name="A",
        fields=[],
        methods=[L5.MethodDef(name="f", parameters=[], returns=L4.Void(), body=L4.Empty())],
    )
    c2 = L5.ClassDef(
        name="B",
        fields=[],
        methods=[L5.MethodDef(name="g", parameters=[], returns=L4.Void(), body=L4.Empty())],
    )
    program = L5.Program(classes=[c1, c2], definitions=[], body=L4.Empty())

    def fake_lower_method_definition(class_name, method, *, fresh, class_env):
        return (
            "DUPLICATE_NAME",
            L4.FuncType(parameters=[], result=L4.Void()),
            L4.Function(params=[], body=L4.Empty()),
        )

    import L5.convert as convert_mod

    monkeypatch.setattr(convert_mod, "_lower_method_definition", fake_lower_method_definition)

    with pytest.raises(TypeError, match="duplicate generated method name"):
        convert_to_l4(program)


