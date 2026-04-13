import pytest
from L3.syntax import Program as L3Program
from L4 import syntax as L4

from L5 import convert_to_l3, convert_to_l4, dummy_parse
from L5 import syntax as L5
from L5.convert import _lower_expression
from L5.inheritance_convert import collect_classes
from L5.minor_convert import LoopContext, SequentialNameGenerator


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


def make_child_class() -> L5.ClassDef:
    return L5.ClassDef(
        name="Child",
        parent="Point",
        fields=[L5.FieldDef(name="y", typeof=L4.Int())],
        methods=[],
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


def make_env():
    return collect_classes([make_point_class(), make_child_class(), make_override_child()])


def test_dummy_parse():
    actual = dummy_parse("hello")
    assert actual.classes == []
    assert actual.definitions == [("hello", L4.Void(), L4.Empty())]
    assert actual.body == L4.Empty()


def test_lower_expression_this_and_loop_controls():
    env = make_env()
    fresh = SequentialNameGenerator()

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


def test_lower_expression_class_features():
    env = make_env()
    fresh = SequentialNameGenerator()

    new_obj = _lower_expression(
        L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(new_obj, L4.HeapAllocate)

    access = _lower_expression(
        L5.FieldAccess(target=L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]), field="x"),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(access, L4.Let)

    assign = _lower_expression(
        L5.FieldAssign(
            target=L5.NewObject(name="Point", arguments=[L4.Immediate(value=1)]),
            field="x",
            value=L4.Immediate(value=3),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(assign, L4.Let)

    inherited_call = _lower_expression(
        L5.MethodCall(
            target=L4.Reference(name="c"),
            method="getX",
            arguments=[],
        ),
        fresh=fresh,
        context={"c": L5.ClassType(name="Child")},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert inherited_call == L4.Call(
        target=L4.Reference(name="Point_getX"),
        arguments=[L4.Reference(name="c")],
    )

    overridden_call = _lower_expression(
        L5.MethodCall(
            target=L4.Reference(name="o"),
            method="getX",
            arguments=[],
        ),
        fresh=fresh,
        context={"o": L5.ClassType(name="OverrideChild")},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert overridden_call == L4.Call(
        target=L4.Reference(name="OverrideChild_getX"),
        arguments=[L4.Reference(name="o")],
    )


def test_lower_expression_minor_features():
    env = make_env()
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
    assert isinstance(actual, L4.If)

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
    assert isinstance(actual, L4.If)

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

    actual = _lower_expression(
        L5.Switch(
            scrutinee=L4.Immediate(value=1),
            cases=[L5.SwitchCase(value=1, body=L4.Immediate(value=10))],
            default=L4.Immediate(value=0),
        ),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(actual, L4.Let)

    actual = _lower_expression(
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
    assert isinstance(actual, L4.Let)
    assert actual.body == L4.Empty()

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


def test_lower_expression_loops_and_l4_nodes():
    env = make_env()
    fresh = SequentialNameGenerator()

    actual = _lower_expression(
        L4.While(condition=L4.Immediate(value=True), run=L4.Empty()),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(actual, L4.LetRec)

    actual = _lower_expression(
        L4.For(times=3, run=L4.Empty()),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(actual, L4.LetRec)

    actual = _lower_expression(
        L4.For(times=L4.Reference(name="n"), run=L4.Empty()),
        fresh=fresh,
        context={"n": L4.Int()},
        class_env=env,
        current_class=None,
        loop=None,
    )
    assert isinstance(actual, L4.LetRec)

    loop = LoopContext(break_flag="breakX", continue_flag="continueX")
    actual = _lower_expression(
        L4.Bunch(expressions=[L4.Immediate(value=1), L4.Immediate(value=2)]),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=loop,
    )
    assert isinstance(actual, L4.Bunch)
    assert isinstance(actual.expressions[1], L4.If)

    assert _lower_expression(
        L4.Bunch(expressions=[]),
        fresh=fresh,
        context={},
        class_env=env,
        current_class=None,
        loop=None,
    ) == L4.Empty()

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
            L4.Call(target=L4.Reference(name="f"), arguments=[L4.Immediate(value=1)]),
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

    assert _lower_expression(
        L4.Reference(name="x"),
        fresh=fresh,
        context={},
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

    with pytest.raises(TypeError, match="unhandled L5 expression"):
        _lower_expression(
            object(),  # type: ignore[arg-type]
            fresh=fresh,
            context={},
            class_env=env,
            current_class=None,
            loop=None,
        )


def test_convert_to_l4_with_inheritance_and_minor_features():
    program = L5.Program(
        classes=[make_point_class(), make_child_class()],
        definitions=[
            (
                "c",
                L5.ClassType(name="Child"),
                L5.NewObject(
                    name="Child",
                    arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
                ),
            ),
            (
                "b",
                L4.Bool(),
                L5.ShortCircuit(
                    operator="&&",
                    left=L4.Immediate(value=True),
                    right=L4.Immediate(value=False),
                ),
            ),
        ],
        body=L5.MethodCall(
            target=L4.Reference(name="c"),
            method="getX",
            arguments=[],
        ),
    )

    actual = convert_to_l4(program)

    assert isinstance(actual, L4.Program)
    names = [name for name, _, _ in actual.definitions]
    assert "Point_getX" in names
    assert "c" in names
    assert "b" in names
    assert actual.body == L4.Call(
        target=L4.Reference(name="Point_getX"),
        arguments=[L4.Reference(name="c")],
    )


def test_convert_to_l4_overridden_method_call():
    program = L5.Program(
        classes=[make_point_class(), make_override_child()],
        definitions=[
            (
                "o",
                L5.ClassType(name="OverrideChild"),
                L5.NewObject(
                    name="OverrideChild",
                    arguments=[L4.Immediate(value=1)],
                ),
            )
        ],
        body=L5.MethodCall(
            target=L4.Reference(name="o"),
            method="getX",
            arguments=[],
        ),
    )

    actual = convert_to_l4(program)
    assert actual.body == L4.Call(
        target=L4.Reference(name="OverrideChild_getX"),
        arguments=[L4.Reference(name="o")],
    )


def test_convert_to_l4_definition_type_mismatch():
    program = L5.Program(
        classes=[],
        definitions=[("x", L4.Int(), L4.Immediate(value=False))],
        body=L4.Empty(),
    )
    with pytest.raises(TypeError, match="definition 'x' expected"):
        convert_to_l4(program)


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

    def fake_lower_method_definition(class_name, method, *, fresh, class_env, infer_expr, lower_expr):
        return (
            "DUPLICATE_NAME",
            L4.FuncType(parameters=[], result=L4.Void()),
            L4.Function(params=[], body=L4.Empty()),
        )

    import L5.convert as convert_mod

    monkeypatch.setattr(convert_mod, "lower_method_definition", fake_lower_method_definition)

    with pytest.raises(TypeError, match="duplicate generated method name"):
        convert_to_l4(program)


def test_convert_to_l3():
    program = L5.Program(classes=[], definitions=[], body=L4.Empty())
    actual = convert_to_l3(program)
    assert isinstance(actual, L3Program)