from pathlib import Path

import pytest
from L3.syntax import Program as L3Program
from L4 import syntax as L4

import L5
from L5 import syntax as L5Syntax
from L5.convert import convert_to_l3, convert_to_l4
from L5.parse import ParseError, parse_expression, parse_file, parse_program, parse_sexp


FACTORIAL_SOURCE = r"""
(l5
  (classes)
  (definitions)
  (letrec
    ((fact
      (-> (int) int)
      (\ ((n int))
        (if
          (== n 0)
          1
          (* n
            (fact (- n 1)))))))
    (fact 5)))
"""


CLASS_SOURCE = r"""
(l5
  (classes
    (class Point
      (fields
        (x int)
        (y int))
      (methods
        (method getX () int
          (. this x))
        (method setX ((v int)) void
          (set-field! this x v)))))
  (definitions
    (def p (class Point)
      (new Point 1 2)))
  (begin
    (call-method p setX 7)
    (call-method p getX)))
"""


def test_parse_sexp_basic_shape():
    actual = parse_sexp(FACTORIAL_SOURCE)

    assert actual[0] == "l5"
    assert actual[1] == ["classes"]
    assert actual[2] == ["definitions"]
    assert actual[3][0] == "letrec"


def test_parse_factorial_program_shape():
    program = parse_program(FACTORIAL_SOURCE)

    assert program.tag == "l5"
    assert program.classes == []
    assert program.definitions == []
    assert isinstance(program.body, L4.LetRec)

    binding_name, binding_type, binding_expr = program.body.bindings[0]
    assert binding_name == "fact"
    assert binding_type == L4.FuncType(parameters=[L4.Int()], result=L4.Int())
    assert isinstance(binding_expr, L4.Function)

    assert isinstance(program.body.body, L4.Call)


def test_parse_factorial_can_convert_to_l3():
    program = parse_program(FACTORIAL_SOURCE)

    actual = convert_to_l3(program)

    assert isinstance(actual, L3Program)


def test_parse_class_program_shape():
    program = parse_program(CLASS_SOURCE)

    assert len(program.classes) == 1
    cls = program.classes[0]

    assert cls.name == "Point"
    assert cls.parent is None
    assert cls.fields == [
        L5Syntax.FieldDef(name="x", typeof=L4.Int()),
        L5Syntax.FieldDef(name="y", typeof=L4.Int()),
    ]
    assert [method.name for method in cls.methods] == ["getX", "setX"]

    assert program.definitions[0][0] == "p"
    assert program.definitions[0][1] == L5Syntax.ClassType(name="Point")
    assert isinstance(program.definitions[0][2], L5Syntax.NewObject)


def test_parse_class_program_can_convert_to_l4():
    program = parse_program(CLASS_SOURCE)

    actual = convert_to_l4(program)

    assert isinstance(actual, L4.Program)
    names = [name for name, _, _ in actual.definitions]
    assert "Point_getX" in names
    assert "Point_setX" in names
    assert "p" in names
    assert isinstance(actual.body, L4.Bunch)


def test_parse_expression_atoms():
    assert parse_expression("1") == L4.Immediate(value=1)
    assert parse_expression("-1") == L4.Immediate(value=-1)
    assert parse_expression("true") == L4.Immediate(value=True)
    assert parse_expression("false") == L4.Immediate(value=False)
    assert parse_expression("nil") == L4.Immediate(value=None)
    assert parse_expression("x") == L4.Reference(name="x")
    assert parse_expression("this") == L5Syntax.This()
    assert parse_expression("empty") == L4.Empty()


def test_parse_expression_l4_nodes():
    assert parse_expression("(+ 1 2)") == L4.Operation(
        operator="+",
        left=L4.Immediate(value=1),
        right=L4.Immediate(value=2),
    )

    assert parse_expression("(if true 1 0)") == L4.If(
        condition=L4.Immediate(value=True),
        consequent=L4.Immediate(value=1),
        otherwise=L4.Immediate(value=0),
    )

    assert parse_expression("(\\ ((x int)) x)") == L4.Function(
        params=[("x", L4.Int())],
        body=L4.Reference(name="x"),
    )

    assert parse_expression("(f 1 2)") == L4.Call(
        target=L4.Reference(name="f"),
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )

    assert parse_expression("(begin 1 2)") == L4.Bunch(
        expressions=[L4.Immediate(value=1), L4.Immediate(value=2)]
    )


def test_parse_expression_l5_nodes():
    assert parse_expression("(&& true false)") == L5Syntax.ShortCircuit(
        operator="&&",
        left=L4.Immediate(value=True),
        right=L4.Immediate(value=False),
    )

    assert parse_expression("(new Point 1 2)") == L5Syntax.NewObject(
        name="Point",
        arguments=[L4.Immediate(value=1), L4.Immediate(value=2)],
    )

    assert parse_expression("(. p x)") == L5Syntax.FieldAccess(
        target=L4.Reference(name="p"),
        field="x",
    )

    assert parse_expression("(set-field! p x 3)") == L5Syntax.FieldAssign(
        target=L4.Reference(name="p"),
        field="x",
        value=L4.Immediate(value=3),
    )

    assert parse_expression("(call-method p getX)") == L5Syntax.MethodCall(
        target=L4.Reference(name="p"),
        method="getX",
        arguments=[],
    )


def test_parse_switch_break_continue_foreach():
    switch = parse_expression(
        """
        (switch x
          (case 0 10)
          (case 1 20)
          (default 30))
        """
    )
    assert isinstance(switch, L5Syntax.Switch)
    assert len(switch.cases) == 2

    assert parse_expression("(break)") == L5Syntax.Break()
    assert parse_expression("(continue)") == L5Syntax.Continue()

    foreach = parse_expression("(foreach (x int) xs 3 x)")
    assert foreach == L5Syntax.Foreach(
        binder="x",
        typeof=L4.Int(),
        target=L4.Reference(name="xs"),
        count=3,
        run=L4.Reference(name="x"),
    )


def test_parse_file(tmp_path: Path):
    source = tmp_path / "factorial.l5"
    source.write_text(FACTORIAL_SOURCE)

    actual = parse_file(source)

    assert isinstance(actual, L5Syntax.Program)
    assert isinstance(actual.body, L4.LetRec)


def test_init_exports_parser():
    assert callable(L5.parse_program)
    assert callable(L5.parse_expression)
    assert callable(L5.parse_file)
    assert callable(L5.parse_sexp)


def test_parse_rejects_bad_program_tag():
    with pytest.raises(ParseError, match="expected program tag"):
        parse_program("(l4 (classes) (definitions) 1)")


def test_parse_rejects_unknown_program_clause():
    with pytest.raises(ParseError, match="unknown program clause"):
        parse_program("(l5 (bad) 1)")


def test_parse_rejects_unclosed_list():
    with pytest.raises(ParseError, match="unclosed"):
        parse_program("(l5 (classes) (definitions) 1")


def test_parse_rejects_unsupported_operator():
    with pytest.raises(ParseError):
        parse_expression("(/ 4 2)")


def test_parse_rejects_bad_binding_shape():
    with pytest.raises(ParseError, match="binding expects 3 items"):
        parse_program("(l5 (classes) (definitions) (let ((x int)) x))")


def test_parse_rejects_bad_field_target():
    with pytest.raises(ParseError, match="get target must be a reference"):
        parse_expression("(get (+ 1 2) 0)")


def test_tokenize_handles_comment_and_parentheses():
    from L5.parse import tokenize

    tokens = tokenize("(+ 1 2) ; comment\nx")
    assert [token.value for token in tokens] == ["(", "+", "1", "2", ")", "x"]
    assert tokens[-1].line == 2
    assert tokens[-1].col == 1


def test_parse_sexp_rejects_empty_input():
    with pytest.raises(ParseError, match="unexpected end of input"):
        parse_sexp("")


def test_parse_sexp_rejects_extra_token():
    with pytest.raises(ParseError, match="unexpected token"):
        parse_sexp("1 2")


def test_parse_sexp_rejects_unexpected_close_paren():
    with pytest.raises(ParseError, match="unexpected"):
        parse_sexp(")")


def test_expect_helpers_error_paths():
    from L5.parse import expect_int, expect_list, expect_positive, expect_symbol

    with pytest.raises(ParseError, match="expected list"):
        expect_list("x", "ctx")

    with pytest.raises(ParseError, match="expected symbol"):
        expect_symbol(1, "ctx")

    with pytest.raises(ParseError, match="empty symbol"):
        expect_symbol("", "ctx")

    with pytest.raises(ParseError, match="expected integer"):
        expect_int(True, "ctx")

    with pytest.raises(ParseError, match="expected integer"):
        expect_int("1", "ctx")

    with pytest.raises(ParseError, match="expected positive"):
        expect_positive(0, "ctx")


def test_parse_program_single_body_form():
    program = parse_program("(l5 42)")

    assert program == L5Syntax.Program(
        classes=[],
        definitions=[],
        body=L4.Immediate(value=42),
    )


def test_parse_program_rejects_empty_program_list():
    with pytest.raises(ParseError, match="empty program"):
        parse_program("()")


def test_parse_program_rejects_non_list_program():
    with pytest.raises(ParseError, match="expected list"):
        parse_program("x")


def test_parse_program_rejects_empty_clause():
    with pytest.raises(ParseError, match="empty program clause"):
        parse_program("(l5 () 1)")


def test_parse_program_rejects_non_list_clause():
    with pytest.raises(ParseError, match="expected list in program clause"):
        parse_program("(l5 bad 1)")


def test_parse_program_accepts_short_form_with_classes_and_definitions():
    source = """
    (l5
      (classes)
      (definitions
        (answer int 42))
      answer)
    """
    program = parse_program(source)

    assert program.classes == []
    assert program.definitions == [("answer", L4.Int(), L4.Immediate(value=42))]
    assert program.body == L4.Reference(name="answer")


def test_parse_class_extends_inline_and_section():
    inline = parse_program(
        """
        (l5
          (classes
            (class Child extends Parent
              (fields)
              (methods)))
          (definitions)
          0)
        """
    )

    section = parse_program(
        """
        (l5
          (classes
            (class Child
              (extends Parent)
              (fields)
              (methods)))
          (definitions)
          0)
        """
    )

    assert inline.classes[0].parent == "Parent"
    assert section.classes[0].parent == "Parent"


def test_parse_class_rejects_bad_shapes():
    with pytest.raises(ParseError, match="at least a class tag and name"):
        parse_program("(l5 (classes (class)) (definitions) 0)")

    with pytest.raises(ParseError, match="must start with 'class'"):
        parse_program("(l5 (classes (struct Point)) (definitions) 0)")

    with pytest.raises(ParseError, match="'extends' must be followed"):
        parse_program("(l5 (classes (class Child extends)) (definitions) 0)")

    with pytest.raises(ParseError, match="empty class section"):
        parse_program("(l5 (classes (class Point ())) (definitions) 0)")

    with pytest.raises(ParseError, match="unknown class section"):
        parse_program("(l5 (classes (class Point (bad))) (definitions) 0)")


def test_parse_field_forms_and_errors():
    program = parse_program(
        """
        (l5
          (classes
            (class Point
              (fields
                (field x int)
                (y int))
              (methods)))
          (definitions)
          0)
        """
    )

    assert program.classes[0].fields == [
        L5Syntax.FieldDef(name="x", typeof=L4.Int()),
        L5Syntax.FieldDef(name="y", typeof=L4.Int()),
    ]

    with pytest.raises(ParseError, match="empty field definition"):
        parse_program("(l5 (classes (class C (fields ()))) (definitions) 0)")

    with pytest.raises(ParseError, match="field definition expects"):
        parse_program("(l5 (classes (class C (fields (field x)))) (definitions) 0)")


def test_parse_method_rejects_bad_shapes():
    with pytest.raises(ParseError, match="method definition expects"):
        parse_program("(l5 (classes (class C (fields) (methods (method f)))) (definitions) 0)")

    with pytest.raises(ParseError, match="method definition must start"):
        parse_program("(l5 (classes (class C (fields) (methods (fn f () int 0)))) (definitions) 0)")


def test_parse_definition_short_form_and_errors():
    program = parse_program(
        """
        (l5
          (classes)
          (definitions
            (x int 1)
            (def y bool true))
          x)
        """
    )

    assert program.definitions == [
        ("x", L4.Int(), L4.Immediate(value=1)),
        ("y", L4.Bool(), L4.Immediate(value=True)),
    ]

    with pytest.raises(ParseError, match="empty definition"):
        parse_program("(l5 (classes) (definitions ()) 0)")

    with pytest.raises(ParseError, match="definition expects"):
        parse_program("(l5 (classes) (definitions (def x int)) 0)")


def test_parse_typed_params_rejects_bad_parameter():
    with pytest.raises(ParseError, match="typed parameter expects"):
        parse_expression(r"(\ ((x int extra)) x)")


def test_parse_all_type_forms():
    assert parse_program("(l5 (classes) (definitions (x (mutable int) (heap-allocate 1))) x)").definitions[0][1] == L4.Mutable(oftype=L4.Int())

    assert parse_program("(l5 (classes) (definitions (xs (list int) (new-list 3 int))) xs)").definitions[0][1] == L4.List(typeof=L4.Int())

    assert parse_program(
        "(l5 (classes) (definitions (p (pair int bool) (new-pair 1 true (pair int bool)))) p)"
    ).definitions[0][1] == L4.Pair(type1=L4.Int(), type2=L4.Bool())

    assert parse_program("(l5 (classes) (definitions (f (func (int bool) int) 0)) 0)").definitions[0][1] == L4.FuncType(
        parameters=[L4.Int(), L4.Bool()],
        result=L4.Int(),
    )

    assert parse_program("(l5 (classes) (definitions (s (symbol Some int) 0)) 0)").definitions[0][1] == L4.Symbol(
        name="Some",
        payload=L4.Int(),
    )


def test_parse_type_rejects_bad_types():
    with pytest.raises(ParseError, match="unknown atomic type"):
        parse_program("(l5 (classes) (definitions (x string 0)) x)")

    with pytest.raises(ParseError, match="empty type"):
        parse_program("(l5 (classes) (definitions (x () 0)) x)")

    with pytest.raises(ParseError, match="unknown type tag"):
        parse_program("(l5 (classes) (definitions (x (array int) 0)) x)")

    with pytest.raises(ParseError, match="class type expects"):
        parse_program("(l5 (classes) (definitions (x (class) 0)) x)")

    with pytest.raises(ParseError, match="mutable type expects"):
        parse_program("(l5 (classes) (definitions (x (mutable int bool) 0)) x)")


def test_parse_list_pair_heap_get_set_capsule_while_for():
    assert parse_expression("(new-list 2 int)") == L4.NewList(size=2, typeof=L4.Int())

    assert parse_expression("(new-pair 1 true (pair int bool))") == L4.NewPair.model_construct(
        tag="newpair",
        val1=L4.Immediate(value=1),
        val2=L4.Immediate(value=True),
        typeof=L4.Pair(type1=L4.Int(), type2=L4.Bool()),
    )

    assert parse_expression("(heap-allocate 1)") == L4.HeapAllocate.model_construct(
        tag="heapallocate",
        val=L4.Immediate(value=1),
    )

    assert parse_expression("(get xs 0)") == L4.Get(
        target=L4.Reference(name="xs"),
        index=0,
    )

    assert parse_expression("(set xs 0 9)") == L4.Set.model_construct(
        tag="set",
        target=L4.Reference(name="xs"),
        index=0,
        value=L4.Immediate(value=9),
    )

    assert parse_expression("(capsule int 1)") == L4.Capsule.model_construct(
        tag="capsule",
        typeof=L4.Int(),
        expression=L4.Immediate(value=1),
    )

    assert parse_expression("(while true empty)") == L4.While.model_construct(
        tag="while",
        condition=L4.Immediate(value=True),
        run=L4.Empty(),
    )

    assert parse_expression("(for 3 empty)") == L4.For.model_construct(
        tag="for",
        times=3,
        run=L4.Empty(),
    )

    assert parse_expression("(for n empty)") == L4.For.model_construct(
        tag="for",
        times=L4.Reference(name="n"),
        run=L4.Empty(),
    )


def test_parse_reference_and_immediate_explicit_forms():
    assert parse_expression("(reference x)") == L4.Reference(name="x")
    assert parse_expression("(immediate 1)") == L4.Immediate(value=1)
    assert parse_expression("(immediate true)") == L4.Immediate(value=True)
    assert parse_expression("(immediate nil)") == L4.Immediate(value=None)

    with pytest.raises(ParseError, match="immediate value must be"):
        parse_expression("(immediate x)")


def test_parse_expr_rejects_empty_expression():
    with pytest.raises(ParseError, match="empty expression"):
        parse_expression("()")


def test_parse_expr_rejects_bad_arities():
    bad_cases = [
        ("(if true 1)", "if expression"),
        ("(let () 1 2)", "let expression"),
        ("(letrec () 1 2)", "letrec expression"),
        (r"(\ ((x int)) x y)", "function expression"),
        ("(+ 1)", "operator"),
        ("(&& true)", "short-circuit"),
        ("(break 1)", "break expression"),
        ("(continue 1)", "continue expression"),
        ("(this 1)", "this expression"),
        ("(. p)", "field access"),
        ("(set-field! p x)", "field assignment"),
        ("(empty 1)", "empty expression"),
        ("(new-list 1)", "new-list expression"),
        ("(new-pair 1 true)", "new-pair expression"),
        ("(heap-allocate)", "heap-allocate expression"),
        ("(get xs)", "get expression"),
        ("(set xs 0)", "set expression"),
        ("(capsule int)", "capsule expression"),
        ("(while true)", "while expression"),
        ("(for 3)", "for expression"),
        ("(reference)", "reference expression"),
        ("(immediate)", "immediate expression"),
    ]

    for source, message in bad_cases:
        with pytest.raises(ParseError, match=message):
            parse_expression(source)


def test_parse_expr_rejects_new_and_method_missing_parts():
    with pytest.raises(ParseError, match="new expression expects a class name"):
        parse_expression("(new)")

    with pytest.raises(ParseError, match="method call expects target and method name"):
        parse_expression("(call-method p)")


def test_parse_expr_rejects_invalid_numbers_for_sizes_and_indices():
    with pytest.raises(ParseError, match="expected positive integer"):
        parse_expression("(new-list 0 int)")

    with pytest.raises(ParseError, match="expected natural number"):
        parse_expression("(get xs -1)")

    with pytest.raises(ParseError, match="expected natural number"):
        parse_expression("(set xs -1 0)")


def test_parse_switch_error_paths():
    with pytest.raises(ParseError, match="expects a scrutinee"):
        parse_expression("(switch x)")

    with pytest.raises(ParseError, match="empty switch clause"):
        parse_expression("(switch x () (default 0))")

    with pytest.raises(ParseError, match="switch case"):
        parse_expression("(switch x (case 0) (default 0))")

    with pytest.raises(ParseError, match="switch case value must be int or bool"):
        parse_expression("(switch x (case foo 1) (default 0))")

    with pytest.raises(ParseError, match="switch default"):
        parse_expression("(switch x (default 0 1))")

    with pytest.raises(ParseError, match="unknown switch clause"):
        parse_expression("(switch x (else 0))")

    with pytest.raises(ParseError, match="requires a default"):
        parse_expression("(switch x (case 0 1))")


def test_parse_foreach_error_paths():
    with pytest.raises(ParseError, match="foreach expression expects"):
        parse_expression("(foreach (x int) xs 3)")

    with pytest.raises(ParseError, match="foreach binder expects"):
        parse_expression("(foreach (x int extra) xs 3 x)")

    with pytest.raises(ParseError, match="foreach target must be a reference"):
        parse_expression("(foreach (x int) (+ 1 2) 3 x)")


def test_parse_call_fallback_with_expression_target():
    actual = parse_expression("((lambda ((x int)) x) 1)")

    assert actual == L4.Call.model_construct(
        tag="call",
        target=L4.Function.model_construct(
            tag="function",
            params=[("x", L4.Int())],
            body=L4.Reference(name="x"),
        ),
        arguments=[L4.Immediate(value=1)],
    )


def test_parse_symbol_errors_in_expression_positions():
    with pytest.raises(ParseError, match="expected symbol"):
        parse_expression("(. p 1)")

    with pytest.raises(ParseError, match="expected symbol"):
        parse_expression("(call-method p 1)")

    with pytest.raises(ParseError, match="expected symbol"):
        parse_expression("(new 1)")


def test_parse_negative_integer_atom():
    assert parse_expression("-123") == L4.Immediate(value=-123)


def test_parse_successful_let_expression_hits_make_let():
    actual = parse_expression("(let ((x int 1)) x)")

    assert actual == L4.Let.model_construct(
        tag="let",
        bindings=[("x", L4.Int(), L4.Immediate(value=1))],
        body=L4.Reference(name="x"),
    )


def test_parse_program_rejects_l5_without_body():
    with pytest.raises(ParseError, match="program must have at least a body"):
        parse_program("(l5)")


def test_parse_field_assignment_aliases():
    assert parse_expression("(field-set! p x 3)") == L5Syntax.FieldAssign(
        target=L4.Reference(name="p"),
        field="x",
        value=L4.Immediate(value=3),
    )

    assert parse_expression("(field-assign p x 4)") == L5Syntax.FieldAssign(
        target=L4.Reference(name="p"),
        field="x",
        value=L4.Immediate(value=4),
    )


def test_parse_method_call_aliases_and_missing_method_name():
    assert parse_expression("(method-call p getX 1)") == L5Syntax.MethodCall(
        target=L4.Reference(name="p"),
        method="getX",
        arguments=[L4.Immediate(value=1)],
    )

    assert parse_expression("(: p getX 1)") == L5Syntax.MethodCall(
        target=L4.Reference(name="p"),
        method="getX",
        arguments=[L4.Immediate(value=1)],
    )

    with pytest.raises(ParseError, match="method call expects target and method name"):
        parse_expression("(method-call p)")


def test_parse_object_and_collection_aliases():
    assert parse_expression("(new-object Point)") == L5Syntax.NewObject(
        name="Point",
        arguments=[],
    )

    assert parse_expression("(newobject Point 1)") == L5Syntax.NewObject(
        name="Point",
        arguments=[L4.Immediate(value=1)],
    )

    assert parse_expression("(newlist 2 int)") == L4.NewList(
        size=2,
        typeof=L4.Int(),
    )

    assert parse_expression("(newpair 1 true (pair int bool))") == L4.NewPair.model_construct(
        tag="newpair",
        val1=L4.Immediate(value=1),
        val2=L4.Immediate(value=True),
        typeof=L4.Pair(type1=L4.Int(), type2=L4.Bool()),
    )

    assert parse_expression("(heapallocate 1)") == L4.HeapAllocate.model_construct(
        tag="heapallocate",
        val=L4.Immediate(value=1),
    )


def test_parse_list_form_this_and_empty():
    assert parse_expression("(this)") == L5Syntax.This()
    assert parse_expression("(empty)") == L4.Empty()