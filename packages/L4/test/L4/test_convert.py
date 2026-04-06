import pytest
from L3.syntax import (
    Abstract,
    Allocate,
    Apply,
    Begin,
    Branch,
    Immediate,
    Let,
    LetRec,
    Load,
    Primitive,
    Program,
    Reference,
    Store,
)
from L4 import syntax as L4
from L4.convert import SequentialNameGenerator, check_expression, convert_to_l3, process_expression


def test_fibonacci():
    program = L4.Program(
        definitions=[
            (
                "fibo",
                L4.FuncType(parameters=[L4.Int()], result=L4.Int()),
                L4.Capsule(
                    typeof=L4.FuncType(parameters=[L4.Int()], result=L4.Int()),
                    expression=L4.Function(
                        params=[("n", L4.Int())],
                        body=L4.If(
                            condition=L4.Operation(
                                operator="<", left=L4.Reference(name="n"), right=L4.Immediate(value=2)
                            ),
                            consequent=L4.Reference(name="n"),
                            otherwise=L4.Operation(
                                operator="+",
                                left=L4.Call(
                                    target=L4.Reference(name="fibo"),
                                    arguments=[
                                        L4.Operation(
                                            operator="-", left=L4.Reference(name="n"), right=L4.Immediate(value=1)
                                        )
                                    ],
                                ),
                                right=L4.Call(
                                    target=L4.Reference(name="fibo"),
                                    arguments=[
                                        L4.Operation(
                                            operator="-", left=L4.Reference(name="n"), right=L4.Immediate(value=2)
                                        )
                                    ],
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name="fibo"), arguments=[L4.Immediate(value=10)]),
    )
    actual = convert_to_l3(program=program)
    expected = Program(
        parameters=[],
        body=Let(
            bindings=[
                (
                    "fibo",
                    Abstract(
                        parameters=["n"],
                        body=Branch(
                            operator="==",
                            left=Immediate(value=1),
                            right=Branch(
                                operator="<",
                                left=Reference(name="n"),
                                right=Immediate(value=2),
                                consequent=Immediate(value=1),
                                otherwise=Immediate(value=0),
                            ),
                            consequent=Reference(name="n"),
                            otherwise=Primitive(
                                operator="+",
                                left=Apply(
                                    target=Reference(name="fibo"),
                                    arguments=[
                                        Primitive(
                                            operator="-",
                                            left=Reference(name="n"),
                                            right=Immediate(value=1),
                                        )
                                    ],
                                ),
                                right=Apply(
                                    target=Reference(name="fibo"),
                                    arguments=[
                                        Primitive(
                                            operator="-",
                                            left=Reference(name="n"),
                                            right=Immediate(value=2),
                                        )
                                    ],
                                ),
                            ),
                        ),
                    ),
                )
            ],
            body=Apply(
                target=Reference(name="fibo"),
                arguments=[Immediate(value=10)],
            ),
        ),
    )

    assert actual == expected

    # fibonacci loop
    program_fibo_with_loop = L4.Program(
        definitions=[
            (
                "fibo",
                L4.FuncType(parameters=[L4.Int()], result=L4.Int()),
                L4.Function(
                    params=[("n", L4.Int())],
                    body=L4.Let(
                        bindings=[
                            ("a", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0))),
                            ("b", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=1))),
                            ("c", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0))),
                        ],
                        body=L4.Bunch(
                            expressions=[
                                L4.For(
                                    times=L4.Reference(name="n"),
                                    run=L4.Bunch(
                                        expressions=[
                                            L4.Set(
                                                target=L4.Reference(name="c"),
                                                index=0,
                                                value=L4.Operation(
                                                    operator="+",
                                                    left=L4.Get(target=L4.Reference(name="a"), index=0),
                                                    right=L4.Get(target=L4.Reference(name="b"), index=0),
                                                ),
                                            ),
                                            L4.Set(
                                                target=L4.Reference(name="a"),
                                                index=0,
                                                value=L4.Get(target=L4.Reference(name="b"), index=0),
                                            ),
                                            L4.Set(
                                                target=L4.Reference(name="b"),
                                                index=0,
                                                value=L4.Get(target=L4.Reference(name="c"), index=0),
                                            ),
                                        ]
                                    ),
                                ),
                                L4.Get(target=L4.Reference(name="a"), index=0),
                            ]
                        ),
                    ),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name="fibo"), arguments=[L4.Immediate(value=10)]),
    )

    actual = convert_to_l3(program=program_fibo_with_loop)
    expected = Program(
        parameters=[],
        body=Let(
            bindings=[
                (
                    "fibo",
                    Abstract(
                        parameters=["n"],
                        body=Let(
                            bindings=[
                                (
                                    "a",
                                    Let(
                                        bindings=[
                                            ("heapallocateval0", Immediate(value=0)),
                                            ("heapallocate0", Allocate(count=1)),
                                        ],
                                        body=Begin(
                                            effects=[
                                                Store(
                                                    base=Reference(name="heapallocate0"),
                                                    index=0,
                                                    value=Reference(name="heapallocateval0"),
                                                )
                                            ],
                                            value=Reference(name="heapallocate0"),
                                        ),
                                    ),
                                ),
                                (
                                    "b",
                                    Let(
                                        bindings=[
                                            ("heapallocateval1", Immediate(value=1)),
                                            ("heapallocate1", Allocate(count=1)),
                                        ],
                                        body=Begin(
                                            effects=[
                                                Store(
                                                    base=Reference(name="heapallocate1"),
                                                    index=0,
                                                    value=Reference(name="heapallocateval1"),
                                                )
                                            ],
                                            value=Reference(name="heapallocate1"),
                                        ),
                                    ),
                                ),
                                (
                                    "c",
                                    Let(
                                        bindings=[
                                            ("heapallocateval2", Immediate(value=0)),
                                            ("heapallocate2", Allocate(count=1)),
                                        ],
                                        body=Begin(
                                            effects=[
                                                Store(
                                                    base=Reference(name="heapallocate2"),
                                                    index=0,
                                                    value=Reference(name="heapallocateval2"),
                                                )
                                            ],
                                            value=Reference(name="heapallocate2"),
                                        ),
                                    ),
                                ),
                            ],
                            body=Begin(
                                effects=[
                                    LetRec(
                                        bindings=[
                                            (
                                                "for_counter0",
                                                Let(
                                                    bindings=[
                                                        ("mutableval0", Reference(name="n")),
                                                        ("mutable0", Allocate(count=1)),
                                                    ],
                                                    body=Begin(
                                                        effects=[
                                                            Store(
                                                                base=Reference(name="mutable0"),
                                                                index=0,
                                                                value=Reference(name="mutableval0"),
                                                            )
                                                        ],
                                                        value=Reference(name="mutable0"),
                                                    ),
                                                ),
                                            ),
                                            (
                                                "for0",
                                                Abstract(
                                                    parameters=[],
                                                    body=Branch(
                                                        operator="==",
                                                        left=Immediate(value=1),
                                                        right=Branch(
                                                            operator="<",
                                                            left=Immediate(value=0),
                                                            right=Load(
                                                                base=Reference(name="for_counter0"),
                                                                index=0,
                                                            ),
                                                            consequent=Immediate(value=1),
                                                            otherwise=Immediate(value=0),
                                                        ),
                                                        consequent=Begin(
                                                            effects=[
                                                                Store(
                                                                    base=Reference(name="for_counter0"),
                                                                    index=0,
                                                                    value=Primitive(
                                                                        operator="-",
                                                                        left=Load(
                                                                            base=Reference(name="for_counter0"),
                                                                            index=0,
                                                                        ),
                                                                        right=Immediate(value=1),
                                                                    ),
                                                                ),
                                                                Begin(
                                                                    effects=[
                                                                        Store(
                                                                            base=Reference(name="c"),
                                                                            index=0,
                                                                            value=Primitive(
                                                                                operator="+",
                                                                                left=Load(
                                                                                    base=Reference(name="a"),
                                                                                    index=0,
                                                                                ),
                                                                                right=Load(
                                                                                    base=Reference(name="b"),
                                                                                    index=0,
                                                                                ),
                                                                            ),
                                                                        ),
                                                                        Store(
                                                                            base=Reference(name="a"),
                                                                            index=0,
                                                                            value=Load(
                                                                                base=Reference(name="b"),
                                                                                index=0,
                                                                            ),
                                                                        ),
                                                                    ],
                                                                    value=Store(
                                                                        base=Reference(name="b"),
                                                                        index=0,
                                                                        value=Load(
                                                                            base=Reference(name="c"),
                                                                            index=0,
                                                                        ),
                                                                    ),
                                                                ),
                                                            ],
                                                            value=Apply(
                                                                target=Reference(name="for0"),
                                                                arguments=[],
                                                            ),
                                                        ),
                                                        otherwise=Immediate(value=0),
                                                    ),
                                                ),
                                            ),
                                        ],
                                        body=Apply(target=Reference(name="for0"), arguments=[]),
                                    )
                                ],
                                value=Load(base=Reference(name="a"), index=0),
                            ),
                        ),
                    ),
                )
            ],
            body=Apply(
                target=Reference(name="fibo"),
                arguments=[Immediate(value=10)],
            ),
        ),
    )

    assert actual == expected


def test_symbol_lookup_and_lists_and_pairs():
    program_with_list_of_pairs_of_bool_and_func = L4.Program(
        definitions=[
            (
                "lis",
                L4.Symbol(
                    name="mylis",
                    payload=L4.Mutable(
                        oftype=L4.List(
                            typeof=L4.Pair(
                                type1=L4.Mutable(oftype=L4.Bool()), type2=L4.FuncType(parameters=[], result=L4.Void())
                            )
                        )
                    ),
                ),
                L4.HeapAllocate(
                    val=L4.NewList(
                        size=5,
                        typeof=L4.Pair(
                            type1=L4.Mutable(oftype=L4.Bool()), type2=L4.FuncType(parameters=[], result=L4.Void())
                        ),
                    )
                ),
            ),
            (
                "fill_lis",
                L4.FuncType(
                    parameters=[L4.Symbol(name="mylis", payload=L4.Void())],
                    result=L4.Void(),
                ),
                L4.Function(
                    params=[
                        (
                            "l",
                            L4.Symbol(name="mylis", payload=L4.Void()),
                        )
                    ],
                    body=L4.Let(
                        bindings=[
                            ("i", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0))),
                        ],
                        body=L4.While(
                            condition=L4.Operation(
                                operator="<",
                                left=L4.Get(target=L4.Reference(name="i"), index=0),
                                right=L4.Immediate(value=5),
                            ),
                            run=L4.Bunch(
                                expressions=[
                                    L4.Set(
                                        target=L4.Reference(name="i"),
                                        index=0,
                                        value=L4.Operation(
                                            operator="+",
                                            left=L4.Get(target=L4.Reference(name="i"), index=0),
                                            right=L4.Immediate(value=1),
                                        ),
                                    ),
                                    L4.Set(
                                        target=L4.Reference(name="l"),
                                        index=2,
                                        value=L4.NewPair(
                                            val1=L4.HeapAllocate(val=L4.Immediate(value=True)),
                                            val2=L4.Function(params=[], body=L4.Empty()),
                                            typeof=L4.Pair(
                                                type1=L4.Mutable(oftype=L4.Bool()),
                                                type2=L4.FuncType(parameters=[], result=L4.Void()),
                                            ),
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name="fill_lis"), arguments=[L4.Reference(name="lis")]),
    )
    actual = convert_to_l3(program=program_with_list_of_pairs_of_bool_and_func)
    expected = Program(
        parameters=[],
        body=Let(
            bindings=[
                (
                    "lis",
                    Let(
                        bindings=[
                            (
                                "heapallocateval0",
                                Let(
                                    bindings=[("list0", Allocate(count=5))],
                                    body=Begin(
                                        effects=[
                                            Store(
                                                base=Reference(name="list0"),
                                                index=0,
                                                value=Immediate(value=0),
                                            ),
                                            Store(
                                                base=Reference(name="list0"),
                                                index=1,
                                                value=Immediate(value=0),
                                            ),
                                            Store(
                                                base=Reference(name="list0"),
                                                index=2,
                                                value=Immediate(value=0),
                                            ),
                                            Store(
                                                base=Reference(name="list0"),
                                                index=3,
                                                value=Immediate(value=0),
                                            ),
                                            Store(
                                                base=Reference(name="list0"),
                                                index=4,
                                                value=Immediate(value=0),
                                            ),
                                        ],
                                        value=Reference(name="list0"),
                                    ),
                                ),
                            ),
                            ("heapallocate0", Allocate(count=1)),
                        ],
                        body=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="heapallocate0"),
                                    index=0,
                                    value=Reference(name="heapallocateval0"),
                                )
                            ],
                            value=Reference(name="heapallocate0"),
                        ),
                    ),
                ),
                (
                    "fill_lis",
                    Abstract(
                        parameters=["l"],
                        body=Let(
                            bindings=[
                                (
                                    "i",
                                    Let(
                                        bindings=[
                                            ("heapallocateval1", Immediate(value=0)),
                                            ("heapallocate1", Allocate(count=1)),
                                        ],
                                        body=Begin(
                                            effects=[
                                                Store(
                                                    base=Reference(name="heapallocate1"),
                                                    index=0,
                                                    value=Reference(name="heapallocateval1"),
                                                )
                                            ],
                                            value=Reference(name="heapallocate1"),
                                        ),
                                    ),
                                )
                            ],
                            body=LetRec(
                                bindings=[
                                    (
                                        "while0",
                                        Abstract(
                                            parameters=[],
                                            body=Branch(
                                                operator="==",
                                                left=Immediate(value=1),
                                                right=Branch(
                                                    operator="<",
                                                    left=Load(base=Reference(name="i"), index=0),
                                                    right=Immediate(value=5),
                                                    consequent=Immediate(value=1),
                                                    otherwise=Immediate(value=0),
                                                ),
                                                consequent=Begin(
                                                    effects=[
                                                        Begin(
                                                            effects=[
                                                                Store(
                                                                    base=Reference(name="i"),
                                                                    index=0,
                                                                    value=Primitive(
                                                                        operator="+",
                                                                        left=Load(
                                                                            base=Reference(name="i"),
                                                                            index=0,
                                                                        ),
                                                                        right=Immediate(value=1),
                                                                    ),
                                                                )
                                                            ],
                                                            value=Store(
                                                                base=Load(
                                                                    base=Reference(name="l"),
                                                                    index=0,
                                                                ),
                                                                index=2,
                                                                value=Let(
                                                                    bindings=[("pair0", Allocate(count=2))],
                                                                    body=Begin(
                                                                        effects=[
                                                                            Store(
                                                                                base=Reference(name="pair0"),
                                                                                index=0,
                                                                                value=Let(
                                                                                    bindings=[
                                                                                        (
                                                                                            "heapallocateval2",
                                                                                            Immediate(value=1),
                                                                                        ),
                                                                                        (
                                                                                            "heapallocate2",
                                                                                            Allocate(count=1),
                                                                                        ),
                                                                                    ],
                                                                                    body=Begin(
                                                                                        effects=[
                                                                                            Store(
                                                                                                base=Reference(
                                                                                                    name="heapallocate2",
                                                                                                ),
                                                                                                index=0,
                                                                                                value=Reference(
                                                                                                    name="heapallocateval2",
                                                                                                ),
                                                                                            )
                                                                                        ],
                                                                                        value=Reference(
                                                                                            name="heapallocate2",
                                                                                        ),
                                                                                    ),
                                                                                ),
                                                                            ),
                                                                            Store(
                                                                                base=Reference(name="pair0"),
                                                                                index=1,
                                                                                value=Abstract(
                                                                                    parameters=[],
                                                                                    body=Immediate(value=0),
                                                                                ),
                                                                            ),
                                                                        ],
                                                                        value=Reference(name="pair0"),
                                                                    ),
                                                                ),
                                                            ),
                                                        )
                                                    ],
                                                    value=Apply(
                                                        target=Reference(name="while0"),
                                                        arguments=[],
                                                    ),
                                                ),
                                                otherwise=Immediate(value=0),
                                            ),
                                        ),
                                    )
                                ],
                                body=Apply(target=Reference(name="while0"), arguments=[]),
                            ),
                        ),
                    ),
                ),
            ],
            body=Apply(
                target=Reference(name="fill_lis"),
                arguments=[Reference(name="lis")],
            ),
        ),
    )
    assert actual == expected

    program_circular_symbol = L4.Program(
        definitions=[
            (
                "a",
                L4.Symbol(
                    name="a",
                    payload=L4.Symbol(name="b", payload=L4.Void()),
                ),
                L4.Empty(),
            ),
            (
                "b",
                L4.Symbol(
                    name="b",
                    payload=L4.Symbol(name="a", payload=L4.Void()),
                ),
                L4.Empty(),
            ),
        ],
        body=L4.Empty(),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_circular_symbol)

    program_out_of_bounds_pair = L4.Program(
        definitions=[
            (
                "pair",
                L4.Symbol(
                    name="mypair",
                    payload=L4.Mutable(
                        oftype=L4.Pair(
                            type1=L4.Mutable(oftype=L4.Bool()), type2=L4.FuncType(parameters=[], result=L4.Void())
                        )
                    ),
                ),
                L4.HeapAllocate(
                    val=L4.NewPair(
                        val1=L4.HeapAllocate(val=L4.Immediate(value=False)),
                        val2=L4.Function(params=[], body=L4.Empty()),
                        typeof=L4.Pair(
                            type1=L4.Mutable(oftype=L4.Bool()), type2=L4.FuncType(parameters=[], result=L4.Void())
                        ),
                    )
                ),
            ),
            (
                "fill",
                L4.FuncType(
                    parameters=[L4.Symbol(name="mypair", payload=L4.Void())],
                    result=L4.Void(),
                ),
                L4.Function(
                    params=[
                        (
                            "l",
                            L4.Symbol(name="mypair", payload=L4.Void()),
                        )
                    ],
                    body=L4.LetRec(
                        bindings=[
                            (
                                "i",
                                L4.Symbol(name="_", payload=L4.Mutable(oftype=L4.Int())),
                                L4.HeapAllocate(val=L4.Immediate(value=0)),
                            ),
                        ],
                        body=L4.While(
                            condition=L4.Operation(
                                operator="<",
                                left=L4.Get(target=L4.Reference(name="i"), index=0),
                                right=L4.Immediate(value=5),
                            ),
                            run=L4.Bunch(
                                expressions=[
                                    L4.Set(
                                        target=L4.Reference(name="i"),
                                        index=0,
                                        value=L4.Operation(
                                            operator="+",
                                            left=L4.Get(target=L4.Reference(name="i"), index=0),
                                            right=L4.Immediate(value=1),
                                        ),
                                    ),
                                    L4.Set(
                                        target=L4.Reference(name="l"),
                                        index=5,
                                        value=L4.HeapAllocate(val=L4.Immediate(value=True)),
                                    ),
                                ]
                            ),
                        ),
                    ),
                ),
            ),
        ],
        body=L4.Call(target=L4.Reference(name="fill"), arguments=[L4.Reference(name="pair")]),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_out_of_bounds_pair)

    program_type_mismatch = L4.Program(
        definitions=[("x", L4.Void(), L4.Immediate(value=1))],
        body=L4.Empty(),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_type_mismatch)

    # not using pair type with newpair
    with pytest.raises(ValueError):
        process_expression(
            expression=L4.NewPair(val1=L4.Immediate(value=0), val2=L4.Immediate(value=0), typeof=L4.Void()),
            context={},
            symbols={},
            fresh=SequentialNameGenerator(),
        )

    program_mutable_ref = L4.Program(
        definitions=[("x", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=1)))],
        body=L4.Get(target=L4.Reference(name="x"), index=0),
    )
    actual = convert_to_l3(program=program_mutable_ref)
    expected = Program(
        parameters=[],
        body=Let(
            bindings=[
                (
                    "x",
                    Let(
                        bindings=[
                            ("heapallocateval0", Immediate(value=1)),
                            ("heapallocate0", Allocate(count=1)),
                        ],
                        body=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="heapallocate0"),
                                    index=0,
                                    value=Reference(name="heapallocateval0"),
                                )
                            ],
                            value=Reference(name="heapallocate0"),
                        ),
                    ),
                )
            ],
            body=Load(base=Reference(name="x"), index=0),
        ),
    )
    assert actual == expected

    program_empty_bunch = L4.Program(
        definitions=[],
        body=L4.Bunch(expressions=[]),
    )
    actual = convert_to_l3(program=program_empty_bunch)
    expected = Program(parameters=[], body=Let(bindings=[], body=Immediate(value=0)))
    assert actual == expected

    program_multiple_symbols = L4.Program(
        definitions=[
            ("a", L4.Symbol(name="a", payload=L4.Symbol(name="b", payload=L4.Void())), L4.Immediate(value=None)),
            ("b", L4.List(typeof=L4.Int()), L4.NewList(size=2, typeof=L4.Int())),
        ],
        body=L4.For(times=1, run=L4.Bunch(expressions=[L4.Get(target=L4.Reference(name="b"), index=1), L4.Empty()])),
    )
    actual = convert_to_l3(program=program_multiple_symbols)
    expected = Program(
        tag="l3",
        parameters=[],
        body=Let(
            tag="let",
            bindings=[
                ("a", Immediate(tag="immediate", value=0)),
                (
                    "b",
                    Let(
                        tag="let",
                        bindings=[("list0", Allocate(tag="allocate", count=2))],
                        body=Begin(
                            tag="begin",
                            effects=[
                                Store(
                                    tag="store",
                                    base=Reference(tag="reference", name="list0"),
                                    index=0,
                                    value=Immediate(tag="immediate", value=0),
                                ),
                                Store(
                                    tag="store",
                                    base=Reference(tag="reference", name="list0"),
                                    index=1,
                                    value=Immediate(tag="immediate", value=0),
                                ),
                            ],
                            value=Reference(tag="reference", name="list0"),
                        ),
                    ),
                ),
            ],
            body=LetRec(
                tag="letrec",
                bindings=[
                    (
                        "for_counter0",
                        Let(
                            tag="let",
                            bindings=[
                                ("mutableval0", Immediate(tag="immediate", value=1)),
                                ("mutable0", Allocate(tag="allocate", count=1)),
                            ],
                            body=Begin(
                                tag="begin",
                                effects=[
                                    Store(
                                        tag="store",
                                        base=Reference(tag="reference", name="mutable0"),
                                        index=0,
                                        value=Reference(tag="reference", name="mutableval0"),
                                    )
                                ],
                                value=Reference(tag="reference", name="mutable0"),
                            ),
                        ),
                    ),
                    (
                        "for0",
                        Abstract(
                            tag="abstract",
                            parameters=[],
                            body=Branch(
                                tag="branch",
                                operator="==",
                                left=Immediate(tag="immediate", value=1),
                                right=Branch(
                                    tag="branch",
                                    operator="<",
                                    left=Immediate(tag="immediate", value=0),
                                    right=Load(
                                        tag="load", base=Reference(tag="reference", name="for_counter0"), index=0
                                    ),
                                    consequent=Immediate(tag="immediate", value=1),
                                    otherwise=Immediate(tag="immediate", value=0),
                                ),
                                consequent=Begin(
                                    tag="begin",
                                    effects=[
                                        Store(
                                            tag="store",
                                            base=Reference(tag="reference", name="for_counter0"),
                                            index=0,
                                            value=Primitive(
                                                tag="primitive",
                                                operator="-",
                                                left=Load(
                                                    tag="load",
                                                    base=Reference(tag="reference", name="for_counter0"),
                                                    index=0,
                                                ),
                                                right=Immediate(tag="immediate", value=1),
                                            ),
                                        ),
                                        Begin(
                                            tag="begin",
                                            effects=[
                                                Load(tag="load", base=Reference(tag="reference", name="b"), index=1)
                                            ],
                                            value=Immediate(tag="immediate", value=0),
                                        ),
                                    ],
                                    value=Apply(
                                        tag="apply", target=Reference(tag="reference", name="for0"), arguments=[]
                                    ),
                                ),
                                otherwise=Immediate(tag="immediate", value=0),
                            ),
                        ),
                    ),
                ],
                body=Apply(tag="apply", target=Reference(tag="reference", name="for0"), arguments=[]),
            ),
        ),
    )
    assert actual == expected

    program_incorrect_ref = L4.Program(
        definitions=[("a", L4.Void(), L4.Immediate(value=None))],
        body=L4.Reference(name="b"),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_incorrect_ref)


def test_get_set_edge():
    program = L4.Program(
        definitions=[("a", L4.Int(), L4.Immediate(value=0))], body=L4.Get(target=L4.Reference(name="a"), index=0)
    )
    actual = convert_to_l3(program=program)
    expected = Program(
        parameters=[],
        body=Let(bindings=[("a", Immediate(value=0))], body=Reference(name="a")),
    )
    assert actual == expected

    program_get_scalar_out_of_bounds = L4.Program(
        definitions=[("a", L4.Int(), L4.Immediate(value=0))], body=L4.Get(target=L4.Reference(name="a"), index=5)
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_get_scalar_out_of_bounds)

    program_get_incorrect_ref = L4.Program(
        definitions=[("a", L4.Int(), L4.Immediate(value=0))], body=L4.Get(target=L4.Reference(name="b"), index=5)
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_get_incorrect_ref)

    # pair out of bounds with get
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Get(target=L4.Reference(name="pair"), index=5),
            context={"pair": L4.Pair(type1=L4.Int(), type2=L4.Int())},
            symbols={},
        )
    # scalar out of bounds with get
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Get(target=L4.Reference(name="a"), index=1),
            context={"a": L4.Int()},
            symbols={},
        )
    get_context_searching_sym = process_expression(
        expression=L4.Get(target=L4.Reference(name="a"), index=0),
        context={"a": L4.Mutable(oftype=L4.Symbol(name="sym", payload=L4.Void()))},
        symbols={"sym": L4.Mutable(oftype=L4.Int())},
        fresh=SequentialNameGenerator(),
    )
    assert get_context_searching_sym == Load(base=Load(base=Reference(name="a"), index=0), index=0)

    set_context_searching_sym = process_expression(
        expression=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=0)),
        context={"a": L4.Mutable(oftype=L4.Symbol(name="sym", payload=L4.Void()))},
        symbols={"sym": L4.Mutable(oftype=L4.Int())},
        fresh=SequentialNameGenerator(),
    )
    assert set_context_searching_sym == Store(
        base=Load(base=Reference(name="a"), index=0), index=0, value=Immediate(value=0)
    )

    check_get_second_element_from_pair = check_expression(
        expression=L4.Get(target=L4.Reference(name="a"), index=1),
        context={"a": L4.Symbol(name="b", payload=L4.Void())},
        symbols={"b": L4.Mutable(oftype=L4.Pair(type1=L4.Int(), type2=L4.Bool()))},
    )

    assert check_get_second_element_from_pair == L4.Bool()

    check_set_second_element_from_pair = check_expression(
        expression=L4.Set(target=L4.Reference(name="a"), index=1, value=L4.Immediate(value=False)),
        context={"a": L4.Symbol(name="b", payload=L4.Void())},
        symbols={"b": L4.Mutable(oftype=L4.Pair(type1=L4.Int(), type2=L4.Bool()))},
    )

    assert check_set_second_element_from_pair == L4.Void()

    program = L4.Program(
        definitions=[("a", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0)))],
        body=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=0)),
    )
    actual = convert_to_l3(program=program)
    expected = Program(
        parameters=[],
        body=Let(
            bindings=[
                (
                    "a",
                    Let(
                        bindings=[
                            ("heapallocateval0", Immediate(value=0)),
                            ("heapallocate0", Allocate(count=1)),
                        ],
                        body=Begin(
                            effects=[
                                Store(
                                    base=Reference(name="heapallocate0"),
                                    index=0,
                                    value=Reference(name="heapallocateval0"),
                                )
                            ],
                            value=Reference(name="heapallocate0"),
                        ),
                    ),
                )
            ],
            body=Store(
                base=Reference(name="a"),
                index=0,
                value=Immediate(value=0),
            ),
        ),
    )
    assert actual == expected

    program_trying_to_set_immutable = L4.Program(
        definitions=[("a", L4.Int(), L4.Immediate(value=0))],
        body=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=1)),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_trying_to_set_immutable)

    program_set_scalar_out_of_bounds = L4.Program(
        definitions=[("a", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0)))],
        body=L4.Set(target=L4.Reference(name="a"), index=4, value=L4.Immediate(value=0)),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_set_scalar_out_of_bounds)

    program_set_incorrect_ref = L4.Program(
        definitions=[("a", L4.Mutable(oftype=L4.Int()), L4.HeapAllocate(val=L4.Immediate(value=0)))],
        body=L4.Set(target=L4.Reference(name="b"), index=0, value=L4.Immediate(value=0)),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_set_incorrect_ref)

    # pair out of bounds with set
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Set(target=L4.Reference(name="pair"), index=5, value=L4.Immediate(value=0)),
            context={"pair": L4.Pair(type1=L4.Int(), type2=L4.Int())},
            symbols={},
        )
    # scalar out of bounds with set
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Set(target=L4.Reference(name="a"), index=1, value=L4.Immediate(value=0)),
            context={"a": L4.Int()},
            symbols={},
        )

    check_set_mutable_symbol_chain = check_expression(
        expression=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=0)),
        context={"a": L4.Mutable(oftype=L4.Symbol(name="b", payload=L4.Void()))},
        symbols={"b": L4.Mutable(oftype=L4.Int())},
    )

    assert check_set_mutable_symbol_chain == L4.Void()

    # attempt setting immutable
    with pytest.raises(ValueError):
        process_expression(
            expression=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=0)),
            context={"a": L4.Symbol(name="b", payload=L4.Void())},
            symbols={"b": L4.Int()},
            fresh=SequentialNameGenerator(),
        )


def test_edge_cases():
    program_unmatching_function_args = L4.Program(
        definitions=[
            (
                "f",
                L4.FuncType(parameters=[L4.Int()], result=L4.Void()),
                L4.Function(params=[("x", L4.Int())], body=L4.Empty()),
            )
        ],
        body=L4.Call(target=L4.Reference(name="f"), arguments=[]),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_unmatching_function_args)

    program_unmatching_function_name = L4.Program(
        definitions=[
            (
                "f",
                L4.FuncType(parameters=[L4.Int()], result=L4.Void()),
                L4.Function(params=[("x", L4.Int())], body=L4.Empty()),
            )
        ],
        body=L4.Call(target=L4.Reference(name="no"), arguments=[]),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_unmatching_function_name)

    program_duplicate_def = L4.Program(
        definitions=[("a", L4.Int(), L4.Immediate(value=0)), ("a", L4.Int(), L4.Immediate(value=0))],
        body=L4.Get(target=L4.Reference(name="a"), index=0),
    )
    with pytest.raises(ValueError):
        convert_to_l3(program=program_duplicate_def)

    # dublicate ref in let
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Let(bindings=[("a", L4.Void(), L4.Empty()), ("a", L4.Void(), L4.Empty())], body=L4.Empty()),
            context={},
            symbols={},
        )

    # dublicate ref in letrec
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.LetRec(
                bindings=[("a", L4.Void(), L4.Empty()), ("a", L4.Void(), L4.Empty())], body=L4.Empty()
            ),
            context={},
            symbols={},
        )

    # dublicate ref in function
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Function(params=[("a", L4.Void()), ("a", L4.Void())], body=L4.Empty()),
            context={},
            symbols={},
        )

    # trying to call a nonfunction
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.Call(target=L4.Reference(name="a"), arguments=[]),
            context={"a": L4.Int()},
            symbols={},
        )

    # incorrect pair type
    with pytest.raises(ValueError):
        check_expression(
            expression=L4.NewPair(val1=L4.Empty(), val2=L4.Empty(), typeof=L4.Void()),
            context={},
            symbols={},
        )
