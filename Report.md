# Project: Extending L0 - L3 with Minor/Major features

*Team members: Dogukan Avci, Ruixiang Jiang (Graduate Group)*

Link: https://github.com/RuixiangJiang/471c/tree/project

## Overview

This proposal describes minor and major language extensions. Two layers are newly implemented, L4 and L5. The major extension for L4 and L5 are static type implementation and Java-type class implementation, respectively.

What L4 does:
- Introduce static type checking and safety
- Provide symbols for custom type functionality
- Create loop functionality
- Create basic types and collection types like pair and list
- Provide a simpler way of creating mutable data and heap allocation
  
What L5 does:
- boolean operators `$$` and `||`
- `switch-case` branches
- `break` and `continue`
- `foreach`
- Class with basic construction methods, inheritance, and overridding.

## L4 (Dogukan Avci)

L4 introduces types on top of L3, and it performs static type checking.
It ensures type cohesion and rejects circular symbolic dependencies during compilation.
It introduces custom types to enable higher-level compilers to simplify their implementations.
L3 did not require any expansion to accommodate the features added in L4.
This is a strong point, since the L3 backend did not require any extra workload to support a safer and more feature-rich compiler on top of it.

## Overview

The main pipeline operation of L4 to L3 conversion goes like so:
```text
L4 program ->
Type checking ->
Lowering and translation ->
L3 program
```

For example:
```text
L4.While(
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
    )
)
```
This code iterates 5 times and assigns the list l at index 2 with new pairs it allocates. The pairs are composed of a boolean and a function.
This L4 expression lowers into L3 with the following form:
```text
LetRec(
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
  body=Apply(target=Reference(name="while0"), arguments=[])
)
```
## Types

L4 has the following types:
```text
Mutable | Int | Bool | FuncType | List | Pair | Symbol | Void
```

-Int, Bool, Void are primitive (concrete) types
-FuncType represents the function signatures with the type of each parameter and the return type of the function
-Symbol is resolved to a concrete type before its type is compared and checked. It serves as the custom type feature
-Pair, List, and Mutable are wrapper types that use other types in their signature

For example:
(mutable int):
is a heap-allocated, modifiable integer value.

(pair int (mutable (pair bool int))):
is a pair type where the first element is an int and the second element is another pair type with a bool and an int.
The initial pair is immutable, but the child pair is mutable.

## Type checking

L4 is very strict when it comes to types.
Compound expressions are strictly checked to ensure that all resolution paths preserve type safety.
Since L4 also offers custom symbolic types, the compatibility of symbols and concrete types is also supported through symbol resolution.
The operands and all expressions need to satisfy expected typing rules to pass the compiler checks.

For example:
(if true A B):
is valid only if A type is equal to B type, and it is checked before runtime.

Similarly:
(+ a 2):
needs to make sure a is of type int before allowing addition.

## Check expression

Check expression acts recursively on ASTs and traverses symbols and
subexpressions until it finds its way to a concrete representation.
To ensure this, each new expression in L4 has a signature. This signature
is a method determining the type the expression ultimately resolves to, while also incorperating symbols
at every level.

Check expression recursively resolves types for the following:
Conditionals
Arithmetics
Functions and function calls:
-Each argument type in a call is checked, and the correct number of arguments along with correct types of arguments must be provided
Immediates
Mutable read / write operations
-Write operation checks if the data type is actually mutable
List and pair access and construction
Let and Letrec
Loops
Sequencing (Bunch)

For example:
((lambda ((x int) (y bool)) x) 7 true)
This is a valid function call, but:
((lambda ((x int) (y bool)) x) 7 9)
This is not because the second argument provided here is an int, but the function expects a bool.

Similarly:
((lambda ((x int) (y bool) (z bool)) x) 7 true true)
This is a valid function call, but:
((lambda ((x int) (y bool) (z bool)) x) 7 true true 55)
This is not because the code attempts to call the method with an extra argument
and the function type does not match.

(set x 0 6)
This is only valid if x’s type has a mutable wrapper.

The custom type (Symbol) also resolves to a concrete type before going through the check:
For example, for the given symbol context:
```text
symbols = {
    "T": L4.Int(),
}
```

Following assertions both pass:
```text
assert resolve_type(
    L4.Symbol(name="T", payload=L4.Bool()),
    symbols=symbols,
) == L4.Int()

assert resolve_type(
    L4.Symbol(name="X", payload=L4.Int()),
    symbols=symbols,
) == L4.Int()
```
Payload section here is the fallback type when the symbol does not exist, so for the second example
the X symbol will fail to resolve and the payload type will be used as the fallback type and pass the check.

## Symbol resolution

Symbol has a separate context where different symbols could reference each other.
Symbol resolution traverses the symbolic context until it finds a concrete representation and detects circular references if they exist.
All final type comparisons are made solely with resolved types.
Symbol type is created to make it easy for developers to reuse types.
They are essentially aliases that make it easier to navigate a strict type-checking compiler.

Symbols could represent large and composite types, for example:
custom_list = (mutable (list (pair (mutable bool) (-> () void)))) 
is a mutable list composed of pairs that contain mutable bools and function types.

For example:
custom_type = int
(let ((x custom_type 5)) (+ x 1) )

This is valid because custom_type will resolve to int prior to type check and pass the checks.

custom_type = custom_type_2
custom_type_2 = custom_type_3
custom_type_3 = custom_type

This is invalid because of the circular dependency of symbols, so the compiler will notify you about the circular symbols.

## Mutability and heap allocation abstraction

L4 introduces Mutable types and lowers the mutable values into explicit store and allocate representations.
If a variable has a mutable type but the initialization is not already heap-allocated, the L4 layer detects this
and inserts the heap allocation automatically, like so:

The following L4 binding:
```text
("x", L4.Mutable(oftype=L4.Int()), L4.Immediate(value=1))
```

is lowered into the following L3 code:
```text
(
    "x",
    Let(
        bindings=[
            ("mutableval0", Immediate(value=1)),
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
)
```
L3 allocate, load, and store are used extensively for the heap allocation.

Trying to set an immutable type will fail. For example:
```text
process_expression(
    expression=L4.Set(target=L4.Reference(name="a"), index=0, value=L4.Immediate(value=0)),
    context={"a": L4.Symbol(name="b", payload=L4.Void())},
    symbols={"b": L4.Int()},
    fresh=SequentialNameGenerator(),
)
```
Here the variable a is of custom type b and resolves to an immutable int, so the compiler will throw an error warning about the attempt.

Get and set expressions abstract away read and write operations and make it easy to work with data.
The get and set expressions ensure structural index checks are correct for scalar values and pairs.
The writes through set expression are checked if the types of the stored value and the value being written match.
L4’s collection types: pair and list make use of get and set heavily to simplify working with an array of data.
Pair and list are not just a collection of data; however, through the access expressions of get and set,
they support type-directed access and update without requring extra back-end level data operations.

For example:
for a pair p that is (pair bool int)
(get p 0):
has type bool
(get p 1):
has type int

## Loops and sequencing

L4 has while and for loops that work with a controlled sequencing of recursive letrecs for L3 conversion.
The for loop creates an internal counter to achieve this, while the while loop works with a periodic check of a given expression that resolves into a bool type.
Loop construction makes use of sequencing and the introduced type-safe mutability, along with the recursion, to achieve
loop behavior without L3 side back-end support.

For example, a for loop translation looks like so in L4:
```text
L4.Program(
    definitions=[
        ("a", L4.Symbol(name="a", payload=L4.Symbol(name="b", payload=L4.Void())), L4.Immediate(value=None)),
        ("b", L4.List(typeof=L4.Int()), L4.NewList(size=2, typeof=L4.Int())),
    ],
    body=L4.For(times=1, run=L4.Bunch(expressions=[L4.Get(target=L4.Reference(name="b"), index=1), L4.Empty()])),
)
```    
The lowered version in L3 looks like so:
```text
Program(
    parameters=[],
    body=Let(
        bindings=[
            ("a", Immediate( value=0)),
            (
                "b",
                Let(
                    bindings=[("list0", Allocate(count=2))],
                    body=Begin(
                        effects=[
                            Store(
                                base=Reference( name="list0"),
                                index=0,
                                value=Immediate( value=0),
                            ),
                            Store(
                                base=Reference(name="list0"),
                                index=1,
                                value=Immediate( value=0),
                            ),
                        ],
                        value=Reference( name="list0"),
                    ),
                ),
            ),
        ],
        body=LetRec(
            bindings=[
                (
                    "for_counter0",
                    Let(
                        bindings=[
                            ("mutableval0", Immediate( value=1)),
                            ("mutable0", Allocate( count=1)),
                        ],
                        body=Begin(
                            effects=[
                                Store(
                                    base=Reference( name="mutable0"),
                                    index=0,
                                    value=Reference( name="mutableval0"),
                                )
                            ],
                            value=Reference( name="mutable0"),
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
                                left=Immediate( value=0),
                                right=Load(
                                   base=Reference( name="for_counter0"), index=0
                                ),
                                consequent=Immediate( value=1),
                                otherwise=Immediate( value=0),
                            ),
                            consequent=Begin(
                                effects=[
                                    Store(
                                        base=Reference( name="for_counter0"),
                                        index=0,
                                        value=Primitive(
                                            operator="-",
                                            left=Load(
                                                base=Reference( name="for_counter0"),
                                                index=0,
                                            ),
                                            right=Immediate( value=1),
                                        ),
                                    ),
                                    Begin(
                                        effects=[
                                            Load( base=Reference( name="b"), index=1)
                                        ],
                                        value=Immediate( value=0),
                                    ),
                                ],
                                value=Apply(
                                   target=Reference( name="for0"), arguments=[]
                                ),
                            ),
                            otherwise=Immediate( value=0),
                        ),
                    ),
                ),
            ],
            body=Apply( target=Reference( name="for0"), arguments=[]),
        ),
    ),
)
```
Heap allocations and internal for_counter0 variable are used in L3 version to keep track correctly.

Sequencing expression is a bunch; it is simply a collection of expressions, but its type is the type of the last expression.
Sequencing makes it easy to reason about the code by creating bigger bodies, unlike the never-ending chaining structure of L3.
For example, the return type of bunch will be void if we perform a set operation at the end.

## Reflection

The course and the project provided deep exposure to the detailed planning and structures of compiler design. 
Understanding the purpose behind each layer and the trade-off between controlling a larger chunk of the program flow and utilizing the abstractions provided for simpler programming became apparent while implementing the transformations on the backend. I concluded that there is no correct or wrong way to design compilers as long as the overhead, performance, and the abstraction provided are acceptable. 
The journey solidified my overall inclination towards types, classes, and inheritance in modern compilers because I prefer my interactions with compilers to be as abstracted away as possible, maybe at the cost of performance at times. Ultimately, it was a fun experience and I enjoyed extending the compiler with features I prefer to work with.

## L5 (Ruixiang Jiang)

L5 is implemented as a high-level front end on top of L4. The backend does not need to understand classes, inheritance, switch, foreach, or short-circuit operators directly. Instead, L5 introduces new AST nodes and then lowers them into existing L4 constructs.

### Overview

The general pipeline is:

```text
L5 source code
    ↓ parser
L5 AST
    ↓ collect class information + type checking + lowering
L4 AST
    ↓ existing L4 → L3 conversion
L3 AST
```

The implementation is split conceptually into three parts:

```text
syntax.py
  Defines L5 AST nodes.

parse.py
  Parses S-expression L5 source code into L5 AST.

minor_convert.py
  Lowers minor language extensions:
    - short-circuit operators
    - switch
    - break / continue
    - foreach

inheritance_convert.py
  Collects class information, inherited fields, inherited methods, and overridden methods.

class_convert.py
  Type-checks and lowers class-related features:
    - class type
    - object construction
    - field access
    - field assignment
    - method call
    - method definition

convert.py
  Coordinates the whole L5 → L4 lowering process.
```

The `/test` folder contains tests towards L5 source codes, and the coverage rate is 100%. Additionally, `test_parse_program_sources.py` supports programs checking - that give a L5 source code as input, check whether the AST is correct. The test programs in the file are generated by ChatGPT.

### Minor Convert Implementation

- Boolean operators `$$` and `||`
  They are lowered into L4 `if` expressions.
  The rule is:
  ```text
  (&& a b)  →  (if a b false)
  (|| a b)  →  (if a true b)
  ```
- `switch-case` branches
  The structure is:
  ```text
  (switch scrutinee
  (case value1 body1)
  (case value2 body2)
  ...
  (default default-body))
  ```
  For example, the following L5 code:
  ```text
  (switch x
  (case 0 10)
  (case 1 20)
  (default 30))
  ```
  It will be transferred to the following L4 code:
  ```text
  (let ((switch_scrutinee0 int x))
  (if (== switch_scrutinee0 0)
      10
      (if (== switch_scrutinee0 1)
          20
          30)))
  ```
- `break` and `continue`
  To minimize the change to L4 level code, the implementations are simple but will introduce performance degration, that to treat them as two boolean flags hard-coded inside the loop body. Before executing each statement, check the two flags first.
  For example, if a loop body is originally:
  ```text
  (begin
  expr1
  expr2
  expr3)
  ```
  It will be:
  ```text
  (begin
  expr1
  (if break_or_continue
      empty
      expr2)
  (if break_or_continue
      empty
      expr3))
  ```
- `foreach`
  The format is:
  ```text
  (foreach (binder type) target count body)
  ```
  For example, the following code:
  ```text
  (foreach (x int) xs 3
  body)
  ```
  It will beome:
  ```text
  (begin
  (let ((x int (get xs 0))) body)
  (let ((x int (get xs 1))) body)
  (let ((x int (get xs 2))) body))
  ```

### Basic Class Implementation

A class is defined as a `mutable payload` at L4. For a class with one field like `(class Box
  (fields
    (value int))
  ...)`, the object type is `(mutable int)`. For two-fields class like `(class Point
  (fields
    (x int)
    (y int))
  ...)`, it becomes `(mutable (pair int int))`. For three fields like `a, b, c`, it becomes `mutable(pair(a, pair(b, c)))`.

Methods are lowered to top-level functions. The receiver object is passed explicitly as the first parameter named `this`. For example, `(method getX () int
  (. this x))` will become:
```text
(Point_getX
  (-> ((mutable (pair int int))) int)
  (lambda ((this (mutable (pair int int))))
    ...))
```

A whole class, for example:
```text
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
```
It will become:
```text
(l4
  (definitions

    (Point_getX
      (-> ((mutable (pair int int))) int)
      (lambda
        ((this (mutable (pair int int))))
        (let
          ((root0 (pair int int)
             (get this 0)))
          (let
            ((field0 int
               (get root0 0)))
            field0))))

    (Point_setX
      (-> ((mutable (pair int int)) int) void)
      (lambda
        ((this (mutable (pair int int)))
         (v int))
        (let
          ((root1 (pair int int)
             (get this 0)))
          (set this 0
            (new-pair
              v
              (get root1 1)
              (pair int int))))))

    (p
      (mutable (pair int int))
      (heap-allocate
        (new-pair
          1
          2
          (pair int int)))))

  (begin
    (Point_setX p 7)
    (Point_getX p)))
```

The important transformations are:
```text
(class Point ...)
  → no direct L4 equivalent; class is removed after lowering

(class Point) type
  → mutable(pair int int)

(new Point 1 2)
  → heap-allocate(pair(1, 2))

(. this x)
  → get field 0 from object payload

(set-field! this x v)
  → rebuild object payload and set it back

(call-method p getX)
  → Point_getX(p)
```

### Inheritance Implementation

Inheritance is handled during class collection. For every class, the compiler builds a `ClassInfo` object containing:
```text
name
parent
own_fields: fields declared directly in this class
all_fields: inherited fields followed by own fields
own_methods: methods declared directly in this class
methods: inherited methods plus own methods
```

For example, the following L5:
```text
(l5
  (classes
    (class Point
      (fields
        (x int))
      (methods))

    (class ColoredPoint extends Point
      (fields
        (color int))
      (methods
        (method sum () int
          (+ (. this x)
             (. this color))))))

  (definitions
    (def p (class ColoredPoint)
      (new ColoredPoint 10 7)))

  (call-method p sum))
```

It will be converted to the following L4:
```text
(l4
  (definitions

    (ColoredPoint_sum
      (-> ((mutable (pair int int))) int)
      (lambda
        ((this (mutable (pair int int))))
        (+
          ;; inherited field x
          (let
            ((root0 (pair int int)
               (get this 0)))
            (let
              ((field0 int
                 (get root0 0)))
              field0))

          ;; own field color
          (let
            ((root1 (pair int int)
               (get this 0)))
            (let
              ((field1 int
                 (get root1 1)))
              field1)))))

    (p
      (mutable (pair int int))
      (heap-allocate
        (new-pair
          10
          7
          (pair int int)))))

  (ColoredPoint_sum p))
```

Here, a `ColoredPoint` object's field contains two variables: `x` at index 0 and `color` at index 1.

### Method Override

Note that L5 supports only overriding, not overloading.

During class collection, the child class first copies the inherited method table, then each child method is inserted into the table. If the method name already exists in the inherited method table, the compiler checks that the signature is the same, i.e., the same parameter list and return type. If so, the child method replaces the inherited one.

Conceptually, after overriding there is:
```text
Parent.methods["value"]
  = MethodInfo(owner="Parent", name="value", ...)
Child.methods["value"]
  = MethodInfo(owner="Child", name="value", ...)
```

The method calls are lowered using `owner + _ + method_name`, for example, `Child.value` will be converted to `Child_value`.

Example:
```text
(l5
  (classes
    (class Point
      (fields)
      (methods
        (method value () int
          1)))

    (class ColoredPoint extends Point
      (fields)
      (methods
        (method value () int
          2))))

  (definitions
    (def p (class Point)
      (new Point))
    (def cp (class ColoredPoint)
      (new ColoredPoint)))

  (begin
    (call-method p value)
    (call-method cp value)))
```

It will become:
```text
(l4
  (definitions

    (Point_value
      (-> ((mutable void)) int)
      (lambda
        ((this (mutable void)))
        1))

    (ColoredPoint_value
      (-> ((mutable void)) int)
      (lambda
        ((this (mutable void)))
        2))

    (p
      (mutable void)
      (heap-allocate empty))

    (cp
      (mutable void)
      (heap-allocate empty)))

  (begin
    (Point_value p)
    (ColoredPoint_value cp)))
```

The current override implementation is static, which means there is no runtime vtable. If we have a class `(def cp (class ColoredPoint)
  (new ColoredPoint))` and call `(call-method cp value)`, it will be actually `(ColoredPoint_value cp)`.

The type equality is strictly checked. For example, `(def p (class Point)
  (new ColoredPoint))` is not supported.