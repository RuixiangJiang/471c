# Proposal: Extending L0 - L3 with Minor/Major features

*Team members: Dogukan Avci, Ruixiang Jiang (Graduate Group)*

## Overview

This proposal describes minor and major language extensions. Two layers are newly implemented, L4 and L5. The major extension for L4 and L5 are static type implementation and Java-type class implementation, respectively.

## L4 (Dogukan)

### Goals:
- Introduce static type checking and safety
- Provide symbols for custom type functionality
- Create loop functionality
- Create basic types and collection types like pair and list
- Provide a simpler way of creating mutable data and heap allocation
### Non-goals:
- Type inference, dynamic types

### Motivation:
Creating simple programs with L3 requires creating big chains of manual code that does not guarantee a correct flow and requires reasoning at runtime for finding bugs. Even a simple loop requires using many terms together. L4 aims to introduce type safety, and loops to solve this problem.

L4 introduces types, and simplifies L3 into a more robust and easier to use expressions. It enables aliasing and introduces simple data structures such as List and Pair. Features like loops, get and set expressions, heap allocation expressions, function signatures both make it much easier to code programs with higher level constructions and make it much safer by strict type checking measures. Main purpose of L4 is to prevent these big chains of load, allocation, begin, store operations and perform with much more smaller expressions while guaranteeing type safety which L3 lacks. Type safety is an important improvement that shifts the responsibility of operation correctness from the programmer to the compiler. It is much more safe to code with L4 since the compiler notifies mistakes before runtime. Programmers who write code that require type consistency, type safety, loops, and overall more abstract layers would benefit from using L4.

### Sample Fibonacci Implementation With L4:
```
L4.Program(
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
```

### Description of L4:

L4 introduces the following functionalities over L3:
- Static type system that has int, bool, void primitive types and pair and list as grouping types
- Get and set operations that use types to understand how to perform fetching and writing and making it easier to work with data
- Expanding L3 Let and LetRec to work with types
- Function signatures to ensure correct calling conventions and the ability to create functions and call them
- Symbols so custom types can be created and used throughout the program
- (Types are immutable by default) Mutability types and a clear division between mutable and immutable types, the ability to save to the heap and reference the heap
- For and while loops
- Type ascription to couple types and expressions for ease of writing, bunch supports grouping multiple expressions together like a body
- Merging primitive and branchlike operator using constructs into Operator and simplification of branch into If
- Empty expression for Void cases and for exiting

Static type system performs type checking to ensure types are compatible. List and pairs can be created with NewList and NewPair and types can be combined that adds features like mutability, or symbols. L4 program has no arguments but has definitions instead that use types and wraps L3 with a let that enables access to the definitions throughout the program. One important distinction here is these definitions can refer to each other like a LetRec, and symbols are resolved early before full L3 conversion to make this possible. Type keywords dictate how data behaves, set cannot work with an immutable type, and aliases can be used with symbols that make it easier to capture high level ideas. Symbols are kept in a separate context and could be referred by other types. Symbol type is used to represent them and needs a payload and a name to refer. Symbol resolution happens at both type checking and code generation steps.Payload of Symbol is the fallback case for cases where symbol was not found in the symbols context. Basic types like Int, Bool, Void represent int, bool, and void which are popular types for writing programs.

List type supports creating a fixed size heap allocated collection with the help of NewList, but all types in a list need to be the same. NewList operation simply is an L3 term that puts Allocate term with required size into Let's binding and puts a Begin term to Let's body. This Begin term saves a Store effect per created index that the base references the Allocate term. The value type is by default an Immediate term with value 0. Body of this Begin term is the reference to the Allocate term. For bundling two different types into one data structure, a pair is used instead that accepts different types, and NewPair is used for creating the pairs. NewPair uses pretty much the same conversion strategy as List but with size of 2 and goes through different type checks. 

Mutable type supports and represents mutability feature of data, and enables the program to edit the heap during execution. Mutable works very similar to HeapAllocate that simply wraps the value with a Let, saves allocation and actual data conversion as bindings with a fresh id, then in the body of let, creates a begin construct that saves Store as an effect that uses the allocation as base and actual resolved data structure as the value, then in the value section of this Begin, it references the allocation. HeapAllocate expression is an explicit version of what Mutable type does and the program checks and avoids double allocation of those two. There is a direct dependence between Mutable and For expressions since For creates a mutable counter internally.

Expressions like If, Call, Reference, Immediate, Operation, Let, LetRec are simply type extended versions of L3 counterparts but Operation combines Primitive and Branch as the sole operator based expression. If expression is simply a branch that compares condition result to an Immediate term of value 1 for equality. L4 Immediate infers type differently than L3 Immediate. Bunch is an expression that converts each expression to a term and saves them into the effects of an L3 Begin term, and the last term to its body, its return type is the type of the last expression passed to it. It makes it easy to write expressions in a flat and easier to understand fashion. Both For and While expressions use Bunch under the hood to group condition check, decrement in For case, and Call at the end of Bunch to loop around. For loop can work with both integers and expressions returning int types. Both For and While use L3 LetRec term to turn the loops into a functional representation that checks the condition and iterates consistently. Run for both loops need to be Void type explicitly. 

Type ascription that makes it easy to group expressions and types is achieved with Capsule expression. Get and Set perform strict type checking (for example index for a Pair type need to be 0 or 1 for both Get and Set) and behave based on the type of data, they make it easy to reason about reading and writing operations in higher levels and create a great base to build on top for more complicated data structures. Get simply uses Load terms under the hood and supports unrolling until the intended data is found. Set uses a similar structure but eventually wraps the term with a Store term for writing. Empty expression is used for Void type where we do not want to do anything else.

Overall, the code generation pipeline is impacted by being extended into performing type checking and symbol resolution. L4 performs type consistency check, correct references check, symbol resolution and conversion, then converts the program into L3 form through code generation.

### Criteria of Success:
I consider the extension a success if it achieves a much more smooth and high level, abstract way of composing programs while providing strict type safety. Writing programs with less lines while representing the idea of the program with more understandable abstract terms is the true success. I want L4 type checker to reject incorrect programs before any code generation or execution. I want all L4 programs to compile into correct L3 programs with efficient and correct structure while symbol based code produces identical code to handwritten version. I want For and While loops to create valid corresponding L3 terms.


## L5 (Ruixiang)

### Extensions List

The major extension for L5 is Java-type classes implementation with inheritance.

The minor extensions for L5 are as follows:
- boolean operators `$$` and `||`
- `switch-case` branches
- `break` and `continue`
- `foreach`
  
### Justification of the Extension

The current L4 language already supports typed expressions, functions, mutable storage, structured data, and loops. However, several common programming patterns are still not naturally expressed.

The four minor extensions improve usability in direct and practical ways:

- `&&` and `||` make boolean logic more natural and safer through short-circuit behavior.
- `switch` provides a clearer and more structured alternative to deeply nested conditionals.
- `break` and `continue` make loops significantly easier to write and understand.
- `foreach` improves iteration over collections and reduces manual indexing stuffs.

The major extension, inheritance-based classes, addresses a different kind of need. It will envolve object-oritented programming, as inheritance features ensure hierarchical abstraction.

These extensions would benefit:

- compiler users writing larger examples or course projects, because the surface language becomes less verbose and easier to reason about;
- students learning compilation and lowering, because the new features provide concrete examples of how high-level constructs are translated into simpler core forms;
- users interested in structured programming and object-oriented design, because inheritance enables more realistic class hierarchies and method reuse.

### Description of the Extension

The usage of minor extensions is the same as current high-level programming languages like C++, Java, and Python.

For the two boolean operators, `&&` and `||` have the same functionalities as what are in C++ language. Similarly, the right-hand side is evaluated conditionally, for example, `a && b` evaluates `b` only if `a` is true.

The `switch` branch supports integer or boolean values. For example:

```
switch n {
  case 0 => 10
  case 1 => 20
  default => 99
}
switch flag {
  case true => 1
  default => 0
}
```

The `break` exits the nearest enclosing loop, while the `continue` skips the rest of the current iteration and proceeds to the next one. For example:

```
while cond {
  if stop {
    break
  }
  if skip {
    continue
  }
  work()
}
```

The `foreach` iterates over a list-like structure. For example:

```
foreach x in xs {
  process(x)
}
```

The class feature contains the following operations:

- Class declaration
  ```c
    class Point { // This creates a named type Point whose objects contain two integer fields.
        x: int;
        y: int;
    }
  ```
- Object construction
  ```c
  let p: Point = new Point(1, 2) in ... // This creates a Point object with x = 1 and y = 2.
  ```
- Field access
  ```c
  p.x // This retrieves the value of field x from object p.
  ```
- Field update
  ```c
  p.x = 5 // This modifies the value of field x from object p.
  ```
- Methods
  ```c
  class Point {
    x: int;
    y: int;

    def getX(): int { // This defines a function inside the class.
      this.x
    }
  }
  ```
- Inheritance
  ```c
  class ColoredPoint extends Point { // This defines a child class extended from Point
    color: int;

    def getColor(): int {
      this.color
    }
  }
  ```
  The subclass is able to reuse parent fields, inherit parent methods, and override inherited methods. Its value is allowed to be used where a parent type is expected.

### Impacts of the Extension

As for the four minor extensions, they will impact the syntax, type-checking and lowering. They are implemented using L4 languages, and they will interact with `for`, `while`, `int`, etc. in L4.

To implement class feature, it should type-check `new`, `this`, field access, method calls, etc. To erase high-level class constructions into lower-level representations, it indicates:
- objects are lowered into mutable payload representations;
- fields are laid out in fixed positions and their access and update become lower-level logic;
- methods become top-level functions;
- inherited fields are flattened to one extended layout.

Additionly, the class can use the other extensions at all levels.

### Success Criteria

The four minor extensions will be successful if:
- `&&` / `||`
  - only boolean operands are accepted
  - short-circuit behavior is preserved in lowering
- `switch-case`
  - scrutinee type is restricted appropriately
  - all branch result types are checked
  - lowering preserves branch meaning
- `break` / `continue`
  - they are rejected outside loops
  - they affect only the nearest loop
  - they work correctly inside loop bodies
- `foreach`
  - binder type checking works
  - target validation works
  - lowering correctly expands iteration structure
  - interaction with break / continue works

The class extension will be successful if:
- users can declare classes;
- users can construct objects;
- users can access and update fields;
- users can define methods;
- users can call methods on objects;
- lowering correctly translates objects and methods into lower-level forms;
- subclasses can extend parent classes;
- inherited fields and methods are visible where appropriate;
- overriding is checked correctly;
- subtype use is accepted where legal;
- invalid inheritance patterns are rejected;
- lowering remains sound.
  
## Timeline

For L4, Dogukan has finished all of the implementations.

For L5, Ruixiang has finished the four minor extensions and a non-inherited class extension. He will implement inheritance in the following 2 weeks.