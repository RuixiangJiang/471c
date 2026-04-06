The language is structured like a tree where the type Program is the root. The direction of execution goes from top to down, or root to leaves. Sequence type represents a collection of things. Program expects a sequence of identifiers, and a body that is of type Term.

There is a generalized type called Term that represents a union of types and employs the tag field for type resolution. Term type is used heavily among all types and enables the types to recursively include one another. Through using the generalized type Term, many more concrete types can be passed to each other as parameters.

Another important type is Identifier which is a non empty string. Identifier is used for type fields that we no longer want to extend. From the name itself, it should be unique at least locally. It is used as, Reference type's name, Let type's binding sequence's first Tuple element, Program's parameter sequence. L3 does not perform any checks to ensure uniqueness of Identifier however.

Branch type is used for control statements like switch and if. In L3, equals and less than operators are supported. Primitive type is used for operators like addition and multiplication and has a clear distinction between left and right terms.

Abstract type is for function definitions, where arguments must be a sequence of Identifiers. The body part of Abstract is where the actual implementation goes. Apply type is used to actually call a function. Reference type is used to access types by their identifiers, this enables access to a larger scope during implementation so the code does not have to repeat. 

Scope, or context is defined with Let and LetRec types. Bindings field is the context and the body field uses the bindings with Reference type inside the body field of Let. The difference between Let and LetRec types is Let sets the bindings immediately while LetRec initializes them and then sets them, so LetRec allows to recursively build the bindings and enables more complex expressions.

Allocate, Load, Store types are used for saving and retrieving information. When the context that is provided by Let is not enough or a data needs to live after the program finished, these types could be used for that purpose.
 