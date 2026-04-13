from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import syntax as L5


@dataclass(frozen=True)
class FieldInfo:
    name: str
    typeof: L5.Type
    owner: str


@dataclass(frozen=True)
class MethodInfo:
    name: str
    parameters: Sequence[tuple[str, L5.Type]]
    returns: L5.Type
    body: L5.Expression
    owner: str


@dataclass(frozen=True)
class ClassInfo:
    name: str
    parent: str | None
    own_fields: Sequence[FieldInfo]
    all_fields: Sequence[FieldInfo]
    own_methods: Mapping[str, MethodInfo]
    methods: Mapping[str, MethodInfo]


def _type_error(message: str) -> TypeError:
    return TypeError(message)


def field_index(info: ClassInfo, field: str) -> int:
    for i, f in enumerate(info.all_fields):
        if f.name == field:
            return i
    raise _type_error(f"class {info.name!r} has no field {field!r}")


def field_info(info: ClassInfo, field: str) -> FieldInfo:
    for f in info.all_fields:
        if f.name == field:
            return f
    raise _type_error(f"class {info.name!r} has no field {field!r}")


def method_info(info: ClassInfo, method: str) -> MethodInfo:
    if method not in info.methods:
        raise _type_error(f"class {info.name!r} has no method {method!r}")
    return info.methods[method]


def _same_signature(parent_method: MethodInfo, child_method: MethodInfo) -> bool:
    return (
        list(parent_method.parameters) == list(child_method.parameters)
        and parent_method.returns == child_method.returns
    )


def collect_classes(classes: Sequence[L5.ClassDef]) -> dict[str, ClassInfo]:
    raw: dict[str, L5.ClassDef] = {}
    for cls in classes:
        if cls.name in raw:
            raise _type_error(f"duplicate class definition for {cls.name!r}")
        raw[cls.name] = cls

    resolved: dict[str, ClassInfo] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> ClassInfo:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise _type_error(f"cyclic inheritance involving {name!r}")
        if name not in raw:
            raise _type_error(f"unknown parent class {name!r}")

        visiting.add(name)
        cls = raw[name]

        parent_info: ClassInfo | None = None
        if cls.parent is not None:
            parent_info = resolve(cls.parent)

        inherited_field_names = {f.name for f in parent_info.all_fields} if parent_info else set()
        inherited_methods = dict(parent_info.methods) if parent_info else {}

        seen_field_names: set[str] = set()
        own_fields: list[FieldInfo] = []
        for field in cls.fields:
            if field.name in seen_field_names:
                raise _type_error(f"duplicate field {field.name!r} in class {cls.name!r}")
            if field.name in inherited_field_names:
                raise _type_error(
                    f"field {field.name!r} in class {cls.name!r} conflicts with inherited field"
                )
            seen_field_names.add(field.name)
            own_fields.append(FieldInfo(name=field.name, typeof=field.typeof, owner=cls.name))

        seen_method_names: set[str] = set()
        own_methods: dict[str, MethodInfo] = {}
        resolved_methods = dict(inherited_methods)

        for method in cls.methods:
            if method.name in seen_method_names:
                raise _type_error(f"duplicate method {method.name!r} in class {cls.name!r}")
            seen_method_names.add(method.name)

            current = MethodInfo(
                name=method.name,
                parameters=method.parameters,
                returns=method.returns,
                body=method.body,
                owner=cls.name,
            )

            if method.name in inherited_methods:
                inherited = inherited_methods[method.name]
                if not _same_signature(inherited, current):
                    raise _type_error(
                        f"method {cls.name}.{method.name} overrides inherited method with incompatible signature"
                    )

            own_methods[method.name] = current
            resolved_methods[method.name] = current

        info = ClassInfo(
            name=cls.name,
            parent=cls.parent,
            own_fields=tuple(own_fields),
            all_fields=tuple((parent_info.all_fields if parent_info else ())) + tuple(own_fields),
            own_methods=own_methods,
            methods=resolved_methods,
        )

        resolved[name] = info
        visiting.remove(name)
        return info

    for name in raw:
        resolve(name)

    return resolved