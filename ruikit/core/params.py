"""Tool parameters: a dataclass in, JSON Schema out, and validated arguments back.

A tool declares its parameters once, as a dataclass whose fields carry a short
description (``param(...)``).  From that one declaration come

* the JSON Schema every provider accepts -- plain, or in the *strict* shape
  (every property required, optional ones nullable, no extra properties) that
  OpenAI's and Anthropic's strict tool use want;
* the Python signature a program sees inside ``code_run``;
* the validation of what a model actually sent, with an error that names the
  field and the legal range, so the model can correct itself.

Only the types a tool call can carry are supported: str, int, float, bool,
``Literal``, ``Optional``, lists, fixed-length tuples and nested dataclasses.
Anything else is a programming error, raised when the tool is registered.
"""

from __future__ import annotations

import copy
import dataclasses
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Literal, Mapping, Tuple, Union, get_args, get_origin

__all__ = ["param", "json_schema", "parse_args", "ArgError", "describe_params", "NoParams"]


class ArgError(ValueError):
    """An argument the schema does not accept.  ``path`` names the field."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


def param(desc: str, *, default: Any = MISSING, minimum: float | None = None,
          maximum: float | None = None, min_items: int | None = None,
          max_items: int | None = None, hidden: bool = False):
    """A documented dataclass field.

    ``hidden`` fields are never shown to a model (the interface fixes them, e.g.
    the step sizes of the step interface); they still validate if given.
    """
    md = {"desc": desc}
    for key, value in (("minimum", minimum), ("maximum", maximum),
                       ("min_items", min_items), ("max_items", max_items)):
        if value is not None:
            md[key] = value
    if hidden:
        md["hidden"] = True
    if default is MISSING:
        return field(metadata=md)
    if isinstance(default, (list, dict, set)):
        return field(default_factory=lambda d=default: copy.deepcopy(d), metadata=md)
    return field(default=default, metadata=md)


@dataclass(frozen=True)
class NoParams:
    """For tools that take no arguments."""


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def _is_optional(tp) -> Tuple[bool, Any]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != len(get_args(tp)):
            return True, (args[0] if len(args) == 1 else Union[tuple(args)])
    return False, tp


def _type_schema(tp, strict: bool) -> dict:
    optional, inner = _is_optional(tp)
    if optional:
        base = _type_schema(inner, strict)
        if "type" in base and isinstance(base["type"], str) and "enum" not in base:
            return {**base, "type": [base["type"], "null"]}
        return {"anyOf": [base, {"type": "null"}]}
    origin = get_origin(inner)
    if inner is str:
        return {"type": "string"}
    if inner is bool:
        return {"type": "boolean"}
    if inner is int:
        return {"type": "integer"}
    if inner is float:
        return {"type": "number"}
    if origin is Literal:
        values = list(get_args(inner))
        kind = "string" if all(isinstance(v, str) for v in values) else "number"
        return {"type": kind, "enum": values}
    if origin in (list, typing.List, tuple, typing.Tuple):
        args = get_args(inner)
        if origin in (tuple, typing.Tuple) and args and args[-1] is not Ellipsis:
            if len(set(args)) != 1:
                raise TypeError(f"tuples must be homogeneous: {inner}")
            return {"type": "array", "items": _type_schema(args[0], strict),
                    "minItems": len(args), "maxItems": len(args)}
        item = args[0] if args else Any
        return {"type": "array", "items": _type_schema(item, strict)}
    if is_dataclass(inner):
        return json_schema(inner, strict=strict)
    raise TypeError(f"unsupported parameter type {inner!r}")


def json_schema(cls, strict: bool = False, hide: Tuple[str, ...] = ()) -> dict:
    """The JSON Schema of a parameter dataclass.

    Plain shape: a property is required when it has no default.  Strict shape:
    every property is required, optional ones become nullable, and a null means
    "use the default" when the arguments are parsed.
    """
    hints = typing.get_type_hints(cls)
    props: Dict[str, dict] = {}
    required = []
    for f in fields(cls):
        if f.metadata.get("hidden") or f.name in hide:
            continue
        schema = _type_schema(hints[f.name], strict)
        has_default = f.default is not MISSING or f.default_factory is not MISSING
        if strict and has_default and "null" not in str(schema.get("type", "")) \
                and "anyOf" not in schema:
            if isinstance(schema.get("type"), str) and "enum" not in schema:
                schema = {**schema, "type": [schema["type"], "null"]}
            else:
                schema = {"anyOf": [schema, {"type": "null"}]}
        desc = f.metadata.get("desc")
        if desc:
            schema = {**schema, "description": desc}
        for key, target in (("minimum", "minimum"), ("maximum", "maximum"),
                            ("min_items", "minItems"), ("max_items", "maxItems")):
            if key in f.metadata:
                schema[target] = f.metadata[key]
        props[f.name] = schema
        if strict or not has_default:
            required.append(f.name)
    out = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


def describe_params(cls) -> str:
    """``name: type -- description`` lines, for code-mode docstrings."""
    hints = typing.get_type_hints(cls)
    lines = []
    for f in fields(cls):
        if f.metadata.get("hidden"):
            continue
        tp = hints[f.name]
        name = getattr(tp, "__name__", None) or str(tp).replace("typing.", "")
        default = ""
        if f.default is not MISSING:
            default = f" = {f.default!r}"
        elif f.default_factory is not MISSING:
            default = f" = {f.default_factory()!r}"
        lines.append(f"{f.name}: {name}{default} -- {f.metadata.get('desc', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _parse_value(tp, value, path: str):
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise ArgError(path, "must not be null")
    origin = get_origin(inner)
    if inner is bool:
        if isinstance(value, bool):
            return value
        raise ArgError(path, f"expected true or false, got {value!r}")
    if inner is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                (isinstance(value, float) and not value.is_integer()):
            raise ArgError(path, f"expected an integer, got {value!r}")
        return int(value)
    if inner is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArgError(path, f"expected a number, got {value!r}")
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            raise ArgError(path, "must be a finite number")
        return out
    if inner is str:
        if not isinstance(value, str):
            raise ArgError(path, f"expected a string, got {value!r}")
        return value
    if origin is Literal:
        allowed = get_args(inner)
        if value not in allowed:
            raise ArgError(path, f"must be one of {list(allowed)}, got {value!r}")
        return value
    if origin in (list, typing.List, tuple, typing.Tuple):
        if not isinstance(value, (list, tuple)):
            raise ArgError(path, f"expected a list, got {value!r}")
        args = get_args(inner)
        if origin in (tuple, typing.Tuple) and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ArgError(path, f"expected exactly {len(args)} values, got {len(value)}")
            return tuple(_parse_value(a, v, f"{path}[{i}]") for i, (a, v) in
                         enumerate(zip(args, value)))
        item = args[0] if args else Any
        out = [_parse_value(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(out) if origin in (tuple, typing.Tuple) else out
    if is_dataclass(inner):
        if isinstance(value, str) and hasattr(inner, "from_text"):
            # a short form some objects have (a target as its object id)
            try:
                value = inner.from_text(value)
            except ValueError as e:
                raise ArgError(path, str(e)) from None
        if not isinstance(value, Mapping):
            raise ArgError(path, f"expected an object, got {value!r}")
        return parse_args(inner, value, prefix=path)
    if inner is Any:
        return value
    raise ArgError(path, f"unsupported type {inner!r}")


def parse_args(cls, raw: Mapping | None, prefix: str = ""):
    """Validate ``raw`` against the dataclass and build it.

    Unknown keys are an error (a misspelt argument must not silently fall back
    to its default); null means "the default"; numeric ranges and list lengths
    declared with ``param`` are enforced.
    """
    raw = dict(raw or {})
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise ArgError(f"{prefix}.{unknown[0]}" if prefix else unknown[0],
                       f"unknown argument; this tool takes {sorted(n for n in names if not _hidden(cls, n))}")
    values = {}
    for f in fields(cls):
        path = f"{prefix}.{f.name}" if prefix else f.name
        has_default = f.default is not MISSING or f.default_factory is not MISSING
        if f.name not in raw or (raw[f.name] is None and has_default
                                 and not _is_optional(hints[f.name])[0]):
            if not has_default:
                raise ArgError(path, "is required")
            continue
        value = _parse_value(hints[f.name], raw[f.name], path)
        md = f.metadata
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in md and value < md["minimum"]:
                raise ArgError(path, f"must be >= {md['minimum']}, got {value}")
            if "maximum" in md and value > md["maximum"]:
                raise ArgError(path, f"must be <= {md['maximum']}, got {value}")
        if isinstance(value, (list, tuple)):
            if "min_items" in md and len(value) < md["min_items"]:
                raise ArgError(path, f"needs at least {md['min_items']} items")
            if "max_items" in md and len(value) > md["max_items"]:
                raise ArgError(path, f"takes at most {md['max_items']} items")
        values[f.name] = value
    return cls(**values)


def _hidden(cls, name: str) -> bool:
    for f in fields(cls):
        if f.name == name:
            return bool(f.metadata.get("hidden"))
    return False


def as_dict(obj) -> dict:
    """A parameter dataclass back to plain JSON values (for logs)."""
    return dataclasses.asdict(obj) if is_dataclass(obj) else dict(obj or {})
