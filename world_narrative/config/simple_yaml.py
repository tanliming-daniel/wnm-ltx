from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass
class _Line:
    indent: int
    text: str


def _strip_comment(line: str) -> str:
    out: list[str] = []
    in_single = False
    in_double = False
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _clean_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for raw in text.splitlines():
        line = _strip_comment(raw.rstrip("\n"))
        if not line.strip():
            continue
        if "\t" in line:
            raise ValueError("tabs are not supported in config indentation")
        indent = len(line) - len(line.lstrip(" "))
        lines.append(_Line(indent=indent, text=line.lstrip(" ")))
    return lines


def _parse_scalar(text: str) -> Any:
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"~"}:
        return None
    if text.startswith("[") or text.startswith("{") or text.startswith("("):
        return ast.literal_eval(text)
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"invalid mapping line: {text!r}")
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _parse_block(lines: list[_Line], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start
    first = lines[start]
    if first.indent < indent:
        return {}, start
    if first.text.startswith("- "):
        items: list[Any] = []
        i = start
        while i < len(lines):
            line = lines[i]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise ValueError(f"unexpected indentation for list item: {line.text!r}")
            if not line.text.startswith("- "):
                break
            item_text = line.text[2:].strip()
            i += 1
            if item_text == "":
                child, i = _parse_block(lines, i, indent + 2)
                items.append(child)
                continue
            if ":" in item_text and not item_text.startswith(("[", "{", "(")):
                key, value = _split_key_value(item_text)
                item: dict[str, Any] = {}
                if value:
                    item[key] = _parse_scalar(value)
                else:
                    child, i = _parse_block(lines, i, indent + 2)
                    item[key] = child
                    items.append(item)
                    continue
                if i < len(lines) and lines[i].indent > indent:
                    child, i = _parse_block(lines, i, lines[i].indent)
                    if isinstance(child, dict):
                        item.update(child)
                    else:
                        item[key] = child
                items.append(item)
            else:
                items.append(_parse_scalar(item_text))
                if i < len(lines) and lines[i].indent > indent:
                    child, i = _parse_block(lines, i, lines[i].indent)
                    if child not in ({}, []):
                        items[-1] = child
        return items, i

    mapping: dict[str, Any] = {}
    i = start
    while i < len(lines):
        line = lines[i]
        if line.indent < indent:
            break
        if line.indent > indent:
            raise ValueError(f"unexpected indentation for mapping entry: {line.text!r}")
        if line.text.startswith("- "):
            break
        key, value = _split_key_value(line.text)
        i += 1
        if value == "":
            if i < len(lines) and lines[i].indent > indent:
                child, i = _parse_block(lines, i, lines[i].indent)
            else:
                child = {}
            mapping[key] = child
        else:
            mapping[key] = _parse_scalar(value)
            if i < len(lines) and lines[i].indent > indent:
                child, i = _parse_block(lines, i, lines[i].indent)
                if isinstance(mapping[key], dict) and isinstance(child, dict):
                    mapping[key].update(child)
                elif isinstance(mapping[key], list) and isinstance(child, list):
                    mapping[key].extend(child)
                else:
                    mapping[key] = child
    return mapping, i


def simple_yaml_load(text: str) -> Any:
    lines = _clean_lines(text)
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0].indent)
    return value
