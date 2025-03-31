from functools import partial
import json
from langchain_core.messages import BaseMessage, base
from langchain_core.runnables.base import RunnableLike
from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel
from typing import Any, Dict, Tuple
from pprint import pprint


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def pretty_print_message(msg: BaseMessage, html=False) -> str:
    title = base.get_msg_title_repr(msg.type.title() + " Message", bold=html)
    if msg.name is not None:
        title += f"\nName: {msg.name}"
    content = get_message_text(msg)
    print(f"{title}\n\n{content}")


def get_msg_title_repr(title: str) -> str:
    """Get a title representation for a message.
    Source: https://api.python.langchain.com/en/latest/_modules/langchain_core/messages/base.html#get_msg_title_repr

    Args:
        title: The title.
        bold: Whether to bold the title. Default is False.

    Returns:
        The title representation.
    """
    padded = (" " + title + " ") if title else ""
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep

    return f"{sep}{padded}{second_sep}"


def compare_objects(obj1: Any, obj2: Any, path: str = "") -> Dict[str, Any]:
    """
    Compare two objects and return a dictionary of differences.

    Args:
        obj1 (Any): The first object.
        obj2 (Any): The second object.
        path (str): The path to the object. Default is "".
    """
    differences = {}

    if isinstance(obj1, BaseModel) and isinstance(obj2, BaseModel):
        # Compare nested Pydantic models
        if obj1.__class__ != obj2.__class__:
            raise ValueError(
                "Both models must be of the same type. Instead got {obj1.__class__} and {obj2.__class__}."
            )

        field_schema_1: Dict[str, Any] = obj1.model_json_schema()["properties"]
        field_schema_2: Dict[str, Any] = obj2.model_json_schema()["properties"]
        assert (
            field_schema_1.keys() == field_schema_2.keys()
        ), f"Field schemas do not match: {field_schema_1} != {field_schema_2}"
        field_names = set(field_schema_1.keys()).union(field_schema_2.keys())

        for field_name in field_names:
            new_path = f"{path}.{field_name}" if path else field_name
            differences.update(
                compare_objects(
                    getattr(obj1, field_name, None),
                    getattr(obj2, field_name, None),
                    new_path,
                )
            )
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        # Compare dictionaries
        all_keys = set(obj1.keys()).union(obj2.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            differences.update(
                compare_objects(obj1.get(key), obj2.get(key), new_path),
            )
    elif isinstance(obj1, list) and isinstance(obj2, list):
        # Compare lists
        max_length = max(len(obj1), len(obj2))

        for idx in range(max_length):
            if idx < len(obj1) and idx < len(obj2):
                new_path = f"{path}[{idx}]"
                differences.update(
                    compare_objects(obj1[idx], obj2[idx], new_path),
                )
            elif idx < len(obj1):
                new_path = f"{path}[{idx}]"
                differences[new_path] = "removed"
            elif idx < len(obj2):
                new_path = f"{path}[{idx}]"
                differences[new_path] = {
                    "added": obj2[idx],
                }
    else:
        # Compare primitive types
        if obj1 != obj2:
            differences[path] = {"updated": obj2}

    # Handle fields that exist in one model but not the other
    if obj1 is None and obj2 is not None:
        differences[path] = {"added": obj2}
    elif obj2 is None and obj1 is not None:
        differences[path] = "removed"

    return differences

