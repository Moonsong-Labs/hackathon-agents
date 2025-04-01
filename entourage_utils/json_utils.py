from abc import ABC, abstractmethod
import dataclasses
import json
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, override
import typing
from genson import SchemaBuilder
import jsonschema
from jsonschema.protocols import Validator
import jsonschema.validators
from pydantic import BaseModel, create_model

from typing import Any, Dict, Generic, List, TypeVar, Union
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

DEFAULT_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"


Description = Union[str, Tuple[str, Optional[Union[Dict[str, "Description"], str]]]]


def validate_schema(
    schema: Union[str, dict],
    check_object_type=True,
) -> bool:
    assert schema is not None
    if isinstance(schema, str):
        schema = json.loads(schema)
    validator: Validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)

    if check_object_type:
        if "type" not in schema or schema["type"] != "object":
            raise jsonschema.SchemaError(
                f"The generated schema is not an object type: {schema}"
            )

    return True


def validate_instance(instance: Union[str, dict], schema: Union[str, dict]):
    if isinstance(instance, str):
        instance = json.loads(instance)

    if isinstance(schema, str):
        schema = json.loads(schema)
    assert isinstance(instance, dict)
    assert isinstance(schema, dict)

    if schema.get("properties"):
        jsonschema.validate(instance=instance, schema=schema)
    else:
        if not len(instance) == 0:
            raise jsonschema.ValidationError(
                f"Non empty instance with an empty schema, instance: {instance}, schema: {schema}"
            )


def schema_json_from_instance(
    instance: dict,
    schema_uri=DEFAULT_SCHEMA_URI,
) -> str:
    if isinstance(instance, str):
        instance = json.loads(instance)
    builder = SchemaBuilder(schema_uri)
    builder.add_object(instance)
    schema = builder.to_schema()

    validate_schema(schema)
    return json.dumps(schema)


def schema_json_from_model(model: BaseModel, schema_uri=DEFAULT_SCHEMA_URI) -> str:
    return json.dumps(model.model_json_schema(schema_uri))


def truncate_string(s: str, max_length: int) -> str:
    s = str(s)
    if max_length is None:
        return s
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s


def convert_to_serializable(obj: Any, truncate: int = None) -> Union[Dict, List, str]:
    if obj is None:
        return None
    dump_json = None
    if isinstance(obj, BaseModel):
        d = obj.model_dump(exclude_none=True)
        return convert_to_serializable(d, truncate=truncate)
    elif isinstance(obj, dict):
        dump_json = {
            k: convert_to_serializable(v, truncate=truncate) for k, v in obj.items()
        }
    elif isinstance(obj, list):
        dump_json = [convert_to_serializable(v, truncate=truncate) for v in obj]
    elif isinstance(obj, (int, float, bool)):
        dump_json = obj
    elif isinstance(obj, str):
        dump_json = truncate_string(obj, truncate)
    else:
        obj_str = json.dumps(obj, default=str)
        dump_json = truncate_string(obj_str, truncate)

    return dump_json


def dumps(d: Any, indent=2, truncate=100):
    d = convert_to_serializable(d, truncate=truncate)
    return json.dumps(d, indent=indent)
