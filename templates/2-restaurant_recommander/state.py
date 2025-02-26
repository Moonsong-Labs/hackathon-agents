from enum import Enum
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field


class Status(str, Enum):
    in_progress = "in_progress"
    solved = "solved"
    failed = "failed"


class Request(BaseModel):
    """A request for a restaurant."""

    cuisine: Optional[str] = None
    location: Optional[str] = None
    max_price: Optional[str] = None
    min_rating: Optional[str] = None
    preferences: Optional[str] = None


class Restaurant(BaseModel):
    name: str
    address: str
    rating: str
    price: str
    url: str


class Reflection(BaseModel):
    is_successful: bool

    feedback: str


class State(BaseModel):
    """A state of the system, the class is mutable throughout the graph execution."""
    request: Request

    status: Optional[Status] = None

    restaurant: Optional[Restaurant] = None

    justification: Optional[str] = None

    reflection: Optional[Reflection] = None

    error: Optional[str] = None

    n_iteration: int = None
