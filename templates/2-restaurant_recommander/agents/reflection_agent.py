from typing import Optional
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from restaurant_agent.state import Reflection, Request, Restaurant
from entourage_poc.agents.chat_agent import ChatAgent


class ReflectionInput(BaseModel):
    """The input for the reflection agent."""

    request: Request

    restaurant: Optional[Restaurant]

    justification: Optional[str]

    error: Optional[str]


class ReflectionAgent(ChatAgent):
    """An agent that reflects on task execueted in current step."""

    def __init__(self, config):
        super().__init__(
            config=config, input_schema=ReflectionInput, output_schema=Reflection
        )

    def reflect(self, input: ReflectionInput, store: BaseStore) -> Reflection:
        return self.invoke_with_chat_history(
            input=input,
            store=store,
        )
