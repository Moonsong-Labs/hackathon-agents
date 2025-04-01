from pydantic import BaseModel
from typing import Dict
from dataclasses import field
from string import Template
from typing import Optional, Any

import re


class AgentConfig(BaseModel):
    """The base configuration of an agent."""

    prompt_templates: Dict[str, str] = field(
        metadata={
            "description": "A dictionary of prompt templates defined for this agent"
        },
        default_factory=dict,
    )

    model: str = field(
        metadata={"description": "The name of the language model to use."},
    )

    provider: str = field(
        metadata={
            "description": "The name of the provider of the language model to use.",
        },
    )

    provider_config: dict[str, Any] = field(
        default={},
        metadata={
            "description": "Additional configuration for the provider.",
        },
    )

    temperature: float = field(
        default=0.0,
        metadata={"description": "The model temperature."},
    )

    max_tokens: Optional[int] = field(
        default=None,
        metadata={"description": "The maximum number of tokens to generate."},
    )

    def get_prompt(self, tag: str, **kwargs) -> str:
        assert tag in self.prompt_templates, (
            f"missing tag: {tag} in {self.prompt_templates.keys()}"
        )
        template = Template(self.prompt_templates[tag])

        keys = kwargs.keys()
        identifiers = template.get_identifiers()

        assert set(keys) == set(
            identifiers
        ), f"Keyword args not matching identifiers, keys = {keys}, identifiers = {
            identifiers
        }"

        assert all([v is not None for v in kwargs.values()]), (
            f"Some values are None, kwargs = {kwargs}"
        )
        kwargs = kwargs.copy() if kwargs else {}
        for kwarg in kwargs:
            if isinstance(kwargs[kwarg], BaseModel):
                kwargs[kwarg] = kwargs[kwarg].model_dump_json()

        return template.substitute(**kwargs)

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for the given parameters."""
        return self.get_prompt("system", **kwargs)

    def get_user_prompt(self, **kwargs) -> str:
        """Get the system prompt for the given parameters."""
        return self.get_prompt("user", **kwargs)


class BaseAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.agent_id = self.__class__.__name__

        def camel_to_snake(name: str) -> str:
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        self.agent_id = camel_to_snake(self.__class__.__name__)
