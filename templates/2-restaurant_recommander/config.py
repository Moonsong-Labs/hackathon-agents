import os
from typing import Any, Dict, List
from entourage_poc.agents.agent import AgentConfig
from entourage_poc import prompt_utils
from pydantic import BaseModel


def agent_config(prompt_directory, prompt_versions, agent_name) -> AgentConfig:
    return AgentConfig(
        **load_prompt_config(
            prompt_directory=prompt_directory,
            prompt_versions=prompt_versions,
            agent_name=agent_name,
        )
    )


def load_prompt_config(prompt_directory, prompt_versions, agent_name) -> Dict[str, Any]:
    prompt_version = prompt_versions[agent_name]
    prompt_file = os.path.join(prompt_directory, agent_name, f"{prompt_version}.prompt")
    prompt_config = prompt_utils.load_prompt_config(prompt_file)
    return prompt_config


DEFAULT_RUN_CONFIG = {
    "configurable": {"thread_id": 102},
    "recursion_limit": 500,
}


class GraphConfig(BaseModel):
    """The configuration for the main application."""

    agent_names: List[str] = [
        "restaurant_recommender",
        "reflection",
    ]

    prompt_directory: str = os.path.join(os.path.dirname(__file__), "prompts")

    prompt_versions: Dict[str, str] = {agent_id: "v1" for agent_id in agent_names}

    agent_configs: Dict[str, AgentConfig] = None

    max_retry_iterations: int = 5

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.agent_configs = {
            f"{agent_name}_agent": agent_config(
                prompt_directory=self.prompt_directory,
                prompt_versions=self.prompt_versions,
                agent_name=agent_name,
            )
            for agent_name in self.agent_names
        }
        return self
