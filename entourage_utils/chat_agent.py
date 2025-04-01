import logging
import traceback

from jsonschema import ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.chat_models import init_chat_model
from typing import Any, Generic, List, Optional, Tuple, TypeVar
from langgraph.store.base import BaseStore, Item
from pydantic import BaseModel
from entourage_utils.agent import BaseAgent, AgentConfig
from entourage_utils import json_utils

logger = logging.getLogger(__name__)
# input and output must be pydantic.BaseModel classes
Input = TypeVar("Input", bound=BaseModel)

Output = TypeVar("Output", bound=BaseModel)


class LLMOutputValidationError(ValueError):
    def __init__(self, *args):
        super().__init__(*args)


class ChatAgent(BaseAgent, Generic[Input, Output]):
    """An agent that wraps an LLM as chat model."""

    def __init__(
        self,
        config: AgentConfig,
        input_schema: Input = None,
        output_schema: Output = None,
        system_prompt_kwargs: dict = None,
        max_validation_trial_fix: int = 3,
        max_validation_trail_redo: int = 3,
        max_chat_history_length: int = 10,
    ) -> None:
        super().__init__(config)

        self.llm: BaseChatModel = self.load_chat_model()

        if input_schema:
            assert issubclass(input_schema, BaseModel), (
                "input_model must be a pydantic BaseModel type"
            )
        self.input_schema: Input = input_schema
        self.output_schema: Output = output_schema
        if output_schema:
            assert issubclass(output_schema, BaseModel), (
                "output_model must be a pydantic BaseModel type"
            )

            self.llm = self.llm.with_structured_output(
                self.output_schema, include_raw=True
            )

        self.max_validation_trial_fix = max_validation_trial_fix
        self.max_validation_trail_redo = max_validation_trail_redo
        self.system_prompt = self.get_system_prompt(
            **system_prompt_kwargs if system_prompt_kwargs else {},
        )
        self.max_chat_history_length = max_chat_history_length

    def get_system_prompt(self, **kwargs):
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()
        if self.input_schema:
            kwargs["input_schema"] = self.input_schema.model_json_schema()
        if self.output_schema:
            kwargs["output_schema"] = self.output_schema.model_json_schema()
        return self.config.get_system_prompt(**kwargs)

    def get_system_message(self):
        return SystemMessage(content=self.system_prompt, name="system")

    def validate_output(self, input: Input, output: Output):
        if self.output_schema:
            try:
                json_utils.validate_instance(
                    output.model_dump(), self.output_schema.model_json_schema()
                )
            except ValidationError as e:
                raise LLMOutputValidationError(
                    f"Output not matching the schema: {e}. Output: {output}, schema: {
                        self.output_schema
                    }"
                )
        return True

    def messages_excerpt(self, messages: List[BaseMessage]):
        return json_utils.dumps(
            [
                ({"type": msg.type, "name": msg.name, "content": msg.content})
                for msg in messages
                if msg.content
            ]
        )

    def add_message(self, content: str, store: BaseStore, name: str = None):
        content = HumanMessage(content, name=name)

        self.append_to_chat_history([content], store=store)

        logger.debug(
            f"Added human message to {
                self.agent_id
            } chat history, new state of messages: {
                self.messages_excerpt(self.load_chat_history(store=store))
            }"
        )
        return self

    def add_reflection_message(self, reflection: str, store: BaseStore):
        return self.add_message(
            content=self.config.get_prompt(tag="reflection", reflection=reflection),
            store=store,
            name="reflection",
        )

    def invoke(
        self,
        input: Any,
        prompt_tag: str = "user",
        **prompt_kwargs,
    ) -> Output:
        system_message = self.get_system_message()

        user_message = HumanMessage(
            self.config.get_prompt(tag=prompt_tag, input=input, **prompt_kwargs),
            name=prompt_tag,
        )

        if self.output_schema:
            output, _ = self._invoke_structured(
                input=input,
                system_message=system_message,
                user_message=user_message,
                chat_history=[],
            )
        else:
            output, _ = self._invoke_unstructured(
                system_message=system_message,
                user_message=user_message,
                chat_history=[],
            )

        return output

    def invoke_with_chat_history(
        self,
        input,
        store: BaseStore,
        prompt_tag="user",
        **prompt_kwargs,
    ) -> Output:
        if store is None:
            logger.warning("Falling back to the invokation without chat history")
            return self.invoke(input=input, prompt_tag=prompt_tag, **prompt_kwargs)
        chat_history: List[BaseMessage] = self.load_chat_history(store=store)

        system_message: SystemMessage = self.get_system_message()

        user_message = HumanMessage(
            self.config.get_prompt(tag=prompt_tag, input=input, **prompt_kwargs),
            name=prompt_tag,
        )

        if self.output_schema:
            output, output_message = self._invoke_structured(
                input=input,
                system_message=system_message,
                user_message=user_message,
                chat_history=chat_history,
            )
        else:
            output, output_message = self._invoke_unstructured(
                system_message=system_message,
                user_message=user_message,
                chat_history=chat_history,
            )
        new_messages = [user_message, output_message]

        self.append_to_chat_history(store=store, messages=new_messages)

        return output

    def _invoke_unstructured(
        self,
        system_message: SystemMessage,
        user_message: HumanMessage,
        chat_history: List[BaseMessage],
    ) -> Tuple[str, AIMessage]:
        messages_input = [system_message] + chat_history + [user_message]
        logger.info(
            f"Agent {self.agent_id}: Invoking structured LLM with {
                len(messages_input)
            } messages = {self.messages_excerpt(messages_input)}"
        )
        logger.debug(
            f"Agent {self.agent_id}: Invoking structured LLM with {
                len(messages_input)
            } messages = {messages_input}"
        )
        output_message = self.llm.invoke(messages_input)
        assert output_message.type == "ai"
        output_message.name = "unstructured_response"
        logger.debug(f"Agent {self.agent_id}: Response from LLM: {output_message}")
        assert isinstance(output_message, AIMessage)
        output: str = output_message.content
        return output, output_message

    def _invoke_structured(
        self,
        input: Input,
        system_message: SystemMessage,
        user_message: HumanMessage,
        chat_history: List[BaseMessage],
        error: LLMOutputValidationError = None,
        n_trial_fix: int = 0,
        n_trial_redo: int = 0,
    ) -> Tuple[Output, AIMessage]:
        messages_input = (
            [system_message] + chat_history + ([user_message] if error is None else [])
        )
        logger.info(
            f"Agent {self.agent_id}: Invoking structured LLM with {
                len(messages_input)
            } messages = {self.messages_excerpt(messages_input)}"
        )
        logger.debug(
            f"Agent {self.agent_id}: Invoking structured LLM with {
                len(messages_input)
            } messages = {messages_input}"
        )

        output = self.llm.invoke(messages_input)
        logger.debug(f"Agent {self.agent_id}: Response from LLM: {output}")
        assert isinstance(output, dict)
        assert output["raw"].type == "ai"
        output_message: AIMessage = output["raw"]

        output: Output = output["parsed"]
        assert isinstance(output, self.output_schema), (
            f"Invalid output type:{type(output)}: {output}"
        )
        output_message.name = "structured_response"

        try:
            self.validate_output(input=input, output=output)
            return output, output_message
        except LLMOutputValidationError as error:
            logger.info(f"Validation error: {error}")
            logger.debug(traceback.format_exc())
            if n_trial_fix < self.max_validation_trial_fix:
                n_trial_fix = n_trial_fix + 1
                error_message: str = HumanMessage(
                    self.config.get_prompt(
                        tag="validation_error",
                        error_message=str(error),
                    ),
                    name="validation_error",
                )
                chat_history = chat_history + [output_message, error_message]

            elif n_trial_redo < self.max_validation_trail_redo:
                n_trial_fix = 0
                n_trial_redo = n_trial_redo + 1
                error = None
                chat_history = []
            else:
                raise error
            logger.info(
                f"Trying to fix the validation error: n_trial_fix = {
                    n_trial_fix
                }, n_trial_redo = {n_trial_redo}"
            )
            output, output_message = self._invoke_structured(
                input=input,
                system_message=system_message,
                user_message=user_message,
                chat_history=chat_history,
                error=error,
                n_trial_fix=n_trial_fix,
                n_trial_redo=n_trial_redo,
            )

            return output, output_message

    def load_chat_history(
        self, store: BaseStore, max_chat_history_length=None
    ) -> List[BaseMessage]:
        item: Optional[Item] = store.get(namespace=self.agent_id, key="chat_history")

        if item is not None:
            value = item.value
            assert "messages" in value
            messages = value["messages"]
            assert isinstance(messages, list)
            if max_chat_history_length is None:
                max_chat_history_length = self.max_chat_history_length
            return messages[-max_chat_history_length:]
        else:
            return []

    def append_to_chat_history(self, messages: List[BaseMessage], store: BaseStore):
        prev_messages = self.load_chat_history(store)
        messages = prev_messages + messages
        store.put(
            namespace=self.agent_id, key="chat_history", value={"messages": messages}
        )
        return self

    def load_chat_model(self) -> BaseChatModel:
        """Load a chat model from a fully specified name."""

        config = self.config

        llm = init_chat_model(
            config.model,
            model_provider=config.provider,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=config.provider_config.get("base_url"),
            api_key=config.provider_config.get("api_key"),
        )
        return llm
