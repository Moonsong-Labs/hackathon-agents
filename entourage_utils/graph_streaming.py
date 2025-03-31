import logging
from typing import Any, Dict, Generic, List, TypeVar
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel
from entourage_utils import logging_utils, json_utils

logger = logging.getLogger(__name__)

State = TypeVar("State", bound=BaseModel)


class StreamGraphUpdates(Generic[State]):
    """
    A class to handle streaming updates for a graph.

    Attributes:
        graph: The graph object that provides the stream of events.
        config: Configuration settings for the graph stream.
        verbosity_level: The verbosity level for the stream.
    Methods:
        __init__(graph, config):
            Initializes the StreamGraphUpdates with a graph and configuration.

        __call__(user_input: str):
            Processes user input and streams graph updates, printing new messages.
    """

    def __init__(
        self,
        graph,
        run_config,
    ):
        self.graph: CompiledGraph = graph
        self.run_config: Dict[str, Any] = run_config
        self.events: List[Dict[str, Any]] = []

    def print_event(self, event: dict, i=None, truncate=None):
        node_name: str = event["node"]
        updated_state: State = event["state"]
        if len(self.events) > 0:
            previous_event: Dict[str, Any] = self.events[-1]
            previous_state: State = previous_event["state"]
        else:
            previous_state = dict()
        logger.info(
            "previous state: " + str(type(previous_state)) + " - " + str(previous_state)
        )
        logger.info(
            "updated state: " + str(type(updated_state)) + " - " + str(updated_state)
        )
        diffs = logging_utils.compare_objects(
            previous_state,
            updated_state,
        )
        logger.info(f"diffs: {diffs}")
        logger.info(
            msg=f"Graph state update event:\n"
            + logging_utils.get_msg_title_repr(title=f"#{i} {node_name}")
            + "\n"
            + json_utils.dumps(diffs, truncate=truncate)
            + "\n"
            + logging_utils.get_msg_title_repr(title=None),
        )

    def __call__(self, initial_state: State, truncate=None) -> List[Dict]:
        event: List[str, Any]
        state_type: State = type(initial_state)
        for event in self.graph.stream(
            input=initial_state,
            config=self.run_config,
            stream_mode="updates",
        ):
            assert isinstance(event, dict)
            assert len(event) == 1, f"updates from more than one node: {event.keys()}"
            node = list(event.keys())[0]
            state: dict = state_type(**event[node]).model_dump()
            event = {"node": node, "state": state}
            self.print_event(event=event, i=len(self.events), truncate=truncate)
            self.events.append(event)


        return self.events
