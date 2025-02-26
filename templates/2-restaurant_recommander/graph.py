from functools import partial
import logging
from typing import Literal
import uuid


from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.base import RunnableLike
from langgraph.store.memory import InMemoryStore
from restaurant_agent.state import State, Status
from restaurant_agent.agents.restaurant_recommender_agent import (
    RestaurantRecommendation,
    RestaurantRecommenderAgent,
)
from restaurant_agent.agents.reflection_agent import (
    ReflectionInput,
    ReflectionAgent,
)
from restaurant_agent.config import GraphConfig

logger = logging.getLogger(__name__)


# NODES


def initialize(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> State:
    state.status = Status.in_progress
    state.n_iteration = 0
    return state


def recommend_restaurant(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
    agent: RestaurantRecommenderAgent,
) -> State:
    if state.reflection:
        agent.add_reflection_message(reflection=state.reflection.feedback, store=store)
    output: RestaurantRecommendation = agent.recommend_restaurant(
        request=state.request, store=store
    )
    state.restaurant = output.restaurant
    state.justification = output.justification
    state.error = output.error
    state.n_iteration += 1
    return state


def reflect(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
    agent: ReflectionAgent,
) -> State:
    input = ReflectionInput(
        request=state.request,
        restaurant=state.restaurant,
        justification=state.justification,
        error=state.error,
    )
    reflection = agent.reflect(input=input, store=store)
    state.reflection = reflection
    return state


# Update status at the end of each step
def decide_next_step(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
    max_retry_iterations: int,
) -> State:
    if state.reflection.is_successful:
        state.status = Status.solved
    else:
        if state.n_iteration < max_retry_iterations:
            state.status = Status.in_progress
        else:
            state.status = Status.failed
            state.error = "Max iterations exceeded"

    return state


# CONDITIONAL EDGES


def route_by_status(state: State) -> Literal[
    "recommend_restaurant",
    "__end__",
]:
    if state.status == Status.in_progress:
        return "recommend_restaurant"
    else:
        return END


# GRAPH


def get_graph(config: GraphConfig) -> CompiledGraph:
    """
    Get the graph for the entourage POC.

    Returns:
        CompiledGraph: The compiled graph object.
    """

    graph = StateGraph(state_schema=State)

    graph_id = uuid.uuid4()
    # Agents
    restaurant_recommender_agent = RestaurantRecommenderAgent(
        config=config.agent_configs["restaurant_recommender_agent"]
    )
    reflection_agent = ReflectionAgent(config=config.agent_configs["reflection_agent"])

    # Nodes
    class Node:
        def print_log_when_executing(self, func, name):
            def wrapper(state: State, config: RunnableConfig, store: BaseStore):
                logger.info(f">> Executing graph node: {name}")
                result = func(state=state, config=config, store=store)
                # executor.print_logs(result)
                return result

            return wrapper

        def __init__(self, func: RunnableLike, **kwargs):
            self.key = func
            self.name = func.__name__
            if kwargs:
                node = partial(func, **kwargs)
            else:
                node = func
            self.node = self.print_log_when_executing(func=node, name=self.name)

    nodes = [
        Node(initialize),
        Node(recommend_restaurant, agent=restaurant_recommender_agent),
        Node(
            reflect,
            agent=reflection_agent,
        ),
        Node(decide_next_step, max_retry_iterations=config.max_retry_iterations),
    ]
    for node in nodes:
        graph.add_node(node.name, node.node)

    # Edges

    graph.add_edge(START, initialize.__name__)
    graph.add_edge(initialize.__name__, recommend_restaurant.__name__)
    graph.add_edge(recommend_restaurant.__name__, reflect.__name__)
    graph.add_edge(reflect.__name__, decide_next_step.__name__)

    graph.add_conditional_edges(decide_next_step.__name__, route_by_status)

    # Graph compilation
    checkpointer = MemorySaver()
    in_memory_store = InMemoryStore()

    graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
    graph.name = "Restaurant Recommendation Agent Template"

    return graph
