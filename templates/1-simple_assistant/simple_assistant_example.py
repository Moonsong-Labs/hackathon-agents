#!/usr/bin/env python3
"""
Simple Assistant Example

This template is a simple exemple on how to use the LangGraph combined with entourage utils.

It is a simple assistant that answers questions directly without external tools
The graph is a directl flow with 2 nodes and 1 edge:
    START => Initialize => SimpleAssistantAgent => END
    
See the restaurant_recommender example for a more complex graph.
"""
import os
from dotenv import load_dotenv
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field

load_dotenv()

import logging
logging.basicConfig(encoding="utf-8", level=20)
logger = logging.getLogger(__name__)

### AGENT INPUT/OUTPUT DEFINITIONS ###
### Those are used to get a structured input/output from the LLM model
### They can also be stored in the state and passed to other agents when invoking them
class AssistantRequest(BaseModel):
    """A request for the simple assistant."""
    question: str = Field(description="The user's question.")


class AssistantResponse(BaseModel):
    """A response from the simple assistant."""
    answer: str = Field(description="The answer to the user's question.")
    score: Optional[int] = Field(
        default=None, 
        description="From 0 to 10 how confident are you in your answer?"
    )

from entourage_utils.chat_agent import ChatAgent
from entourage_utils.agent import AgentConfig
from langgraph.store.base import BaseStore

### AGENT DEFINITION ###
### The ChatAgent class is a wrapper around the LLM model, with support for chat history (not needed in our exemple)
### The prompts are embedded here but see the restaurant_recommender example for a proper implementation.
class SimpleAssistantAgent(ChatAgent):
    """A simple assistant that answers questions directly without external tools."""

    def __init__(self):
        super().__init__(
            config=AgentConfig(
                # provider="ollama", model="PetrosStav/gemma3-tools:12b",
                provider="google_genai", model="gemini-2.0-flash",
                # provider="openai", model="gpt-4o-mini",
                temperature=0.7,
                prompt_templates={
### The system prompt, the input schema and output schema are passed in this init method
                    "system":  '''You are a helpful assistant that answers questions directly without external tools.
# Input format

You are given information about the current State in the following schema:
$input_schema

# Output format

The output format should have the following schema:
$output_schema''',
### The user prompt, the input is passed when invoked in the answer_question method
"user": '''$input'''
                },
            ),
            input_schema=AssistantRequest,
            output_schema=AssistantResponse,
        )
    
    ## Invokes the LLM model with the user's question and returns the answer directly
    def answer_question(self, request: AssistantRequest, store: BaseStore) -> AssistantResponse:
        return self.invoke_with_chat_history(
            input=request.question,
            store=store,
            prompt_tag="user",
        ) 


### STATE DEFINITION, MUTABLE THROUGHOUT THE GRAPH EXECUTION ###
class State(BaseModel):
    """A state of the system, the class is mutable throughout the graph execution."""
    request: AssistantRequest
    response: Optional[AssistantResponse] = None
    

## Simple graph configuration, no customization
class GraphConfig(BaseModel):
    """The configuration for the main application."""

    ## In this simple exemple, we only have one agent.
    ## Names are used to identify the agents in the graph.
    ## They are assigned to the agents in the graph definition.
    agent_names: List[str] = [
        "simple_assistant",
    ]

    
### GRAPH DEFINITIONS, DEFINES THE EXECUTION FLOW ###
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.base import RunnableLike
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import uuid
from functools import partial

## Initialize the state. Will be called everytime the graph is executed.
## Not doing anything in our case but could be used to maintain the progress of the graph if needed.
def initialize(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> State:
    return state

## Declare the simple assistant node. Tells the graph to execute the simple assistant agent.
def simple_assistant(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
    agent: SimpleAssistantAgent,
) -> State:
    
    output: AssistantResponse = agent.answer_question(
        request=state.request, store=store
    )
    logger.info(f"Output: {output}")
    ## Stores the response in the state for later use
    state.response = output
    return state

def get_graph(config: GraphConfig) -> CompiledGraph:
    ## Our graph takes a State object that will be used to store the state during the whole graph flow
    graph = StateGraph(state_schema=State)

    ## Unique identifier for the graph. Just for logging purposes
    graph_id = uuid.uuid4()
    # Agents
    
    simple_assistant_agent = SimpleAssistantAgent()

    # Nodes
    class Node:
        def print_log_when_executing(self, func, name):
            def wrapper(state: State, config: RunnableConfig, store: BaseStore):
                logger.info(f">> Executing graph [#{graph_id}] node: {name}")
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

    ## The nodes of the graph. In this case, it is a direct flow, no conditional edges or parallel nodes...
    nodes = [
        Node(initialize),
        Node(
            simple_assistant,
            agent=simple_assistant_agent,
        )
    ]
    for node in nodes:
        graph.add_node(node.name, node.node)

    # Edges
    ## The edges of the graph. They define the flow of the graph.
    ## They use the names of the nodes to connect them.
    graph.add_edge(START, initialize.__name__)
    graph.add_edge(initialize.__name__, simple_assistant.__name__)
    ## For more complex graphs, see the restaurant_recommender example.


    # Graph compilation
    checkpointer = MemorySaver()
    in_memory_store = InMemoryStore()

    ## The checkpoint and memory store are needed for allowing the graph to be interrupted and resumed.
    graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
    graph.name = "Simple Assistant Agent Template"
    return graph


from entourage_utils.graph_streaming import StreamGraphUpdates

### MAIN ###    
def main():
    logger.info("Simple Assistant Demo\n")
    
    # Example questions to ask
    request = AssistantRequest(
        question="What's the difference between classical and quantum computing?"
    )
    initial_state = State(request=request)
    events = StreamGraphUpdates(
        graph=get_graph(GraphConfig()),
        run_config={
            "configurable": {"thread_id": 102},
            "recursion_limit": 500,
        },
    )(initial_state=initial_state)
    
    print("-" * 80)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main() 