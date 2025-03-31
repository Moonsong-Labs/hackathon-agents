# Install

```
uv pip install -r pyproject.toml
```

# Template 1: Run Simple Agent

This template contains all in 1 file to easily understand each component, directly described inline.

```
uv run templates/1-simple_assistant/simple_assistant_example.py
```

# Template 2: Run Restaurant Recommander Agent

This template is more complex and utilizes the potential of the graph to make multiple agents interact with each other.

## The graph

As a traditional **graph**, it combines **nodes** ("actions", defined as a function usually calling an agent) using **edges** (a directive pointing to the next node to execute). The whole flow of execution maintains a **state** with variables that can be re-used by nodes/edges.

## The state

The [state.py](./templates/2-restaurant_recommander/state.py) contains the structure used by the nodes which are maintained in the final **State** class (provided to the graph). That class gets passed to nodes/edges

## 2 agents

* [reflection_agent](./templates/2-restaurant_recommander/agents/reflection_agent.py): Main agent that performs the "thinking part" of the scenario.
* [restaurant_recommender_agent](./templates/2-restaurant_recommander/agents/restaurant_recommender_agent.py): Restaurant Recommender agent that performs a web-search (using tavily search. Requires `TAVILY_API_KEY`) to get information about restaurant.

## 4 nodes

The nodes are defined in the [graph.py](./templates/2-restaurant_recommander/graph.py) as functions:

* *initialize*: Used usually to configure the whole workflow
* *recommend_restaurant*: The node that calls the restaurant recommender agent
* *reflect*: The node that calls the reflection agent for more thinking

* *decide_next_step*: This one is used to provide conditional control over the steps (specially useful to end or limit the flow)

## 1 custom edge

Sequential edges are defined directly within the add_edge calls but for more complex edges, we can also provide functions to specify which node to call.

* *route_by_status*: will call the restaurant_recommender if the reflection agent hasn't finished (basically if it needs more data from the recommender)

## Building the whole graph

* *get_graph*: Defines the list of nodes and their edges. (It also wrap the nodes with a `print_log_when_executing` for providing more logs). The order of the Node list is not relevant. The flow of execution is defined by the add_edge, which contains "START" to define the first step.


## Executing it
```
uv run templates/2-restaurant_recommander/restaurant_recommander_example.py
```

# Repository History (only for Alan)
```
uv add git+https://github.com/Moonsong-Labs/entourage.git@hackathon-template#subdirectory=entourage-utils
```
