from dotenv import load_dotenv
import logging

    
logging.basicConfig(encoding="utf-8", level=20)
logging.getLogger(__name__).info("CIAO")

load_dotenv()


from restaurant_agent.graph import GraphConfig, get_graph
from restaurant_agent.config import GraphConfig
from entourage_poc import json_utils
from restaurant_agent import config
run_config = config.DEFAULT_RUN_CONFIG

def main():
    graph_config: GraphConfig = GraphConfig()
    print(json_utils.dumps(graph_config.model_dump(), indent=2))
    
    graph = get_graph(graph_config)

    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))

    from restaurant_agent.state import Request, State
    from entourage_poc.graph_streaming import StreamGraphUpdates

    request = Request(
        cuisine="asian",
        location="Montbonnot-St-Martin",
        max_price="120$",
    )
    initial_state = State(request=request)

    events = StreamGraphUpdates(
        graph=get_graph(graph_config),
        run_config=run_config,
    )(initial_state=initial_state)

    # Print details about the request
    print("\nRequest Details:")
    print(f"Cuisine: {request.cuisine}")
    print(f"Location: {request.location}")
    print(f"Max Price: {request.max_price}")
    print(f"Min Rating: {request.min_rating}")
    print(f"Preferences: {request.preferences}")
    

if __name__ == "__main__":
    main()
