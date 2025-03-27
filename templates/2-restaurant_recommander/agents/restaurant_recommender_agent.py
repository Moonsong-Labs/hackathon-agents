from dataclasses import field
import logging
from typing import Dict, List, Optional, Union, override

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from restaurant_agent.state import Request, Restaurant

from entourage_poc.agents.chat_agent import ChatAgent, LLMOutputValidationError
from entourage_poc import json_utils

from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


class WebResults(BaseModel):
    web_results: List[Dict]


class RestaurantRecommenderInput(BaseModel):
    input: Union[Request, WebResults]


class WebSearchQuery(BaseModel):
    query: str = Field(description="The search query string.")
    max_results: int = Field(
        description="The maximum number of search results to return.",
    )


class RestaurantRecommendation(BaseModel):
    restaurant: Optional[Restaurant]

    justification: Optional[str] = Field(
        description="The justification for the recommendation."
    )

    error: Optional[str] = Field(description="The error message if any.")


class RestaurantRecommenderOutput(BaseModel):
    output: Union[RestaurantRecommendation, WebSearchQuery]


class RestaurantRecommenderAgent(ChatAgent):
    """An agent that recommend restaurants based on user preferences."""

    def __init__(self, config):
        super().__init__(
            config=config,
            input_schema=RestaurantRecommenderInput,
            output_schema=RestaurantRecommenderOutput,
        )

        self.tavily_search_tool = TavilySearchResults()

    @override
    def validate_output(
        self, input: RestaurantRecommenderInput, output: RestaurantRecommenderOutput
    ):
        super().validate_output(input, output)

        input = input.input

        output = output.output

        logger

        if isinstance(input, WebResults):
            if isinstance(output, WebSearchQuery):
                raise LLMOutputValidationError(
                    f"Expected RestaurantRecommendation output, but got WebSearchQuery output: {
                        output
                    }"
                )

        if isinstance(output, WebSearchQuery):
            if len(output.query) == 0:
                raise LLMOutputValidationError("Empty query.")

            if output.max_results <= 0 or output.max_results > 10:
                raise LLMOutputValidationError("Invalid max_results value.")
        elif isinstance(output, RestaurantRecommendation):
            if output.error:
                if output.restaurant is not None or output.justification:
                    raise LLMOutputValidationError(
                        "Expected error message only, but got restaurant recommendation or justification."
                    )
            else:
                if output.restaurant is None or not output.justification:
                    raise LLMOutputValidationError(
                        "Expected both restaurant recommendation and justification, but got no error message."
                    )

    def recommend_restaurant(
        self, request: Request, store: BaseStore
    ) -> RestaurantRecommendation:
        input = RestaurantRecommenderInput(input=request)
        output = self.invoke_with_chat_history(
            input=input,
            store=store,
            prompt_tag="recommend_restaurant",
        ).output

        logger.info(f"Output: {output}")

        if isinstance(output, RestaurantRecommendation):
            return output
        else:
            output: WebSearchQuery
            response: dict = self.tavily_search_tool.api_wrapper.raw_results(
                query=output.query,
                max_results=output.max_results,
                include_raw_content=True,
                search_depth="basic",
            )

            web_results: List[Dict] = response.get("results")

            logger.info(f"Web Results: {json_utils.dumps(web_results)}")

            if web_results:
                return self.recommend_restaurant_from_web_results(
                    web_results=WebResults(web_results=web_results),
                    store=store,
                )
            else:
                return RestaurantRecommendation(
                    restaurant=None,
                    justification=None,
                    error=f"No restaurant found with the web search: {output.query}.",
                )

    def recommend_restaurant_from_web_results(
        self, web_results: WebResults, store: BaseStore
    ) -> RestaurantRecommendation:
        input = RestaurantRecommenderInput(input=web_results)
        output: RestaurantRecommendation = self.invoke_with_chat_history(
            input=input, store=store, prompt_tag="recommend_restaurant_from_web_results"
        ).output

        logger.info(f"Output: {output}")

        return output
