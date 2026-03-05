from __future__ import annotations

import inspect
from collections.abc import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from neo4j_factual_chatbot.schemas.chat import (
    ActionPlan,
    ChatbotState,
    FactTriple,
    OperationResult,
)


def _json_symbol_path(symbol: type) -> tuple[str, ...]:
    return (*symbol.__module__.split("."), symbol.__name__)


def _build_checkpoint_serializer() -> JsonPlusSerializer:
    symbols = [ActionPlan, FactTriple, OperationResult]
    params = inspect.signature(JsonPlusSerializer).parameters

    if "allowed_msgpack_modules" in params:
        return JsonPlusSerializer(
            allowed_msgpack_modules=symbols,
            allowed_json_modules=[_json_symbol_path(symbol) for symbol in symbols],
        )

    return JsonPlusSerializer(
        allowed_json_modules=[_json_symbol_path(symbol) for symbol in symbols]
    )


def build_chatbot_graph(
    *,
    classify_handler: Callable[[ChatbotState], dict[str, object]],
    add_handler: Callable[[ChatbotState], dict[str, object]],
    inquire_handler: Callable[[ChatbotState], dict[str, object]],
    update_handler: Callable[[ChatbotState], dict[str, object]],
    delete_handler: Callable[[ChatbotState], dict[str, object]],
    respond_handler: Callable[[ChatbotState], dict[str, object]],
    route_handler: Callable[[ChatbotState], str],
):
    serializer = _build_checkpoint_serializer()
    graph = StateGraph(ChatbotState)
    graph.add_node("classify_intent", classify_handler)
    graph.add_node("add_fact", add_handler)
    graph.add_node("inquire_fact", inquire_handler)
    graph.add_node("update_fact", update_handler)
    graph.add_node("delete_fact", delete_handler)
    graph.add_node("respond", respond_handler)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_handler,
        {
            "add_fact": "add_fact",
            "inquire_fact": "inquire_fact",
            "update_fact": "update_fact",
            "delete_fact": "delete_fact",
            "respond": "respond",
        },
    )
    graph.add_edge("add_fact", "respond")
    graph.add_edge("inquire_fact", "respond")
    graph.add_edge("update_fact", "respond")
    graph.add_edge("delete_fact", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver(serde=serializer))
