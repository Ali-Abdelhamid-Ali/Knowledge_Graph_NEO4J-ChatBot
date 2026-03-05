from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an intent planner for a Neo4j factual chatbot. "
                "Classify the latest user input into exactly one intent: add, inquire, update, delete, or clarify. "
                "Use the recent conversation history to resolve follow-up statements. "
                "Never output Cypher. Never guess missing details for destructive operations. "
                "Return strict JSON only with this exact shape: "
                '{{"intent":"add|inquire|update|delete|clarify",'
                '"fact":{{"subject":null,"relation":null,"object":null}},'
                '"target_fact":{{"subject":null,"relation":null,"object":null}},'
                '"replacement_fact":{{"subject":null,"relation":null,"object":null}},'
                '"clarification_question":""}}. '
                "Rules: add uses fact only; inquire uses fact as search filters; "
                "update uses target_fact and replacement_fact; delete uses fact only; "
                "clarify must include a short question in the same language as the user when possible. "
                'Example add: {{"intent":"add","fact":{{"subject":"Ali","relation":"lives in","object":"Cairo"}},"target_fact":{{"subject":null,"relation":null,"object":null}},"replacement_fact":{{"subject":null,"relation":null,"object":null}},"clarification_question":""}}. '
                'Example inquire: {{"intent":"inquire","fact":{{"subject":"Ali","relation":"lives in","object":null}},"target_fact":{{"subject":null,"relation":null,"object":null}},"replacement_fact":{{"subject":null,"relation":null,"object":null}},"clarification_question":""}}. '
                'Example update: {{"intent":"update","fact":{{"subject":null,"relation":null,"object":null}},"target_fact":{{"subject":"Ali","relation":"lives in","object":"Cairo"}},"replacement_fact":{{"subject":"Ali","relation":"lives in","object":"Giza"}},"clarification_question":""}}. '
                'Example clarify: {{"intent":"clarify","fact":{{"subject":null,"relation":null,"object":null}},"target_fact":{{"subject":null,"relation":null,"object":null}},"replacement_fact":{{"subject":null,"relation":null,"object":null}},"clarification_question":"Which fact about Ali should I delete?"}}.'
            ),
        ),
        (
            "user",
            (
                "Recent conversation history (JSON): {history_json}\n"
                "Latest user input: {user_input}"
            ),
        ),
    ]
)


RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a response synthesizer for a Neo4j factual chatbot. "
                "Write a concise natural-language answer grounded only in the provided database operation result. "
                "Do not invent facts. Do not mention Cypher, JSON, prompts, or internal implementation. "
                "If the result is ambiguous, ask the user to choose from the listed facts. "
                "If nothing was found, state that clearly. "
                "Answer in the same language as the user's message when possible. "
                "Return plain text only."
            ),
        ),
        (
            "user",
            (
                "User message: {user_input}\n"
                "Intent: {intent}\n"
                "Operation result (JSON): {result_json}"
            ),
        ),
    ]
)
