from __future__ import annotations

import json
import logging
import re
from uuid import uuid4

from langchain_cohere import ChatCohere
from pydantic import ValidationError

from neo4j_factual_chatbot.core.config import Settings
from neo4j_factual_chatbot.graph.repository import Neo4jFactRepository
from neo4j_factual_chatbot.llm.parsers import (
    get_response_text,
    get_token_usage,
    parse_first_json,
)
from neo4j_factual_chatbot.prompts.chatbot import (
    CLASSIFICATION_PROMPT,
    RESPONSE_PROMPT,
)
from neo4j_factual_chatbot.schemas.chat import (
    ActionPlan,
    ChatbotState,
    OperationResult,
)
from neo4j_factual_chatbot.workflow.langgraph_flow import build_chatbot_graph

logger = logging.getLogger(__name__)
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
MAX_HISTORY_MESSAGES = 12
MAX_STORED_HISTORY = 24


class Neo4jFactualChatbotService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = Neo4jFactRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        self.llm = ChatCohere(
            model=settings.model_name,
            temperature=0.0,
            cohere_api_key=settings.cohere_api_key,
        )
        self.session_id = uuid4().hex
        self.graph_config = {"configurable": {"thread_id": self.session_id}}
        self.graph_app = build_chatbot_graph(
            classify_handler=self._classify_intent_node,
            add_handler=self._add_fact_node,
            inquire_handler=self._inquire_fact_node,
            update_handler=self._update_fact_node,
            delete_handler=self._delete_fact_node,
            respond_handler=self._respond_node,
            route_handler=self._route_after_classify,
        )

    def verify_connection(self) -> None:
        self.repository.verify_connection()

    def close(self) -> None:
        self.repository.close()

    def handle_message(self, user_input: str) -> str:
        result = self.graph_app.invoke({"user_input": user_input}, config=self.graph_config)
        answer = str(result.get("assistant_response", "")).strip()
        return answer or self._default_error_message(user_input)

    @staticmethod
    def is_arabic(text: str | None) -> bool:
        return bool(text and ARABIC_RE.search(text))

    @staticmethod
    def _recent_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        return list((history or [])[-MAX_HISTORY_MESSAGES:])

    @staticmethod
    def _public_fact_view(fact: dict[str, str]) -> dict[str, str]:
        return {
            "subject": fact.get("subject", ""),
            "relation": fact.get("relation", ""),
            "object": fact.get("object", ""),
            "fact_text": fact.get("fact_text", ""),
        }

    def _public_result_view(self, result: OperationResult | None) -> dict[str, object]:
        if result is None:
            return {}
        payload = result.model_dump()
        payload["facts"] = [self._public_fact_view(fact) for fact in result.facts]
        return payload

    @staticmethod
    def _format_fact(fact: dict[str, str]) -> str:
        fact_text = fact.get("fact_text")
        if fact_text:
            return fact_text
        return " ".join(
            part for part in (fact.get("subject"), fact.get("relation"), fact.get("object")) if part
        ).strip()

    def _default_error_message(self, user_input: str) -> str:
        if self.is_arabic(user_input):
            return "حدث خطأ داخلي أثناء تنفيذ الطلب. حاول مرة أخرى بصياغة أوضح."
        return "An internal error happened while handling your request. Please try again with clearer wording."

    def _clarify_plan(self, user_input: str, question: str | None = None) -> ActionPlan:
        if question:
            clarification_question = question
        elif self.is_arabic(user_input):
            clarification_question = (
                "من فضلك وضح الطلب بشكل أدق حتى أتمكن من تنفيذ الإضافة أو الاستعلام أو التعديل أو الحذف."
            )
        else:
            clarification_question = (
                "Please clarify your request so I can safely add, inquire, update, or delete a fact."
            )
        return ActionPlan(intent="clarify", clarification_question=clarification_question)

    def _fallback_response(
        self,
        user_input: str,
        plan: ActionPlan,
        result: OperationResult | None,
    ) -> str:
        if plan.intent == "clarify":
            return plan.clarification_question or self._clarify_plan(user_input).clarification_question

        if result is None:
            return self._default_error_message(user_input)

        arabic = self.is_arabic(user_input)
        facts = [self._format_fact(fact) for fact in result.facts if self._format_fact(fact)]

        if result.status == "not_found":
            return "لم أجد أي حقيقة مطابقة." if arabic else "I could not find a matching fact."

        if result.status == "ambiguous":
            if facts:
                joined = "; ".join(facts)
                if arabic:
                    return f"وجدت أكثر من حقيقة مطابقة: {joined}. من فضلك حدد أي واحدة تقصد."
                return f"I found multiple matching facts: {joined}. Please specify which one you mean."
            if arabic:
                return "وجدت أكثر من نتيجة مطابقة. من فضلك حدد الحقيقة المقصودة."
            return "I found multiple matching facts. Please specify the intended one."

        if result.status == "error":
            return result.message or self._default_error_message(user_input)

        if plan.intent == "add" and facts:
            return f"تم حفظ الحقيقة: {facts[0]}." if arabic else f"I stored the fact: {facts[0]}."
        if plan.intent == "update" and facts:
            return f"تم تحديث الحقيقة إلى: {facts[0]}." if arabic else f"I updated the fact to: {facts[0]}."
        if plan.intent == "delete" and facts:
            return f"تم حذف الحقيقة: {facts[0]}." if arabic else f"I deleted the fact: {facts[0]}."
        if plan.intent == "inquire":
            if len(facts) == 1:
                return facts[0]
            if facts:
                joined = "; ".join(facts)
                return f"وجدت {len(facts)} حقائق: {joined}." if arabic else f"I found {len(facts)} facts: {joined}."

        return result.message or self._default_error_message(user_input)

    def _invoke_llm(self, messages):
        response = self.llm.invoke(messages)
        return get_response_text(response), get_token_usage(response)

    def _classify_intent(self, user_input: str, history: list[dict[str, str]]) -> ActionPlan:
        messages = CLASSIFICATION_PROMPT.format_messages(
            history_json=json.dumps(self._recent_history(history), ensure_ascii=False),
            user_input=user_input,
        )

        try:
            raw_text, token_usage = self._invoke_llm(messages)
            logger.info(
                "Classifier tokens prompt=%s completion=%s total=%s",
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                token_usage.total_tokens,
            )
            plan = ActionPlan.model_validate(parse_first_json(raw_text))
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Classifier parse failure: %s", exc)
            return self._clarify_plan(user_input)
        except Exception:
            logger.exception("Classifier LLM call failed")
            return self._clarify_plan(user_input)

        return self._enforce_safe_plan(user_input, plan)

    def _enforce_safe_plan(self, user_input: str, plan: ActionPlan) -> ActionPlan:
        if plan.intent == "add" and not plan.fact.is_complete():
            if self.is_arabic(user_input):
                return self._clarify_plan(user_input, "من فضلك اذكر الحقيقة كاملة: الموضوع والعلاقة والقيمة.")
            return self._clarify_plan(user_input, "Please provide the full fact with subject, relation, and object.")

        if plan.intent == "update":
            if not plan.target_fact.has_any_value():
                if self.is_arabic(user_input):
                    return self._clarify_plan(user_input, "من فضلك حدد الحقيقة الحالية التي تريد تعديلها.")
                return self._clarify_plan(user_input, "Please specify the existing fact that should be updated.")
            if not plan.replacement_fact.is_complete():
                if self.is_arabic(user_input):
                    return self._clarify_plan(user_input, "من فضلك اذكر الحقيقة الجديدة كاملة بعد التعديل.")
                return self._clarify_plan(user_input, "Please provide the full replacement fact for the update.")

        if plan.intent == "delete" and not plan.fact.has_any_value():
            if self.is_arabic(user_input):
                return self._clarify_plan(user_input, "من فضلك حدد الحقيقة التي تريد حذفها.")
            return self._clarify_plan(user_input, "Please specify which fact should be deleted.")

        if plan.intent == "clarify" and not plan.clarification_question:
            return self._clarify_plan(user_input)

        return plan

    def _synthesize_response(
        self,
        user_input: str,
        plan: ActionPlan,
        result: OperationResult,
    ) -> str:
        messages = RESPONSE_PROMPT.format_messages(
            user_input=user_input,
            intent=plan.intent,
            result_json=json.dumps(self._public_result_view(result), ensure_ascii=False),
        )
        raw_text, token_usage = self._invoke_llm(messages)
        logger.info(
            "Responder tokens prompt=%s completion=%s total=%s",
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            token_usage.total_tokens,
        )
        response_text = raw_text.strip()
        if not response_text:
            raise ValueError("Empty synthesis response")
        return response_text

    def _classify_intent_node(self, state: ChatbotState) -> dict[str, object]:
        user_input = (state.get("user_input") or "").strip()
        history = list(state.get("history", []))
        return {
            "planned_action": self._classify_intent(user_input, history),
            "db_result": None,
            "assistant_response": "",
            "error": None,
        }

    def _add_fact_node(self, state: ChatbotState) -> dict[str, object]:
        plan = state.get("planned_action") or self._clarify_plan(state.get("user_input", ""))
        result = self.repository.add_fact(plan.fact)
        return {"db_result": result, "error": result.message if result.status == "error" else None}

    def _inquire_fact_node(self, state: ChatbotState) -> dict[str, object]:
        plan = state.get("planned_action") or self._clarify_plan(state.get("user_input", ""))
        result = self.repository.find_facts(plan.fact, self.settings.neo4j_result_limit)
        return {"db_result": result, "error": result.message if result.status == "error" else None}

    def _update_fact_node(self, state: ChatbotState) -> dict[str, object]:
        plan = state.get("planned_action") or self._clarify_plan(state.get("user_input", ""))
        result = self.repository.update_fact(
            plan.target_fact,
            plan.replacement_fact,
            search_limit=self.settings.neo4j_result_limit,
            ambiguous_limit=self.settings.neo4j_ambiguous_limit,
        )
        return {"db_result": result, "error": result.message if result.status == "error" else None}

    def _delete_fact_node(self, state: ChatbotState) -> dict[str, object]:
        plan = state.get("planned_action") or self._clarify_plan(state.get("user_input", ""))
        result = self.repository.delete_fact(plan.fact, self.settings.neo4j_ambiguous_limit)
        return {"db_result": result, "error": result.message if result.status == "error" else None}

    def _respond_node(self, state: ChatbotState) -> dict[str, object]:
        user_input = state.get("user_input", "")
        plan = state.get("planned_action") or self._clarify_plan(user_input)
        result = state.get("db_result")

        if plan.intent == "clarify":
            assistant_response = plan.clarification_question or self._clarify_plan(user_input).clarification_question
        else:
            try:
                if result is None:
                    raise ValueError("Missing operation result")
                assistant_response = self._synthesize_response(user_input, plan, result)
            except Exception:
                logger.exception("Response synthesis failed")
                assistant_response = self._fallback_response(user_input, plan, result)

        history = list(state.get("history", []))
        history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response},
            ]
        )
        history = history[-MAX_STORED_HISTORY:]

        return {"assistant_response": assistant_response, "history": history}

    @staticmethod
    def _route_after_classify(state: ChatbotState) -> str:
        plan = state.get("planned_action")
        if plan is None:
            return "respond"
        return {
            "add": "add_fact",
            "inquire": "inquire_fact",
            "update": "update_fact",
            "delete": "delete_fact",
            "clarify": "respond",
        }.get(plan.intent, "respond")
