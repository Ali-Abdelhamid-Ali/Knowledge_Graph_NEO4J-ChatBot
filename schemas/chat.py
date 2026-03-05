from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FactTriple(BaseModel):
    subject: str | None = None
    relation: str | None = None
    object: str | None = None

    @field_validator("subject", "relation", "object", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).strip().split())
        return text or None

    def has_any_value(self) -> bool:
        return any((self.subject, self.relation, self.object))

    def is_complete(self) -> bool:
        return all((self.subject, self.relation, self.object))


class ActionPlan(BaseModel):
    intent: Literal["add", "inquire", "update", "delete", "clarify"]
    fact: FactTriple = Field(default_factory=FactTriple)
    target_fact: FactTriple = Field(default_factory=FactTriple)
    replacement_fact: FactTriple = Field(default_factory=FactTriple)
    clarification_question: str = ""

    @field_validator("clarification_question", mode="before")
    @classmethod
    def clean_question(cls, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())


class OperationResult(BaseModel):
    status: Literal["ok", "not_found", "ambiguous", "error"]
    message: str
    affected_count: int = 0
    facts: list[dict[str, str]] = Field(default_factory=list)
    token_usage_prompt: int = 0
    token_usage_completion: int = 0
    token_usage_total: int = 0


class ChatbotState(TypedDict, total=False):
    user_input: str
    history: list[dict[str, str]]
    planned_action: ActionPlan | None
    db_result: OperationResult | None
    assistant_response: str
    error: str | None
