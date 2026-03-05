from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from neo4j_factual_chatbot.schemas.chat import FactTriple, OperationResult


def canonicalize(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split()).lower()
    return normalized or None


class Neo4jFactRepository:
    def __init__(self, uri: str, username: str, password: str, database: str) -> None:
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def verify_connection(self) -> None:
        self.driver.verify_connectivity()
        with self.driver.session(database=self.database) as session:
            record = session.run("RETURN 1 AS ok").single()
            if not record or record["ok"] != 1:
                raise ValueError("Neo4j connectivity check failed")

    def add_fact(self, fact: FactTriple) -> OperationResult:
        if not fact.is_complete():
            return self._error_result("Fact must include subject, relation, and object.")

        payload = self._fact_payload(fact)
        now = datetime.now(timezone.utc)

        try:
            with self.driver.session(database=self.database) as session:
                stored = session.execute_write(self._add_fact_tx, payload, now)
        except Neo4jError as exc:
            return self._error_result(f"Database error while adding fact: {exc}")

        return OperationResult(
            status="ok",
            message="Fact stored successfully.",
            affected_count=1,
            facts=[stored],
        )

    def find_facts(self, fact: FactTriple, limit: int) -> OperationResult:
        filters = {
            "subject_canonical": canonicalize(fact.subject),
            "relation_canonical": canonicalize(fact.relation),
            "object_canonical": canonicalize(fact.object),
            "limit": limit,
        }

        try:
            with self.driver.session(database=self.database) as session:
                facts = session.execute_read(self._find_facts_tx, filters)
        except Neo4jError as exc:
            return self._error_result(f"Database error while searching facts: {exc}")

        if not facts:
            return OperationResult(
                status="not_found",
                message="No matching facts were found.",
                affected_count=0,
                facts=[],
            )

        return OperationResult(
            status="ok",
            message="Matching facts retrieved successfully.",
            affected_count=len(facts),
            facts=facts,
        )

    def update_fact(
        self,
        target: FactTriple,
        replacement: FactTriple,
        *,
        search_limit: int,
        ambiguous_limit: int,
    ) -> OperationResult:
        if not target.has_any_value():
            return self._error_result("Target fact details are required for update.")
        if not replacement.is_complete():
            return self._error_result("Replacement fact must include subject, relation, and object.")

        candidates = self.find_facts(target, ambiguous_limit)
        if candidates.status in {"not_found", "error"}:
            return candidates
        if len(candidates.facts) != 1:
            return OperationResult(
                status="ambiguous",
                message="More than one fact matches the requested update.",
                affected_count=min(len(candidates.facts), ambiguous_limit),
                facts=candidates.facts[:ambiguous_limit],
            )

        payload = self._fact_payload(replacement)
        now = datetime.now(timezone.utc)

        try:
            with self.driver.session(database=self.database) as session:
                updated = session.execute_write(
                    self._replace_fact_tx,
                    candidates.facts[0],
                    payload,
                    now,
                )
        except (Neo4jError, ValueError) as exc:
            return self._error_result(f"Database error while updating fact: {exc}")

        return OperationResult(
            status="ok",
            message="Fact updated successfully.",
            affected_count=1,
            facts=[updated],
        )

    def delete_fact(self, fact: FactTriple, ambiguous_limit: int) -> OperationResult:
        if not fact.has_any_value():
            return self._error_result("Fact details are required for deletion.")

        candidates = self.find_facts(fact, ambiguous_limit)
        if candidates.status in {"not_found", "error"}:
            return candidates
        if len(candidates.facts) != 1:
            return OperationResult(
                status="ambiguous",
                message="More than one fact matches the requested deletion.",
                affected_count=min(len(candidates.facts), ambiguous_limit),
                facts=candidates.facts[:ambiguous_limit],
            )

        try:
            with self.driver.session(database=self.database) as session:
                deleted_count = session.execute_write(self._delete_fact_tx, candidates.facts[0])
        except Neo4jError as exc:
            return self._error_result(f"Database error while deleting fact: {exc}")

        if deleted_count != 1:
            return OperationResult(
                status="not_found",
                message="The fact no longer exists.",
                affected_count=0,
                facts=[],
            )

        return OperationResult(
            status="ok",
            message="Fact deleted successfully.",
            affected_count=1,
            facts=[candidates.facts[0]],
        )

    @staticmethod
    def _error_result(message: str) -> OperationResult:
        return OperationResult(status="error", message=message)

    @staticmethod
    def _fact_payload(fact: FactTriple) -> dict[str, str]:
        return {
            "subject_name": fact.subject or "",
            "subject_canonical": canonicalize(fact.subject) or "",
            "relation_name": fact.relation or "",
            "relation_canonical": canonicalize(fact.relation) or "",
            "object_name": fact.object or "",
            "object_canonical": canonicalize(fact.object) or "",
            "fact_text": f"{fact.subject} {fact.relation} {fact.object}",
        }

    @staticmethod
    def _normalize_string(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    @classmethod
    def _normalize_fact_record(cls, record: dict[str, Any]) -> dict[str, str]:
        subject = cls._normalize_string(record.get("subject") or record.get("subject_canonical"))
        relation = cls._normalize_string(record.get("relation") or record.get("relation_canonical"))
        object_name = cls._normalize_string(record.get("object") or record.get("object_canonical"))

        subject_canonical = cls._normalize_string(
            record.get("subject_canonical") or canonicalize(subject) or ""
        )
        relation_canonical = cls._normalize_string(
            record.get("relation_canonical") or canonicalize(relation) or ""
        )
        object_canonical = cls._normalize_string(
            record.get("object_canonical") or canonicalize(object_name) or ""
        )

        fact_text = cls._normalize_string(record.get("fact_text"))
        if not fact_text:
            fact_text = " ".join(part for part in (subject, relation, object_name) if part)

        return {
            "subject": subject,
            "subject_canonical": subject_canonical,
            "relation": relation,
            "relation_canonical": relation_canonical,
            "object": object_name,
            "object_canonical": object_canonical,
            "fact_text": fact_text,
        }

    @staticmethod
    def _add_fact_tx(tx, payload: dict[str, str], now: datetime) -> dict[str, str]:
        query = """
        MERGE (subject:Entity {canonical_name: $subject_canonical})
          ON CREATE SET subject.created_at = $now
        SET subject.name = $subject_name, subject.updated_at = $now
        MERGE (object:Entity {canonical_name: $object_canonical})
          ON CREATE SET object.created_at = $now
        SET object.name = $object_name, object.updated_at = $now
        MERGE (subject)-[rel:FACT {canonical_relation: $relation_canonical}]->(object)
          ON CREATE SET rel.created_at = $now
        SET rel.relation = $relation_name,
            rel.fact_text = $fact_text,
            rel.updated_at = $now
        RETURN subject.name AS subject,
               subject.canonical_name AS subject_canonical,
               rel.relation AS relation,
               rel.canonical_relation AS relation_canonical,
               object.name AS object,
               object.canonical_name AS object_canonical,
               rel.fact_text AS fact_text
        """
        record = tx.run(query, **payload, now=now).single()
        if not record:
            raise ValueError("Failed to store fact")
        return Neo4jFactRepository._normalize_fact_record(dict(record))

    @staticmethod
    def _find_facts_tx(tx, filters: dict[str, str | int | None]) -> list[dict[str, str]]:
        query = """
        MATCH (subject:Entity)-[rel:FACT]->(object:Entity)
        WHERE ($subject_canonical IS NULL OR subject.canonical_name = $subject_canonical)
          AND ($relation_canonical IS NULL OR rel.canonical_relation = $relation_canonical)
          AND ($object_canonical IS NULL OR object.canonical_name = $object_canonical)
        RETURN subject.name AS subject,
               subject.canonical_name AS subject_canonical,
               rel.relation AS relation,
               rel.canonical_relation AS relation_canonical,
               object.name AS object,
               object.canonical_name AS object_canonical,
               rel.fact_text AS fact_text
        ORDER BY subject.name, rel.relation, object.name
        LIMIT $limit
        """
        return [Neo4jFactRepository._normalize_fact_record(dict(record)) for record in tx.run(query, **filters)]

    @staticmethod
    def _replace_fact_tx(
        tx,
        current_fact: dict[str, str],
        replacement_payload: dict[str, str],
        now: datetime,
    ) -> dict[str, str]:
        delete_query = """
        MATCH (subject:Entity {canonical_name: $subject_canonical})
              -[rel:FACT {canonical_relation: $relation_canonical}]->
              (object:Entity {canonical_name: $object_canonical})
        DELETE rel
        """
        result = tx.run(
            delete_query,
            subject_canonical=current_fact["subject_canonical"],
            relation_canonical=current_fact["relation_canonical"],
            object_canonical=current_fact["object_canonical"],
        )
        if result.consume().counters.relationships_deleted != 1:
            raise ValueError("The fact to update was not found")

        return Neo4jFactRepository._add_fact_tx(tx, replacement_payload, now)

    @staticmethod
    def _delete_fact_tx(tx, fact: dict[str, str]) -> int:
        query = """
        MATCH (subject:Entity {canonical_name: $subject_canonical})
              -[rel:FACT {canonical_relation: $relation_canonical}]->
              (object:Entity {canonical_name: $object_canonical})
        DELETE rel
        """
        result = tx.run(
            query,
            subject_canonical=fact["subject_canonical"],
            relation_canonical=fact["relation_canonical"],
            object_canonical=fact["object_canonical"],
        )
        return result.consume().counters.relationships_deleted
