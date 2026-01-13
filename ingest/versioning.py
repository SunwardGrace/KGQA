import logging
import re
from datetime import datetime
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


class GraphVersioning:
    def __init__(self, neo4j_client):
        self.client = neo4j_client
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            self.client.run_query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ChangeSet) REQUIRE c.change_id IS UNIQUE", {}
            )
        except Exception as e:
            logger.warning(f"ChangeSet constraint may already exist: {e}")

    def begin_changeset(self, actor: str, reason: str, source: str = "pipeline") -> str:
        change_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        cypher = """
            CREATE (c:ChangeSet {
                change_id: $change_id,
                actor: $actor,
                reason: $reason,
                source: $source,
                created_at: $now,
                status: 'active'
            })
            RETURN c.change_id as id
        """
        self.client.run_query(cypher, {
            "change_id": change_id,
            "actor": actor,
            "reason": reason,
            "source": source,
            "now": now,
        })
        logger.info(f"Created changeset: {change_id}")
        return change_id

    def upsert_triple(self, triple: dict, change_id: str) -> None:
        head = triple.get("head", "")
        tail = triple.get("tail", "")
        relation = triple.get("relation", "RELATED_TO")
        if not head or not tail:
            return

        if not re.match(r"^[A-Z0-9_]+$", relation):
            logger.warning(f"Skipping triple with invalid relation type: {relation}")
            return

        now = datetime.now().isoformat()
        revision = self._get_next_revision(head, relation, tail)

        cypher = f"""
            MATCH (c:ChangeSet {{change_id: $change_id}})
            MERGE (h {{name: $head}})
            MERGE (t {{name: $tail}})
            CREATE (h)-[r:{relation} {{
                revision: $revision,
                active: true,
                valid_from: $now,
                change_id: $change_id,
                confidence: $confidence,
                source: $source
            }}]->(t)
            CREATE (c)-[:INCLUDES_CHANGE]->(:AuditEvent {{
                timestamp: $now,
                action: 'CREATE',
                head: $head,
                relation: $relation_str,
                tail: $tail,
                revision: $revision
            }})
        """
        self.client.run_query(cypher, {
            "change_id": change_id,
            "head": head,
            "tail": tail,
            "relation_str": relation,
            "revision": revision,
            "confidence": triple.get("confidence", 1.0),
            "source": triple.get("source", "unknown"),
            "now": now,
        })

    def _get_next_revision(self, head: str, relation: str, tail: str) -> int:
        cypher = f"""
            MATCH (h {{name: $head}})-[r:{relation}]->(t {{name: $tail}})
            RETURN max(r.revision) as max_rev
        """
        results = self.client.run_query(cypher, {"head": head, "tail": tail})
        if results and results[0].get("max_rev") is not None:
            return results[0]["max_rev"] + 1
        return 1

    def rollback(self, change_id: str) -> int:
        now = datetime.now().isoformat()
        cypher = """
            MATCH (c:ChangeSet {change_id: $change_id})
            SET c.status = 'rolled_back', c.rolled_back_at = $now
            WITH c
            MATCH (c)-[:INCLUDES_CHANGE]->(a:AuditEvent)
            SET a.rolled_back = true
            RETURN count(a) as count
        """
        results = self.client.run_query(cypher, {"change_id": change_id, "now": now})
        count = results[0]["count"] if results else 0

        deactivate_cypher = """
            MATCH ()-[r {change_id: $change_id}]->()
            SET r.active = false, r.valid_to = $now
            RETURN count(r) as edges
        """
        self.client.run_query(deactivate_cypher, {"change_id": change_id, "now": now})

        logger.info(f"Rolled back changeset {change_id}: {count} events")
        return count

    def list_changesets(self, limit: int = 50) -> list[dict]:
        cypher = """
            MATCH (c:ChangeSet)
            OPTIONAL MATCH (c)-[:INCLUDES_CHANGE]->(a:AuditEvent)
            WITH c, count(a) as event_count
            RETURN c.change_id as change_id, c.actor as actor, c.reason as reason,
                   c.source as source, c.created_at as created_at, c.status as status,
                   event_count
            ORDER BY c.created_at DESC
            LIMIT $limit
        """
        return self.client.run_query(cypher, {"limit": limit})

    def get_audit_events(self, change_id: str) -> list[dict]:
        cypher = """
            MATCH (c:ChangeSet {change_id: $change_id})-[:INCLUDES_CHANGE]->(a:AuditEvent)
            RETURN a.timestamp as timestamp, a.action as action,
                   a.head as head, a.relation as relation, a.tail as tail,
                   a.revision as revision, a.rolled_back as rolled_back
            ORDER BY a.timestamp
        """
        return self.client.run_query(cypher, {"change_id": change_id})

    def get_entity_history(self, entity_name: str) -> list[dict]:
        cypher = """
            MATCH (n {name: $name})
            OPTIONAL MATCH (n)-[r]->()
            WHERE r.change_id IS NOT NULL
            RETURN r.change_id as change_id, type(r) as relation,
                   r.revision as revision, r.valid_from as valid_from,
                   r.valid_to as valid_to, r.active as active
            ORDER BY r.valid_from DESC
        """
        return self.client.run_query(cypher, {"name": entity_name})
