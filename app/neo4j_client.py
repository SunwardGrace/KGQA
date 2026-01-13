import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, cypher: str, params: dict = None) -> list[dict]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def search_entity(self, name: str, label: str = None) -> list[dict]:
        """搜索实体，优先精确匹配，其次包含匹配，按名称长度排序（优先短名称）"""
        if label:
            query = f"""
            MATCH (n:{label})
            WHERE toLower(n.name) = toLower($name)
               OR toLower(n.name) CONTAINS toLower($name)
               OR (n.aliases IS NOT NULL AND ANY(a IN n.aliases WHERE toLower(a) CONTAINS toLower($name)))
            RETURN n.name as name, labels(n) as labels, n.aliases as aliases,
                   CASE WHEN toLower(n.name) = toLower($name) THEN 0 ELSE 1 END as match_type,
                   size(n.name) as name_len
            ORDER BY match_type, name_len
            LIMIT 10
            """
        else:
            query = """
            MATCH (n)
            WHERE toLower(n.name) = toLower($name)
               OR toLower(n.name) CONTAINS toLower($name)
               OR (n.aliases IS NOT NULL AND ANY(a IN n.aliases WHERE toLower(a) CONTAINS toLower($name)))
            RETURN n.name as name, labels(n) as labels, n.aliases as aliases,
                   CASE WHEN toLower(n.name) = toLower($name) THEN 0 ELSE 1 END as match_type,
                   size(n.name) as name_len
            ORDER BY match_type, name_len
            LIMIT 10
            """
        return self.run_query(query, {"name": name})

    def get_node_by_name(self, name: str, label: str = None) -> dict | None:
        if label:
            query = f"MATCH (n:{label} {{name: $name}}) RETURN n LIMIT 1"
        else:
            query = "MATCH (n {name: $name}) RETURN n LIMIT 1"
        results = self.run_query(query, {"name": name})
        if results:
            return dict(results[0]["n"])
        return None

    def get_graph_stats(self) -> dict:
        node_count = self.run_query("MATCH (n) RETURN count(n) as c")[0]["c"]
        rel_count = self.run_query("MATCH ()-[r]->() RETURN count(r) as c")[0]["c"]
        labels = self.run_query("CALL db.labels() YIELD label RETURN collect(label) as labels")[0]["labels"]
        rel_types = self.run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")[0]["types"]
        return {
            "node_count": node_count,
            "relationship_count": rel_count,
            "labels": labels,
            "relationship_types": rel_types,
        }

    def sample_nodes(self, label: str, limit: int = 50) -> list[dict]:
        query = f"MATCH (n:{label}) RETURN n.name as name LIMIT $limit"
        return self.run_query(query, {"limit": limit})

    def sample_relationships(self, rel_type: str, limit: int = 50) -> list[dict]:
        query = f"""
        MATCH (h)-[r:{rel_type}]->(t)
        RETURN h.name as head, type(r) as relation, t.name as tail,
               r.edge_score as score, r.source as source, r.conflict as conflict
        ORDER BY r.edge_score DESC
        LIMIT $limit
        """
        return self.run_query(query, {"limit": limit})
