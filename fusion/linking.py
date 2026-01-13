from collections import defaultdict
from rapidfuzz import fuzz
from .normalize import normalize_text, extract_aliases_from_brackets


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class EntityLinker:
    def __init__(self, fuzzy_threshold: int = 90):
        self.fuzzy_threshold = fuzzy_threshold
        self.uf = UnionFind()
        self.canonical_map: dict[str, str] = {}
        self.alias_map: dict[str, set[str]] = defaultdict(set)
        self.entity_types: dict[str, str] = {}

    def add_entity(self, name: str, entity_type: str = None):
        main_name, aliases = extract_aliases_from_brackets(name)
        normalized = normalize_text(main_name)
        if not normalized:
            return
        self.uf.find(normalized)
        self.alias_map[normalized].add(normalized)
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            if normalized_alias:
                self.uf.union(normalized, normalized_alias)
                self.alias_map[normalized].add(normalized_alias)
        if entity_type:
            self.entity_types[normalized] = entity_type

    def merge_by_rules(self, entities: list[str]):
        normalized_list = [normalize_text(e) for e in entities if e]
        normalized_list = [n for n in normalized_list if n]
        for i, e1 in enumerate(normalized_list):
            for e2 in normalized_list[i + 1:]:
                if e1 == e2:
                    self.uf.union(e1, e2)

    def merge_by_similarity(self, entities: list[str]):
        normalized_list = list(set(normalize_text(e) for e in entities if e))
        normalized_list = [n for n in normalized_list if n]
        for i, e1 in enumerate(normalized_list):
            for e2 in normalized_list[i + 1:]:
                score = fuzz.ratio(e1, e2)
                if score >= self.fuzzy_threshold:
                    self.uf.union(e1, e2)

    def build_canonical_map(self):
        groups = defaultdict(list)
        for entity in self.uf.parent.keys():
            root = self.uf.find(entity)
            groups[root].append(entity)
        for root, members in groups.items():
            canonical = min(members, key=len)
            for m in members:
                self.canonical_map[m] = canonical
                self.alias_map[canonical].update(members)

    def get_canonical(self, name: str) -> str:
        normalized = normalize_text(name)
        if normalized in self.canonical_map:
            return self.canonical_map[normalized]
        return normalized

    def get_aliases(self, canonical: str) -> list[str]:
        return list(self.alias_map.get(canonical, [canonical]))

    def export_mapping(self) -> dict:
        return {
            "canonical_map": dict(self.canonical_map),
            "alias_map": {k: list(v) for k, v in self.alias_map.items()},
        }
