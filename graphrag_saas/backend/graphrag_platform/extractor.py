"""Entity and relationship extraction module.

This file replicates the heuristic entity extractor used in previous
versions of the GraphRAG platform. It identifies multi‑word proper
nouns and all‑caps abbreviations to approximate key entities from
technical documents. Relationships are inferred from co‑occurrences.
"""

import itertools
import re
from typing import Dict, List, Tuple


# stopword lists as defined previously
STOPWORDS = {
    "The", "A", "An", "And", "Or", "In", "Of", "To", "For", "On",
    "This", "That", "As", "At", "By", "With", "From", "Into"
}

LOWER_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "of", "to", "for", "on",
    "this", "that", "as", "at", "by", "with", "from", "into", "is",
    "are", "was", "were", "be", "has", "have", "it", "its", "their",
    "they", "we", "our", "which", "these", "those"
}


class EntityExtractor:
    """Extract entities and simple relationships from text chunks."""

    def __init__(self) -> None:
        self.entities: Dict[str, int] = {}
        self.entity_counter = 0
        self.relations: List[Tuple[int, int, str]] = []
        self.relation_weights: Dict[Tuple[int, int], int] = {}

    def _get_entity_id(self, name: str) -> int:
        if name not in self.entities:
            self.entities[name] = self.entity_counter
            self.entity_counter += 1
        return self.entities[name]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", text)

    def _extract_entities_from_tokens(self, tokens: List[str]) -> List[str]:
        entities: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            if token in STOPWORDS:
                i += 1
                continue
            if token and token[0].isupper() and not token.isupper():
                phrase_tokens = [token]
                j = i + 1
                while j < n and tokens[j][0].isupper() and not tokens[j].isupper():
                    phrase_tokens.append(tokens[j])
                    j += 1
                if len(phrase_tokens) >= 2:
                    entity_name = " ".join(phrase_tokens).strip()
                    entities.append(entity_name)
                    i = j
                    continue
            if token.isupper() and len(token) >= 2:
                entities.append(token)
            i += 1
        for token in tokens:
            if token[0].isupper() and token not in STOPWORDS and not token.isupper():
                if len(token) >= 4:
                    entities.append(token)
        return entities

    def extract(self, chunk: str) -> Tuple[Dict[int, str], List[Tuple[int, int, str]]]:
        tokens = self._tokenize(chunk)
        candidate_entities = self._extract_entities_from_tokens(tokens)
        entity_ids: List[int] = []
        for name in candidate_entities:
            eid = self._get_entity_id(name)
            entity_ids.append(eid)
        unique_ids = list(dict.fromkeys(entity_ids))
        for src_idx, tgt_idx in itertools.combinations(unique_ids, 2):
            if src_idx == tgt_idx:
                continue
            key_fwd = (src_idx, tgt_idx)
            key_rev = (tgt_idx, src_idx)
            self.relation_weights[key_fwd] = self.relation_weights.get(key_fwd, 0) + 1
            self.relation_weights[key_rev] = self.relation_weights.get(key_rev, 0) + 1
            self.relations.append((src_idx, tgt_idx, "related_to"))
            self.relations.append((tgt_idx, src_idx, "related_to"))
        id_to_name = {idx: name for name, idx in self.entities.items()}
        rels_copy = list(self.relations)
        return id_to_name, rels_copy