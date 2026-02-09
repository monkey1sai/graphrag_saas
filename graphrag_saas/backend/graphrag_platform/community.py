"""Simple community detection for entities.

Copied from the previous `graphrag_platform`. It assigns entities to
communities based on the first alphabetic character of their names and
builds mappings between entities, communities, and document chunks.
This implementation is a placeholder for proper Louvain clustering.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


class CommunityIndex:
    """Index mapping entities and document chunks to simple communities."""

    def __init__(self, entity_names: Dict[int, str], chunks: List[str]) -> None:
        self.entity_to_comm: Dict[int, str] = {}
        self.entity_names = entity_names
        self.comm_to_chunks: Dict[str, Set[int]] = {}
        # assign community to each entity based on its name
        for eid, name in entity_names.items():
            if not name:
                comm = '#'
            else:
                m = re.search(r"[A-Za-z]", name)
                comm = m.group(0).lower() if m else '#'
            self.entity_to_comm[eid] = comm
        # build name->comm mapping
        self.name_to_comm: Dict[str, str] = {}
        for eid, name in entity_names.items():
            if name:
                self.name_to_comm[name.lower()] = self.entity_to_comm[eid]
        # assign chunks to communities by scanning for entity names
        for idx, chunk in enumerate(chunks):
            text = chunk.lower()
            matched_comms: Set[str] = set()
            for name_lower, comm in self.name_to_comm.items():
                if name_lower in text:
                    matched_comms.add(comm)
            for comm in matched_comms:
                self.comm_to_chunks.setdefault(comm, set()).add(idx)

    def get_comm_for_entity(self, eid: int) -> str:
        return self.entity_to_comm.get(eid, '#')

    def get_chunks_for_comm(self, comm: str) -> Set[int]:
        return self.comm_to_chunks.get(comm, set())

    def get_comm_for_query(self, query: str) -> List[str]:
        q_lower = query.lower()
        matched: Set[str] = set()
        for name_lower, comm in self.name_to_comm.items():
            if name_lower in q_lower:
                matched.add(comm)
        return list(matched)