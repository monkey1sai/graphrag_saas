"""Toy hierarchy builder for entities.

This module constructs a simple two‑level hierarchy from entity names.
It is identical to the implementation used previously. Entities are
grouped by the first and second characters of their names. The
hierarchy is represented as a mapping from level numbers to
community structures.
"""

from typing import Dict, List, Tuple


class HierarchyBuilder:
    """Build a toy hierarchy from the resolved entities."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def build(entities: Dict[int, str]) -> Dict[int, Dict[int, List[int]]]:
        level0: Dict[int, List[int]] = {eid: [eid] for eid in entities}
        prefix_groups: Dict[str, List[int]] = {}
        for eid, name in entities.items():
            first_char = name[0].lower() if name else "#"
            prefix_groups.setdefault(first_char, []).append(eid)
        level1: Dict[int, List[int]] = {}
        for idx, (prefix, members) in enumerate(sorted(prefix_groups.items())):
            level1[idx] = members
        second_groups: Dict[str, List[int]] = {}
        for cid, members in level1.items():
            if members:
                name = entities[members[0]]
                key = name[1].lower() if len(name) > 1 else "#"
            else:
                key = "#"
            second_groups.setdefault(key, []).append(cid)
        level2: Dict[int, List[int]] = {}
        for idx, (key, members) in enumerate(sorted(second_groups.items())):
            level2[idx] = members
        return {0: level0, 1: level1, 2: level2}