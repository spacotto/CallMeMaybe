from typing import Set, List, Optional


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        # Fast-lookup caches for O(1) validation
        self.valid_suffixes: Set[str] = set()
        self.valid_prefixes: Set[str] = set()


class SchemaTrie:
    """A Prefix Tree with O(1) set-based lookups for fast token masking."""

    def __init__(self, allowed_strings: List[str]) -> None:
        self.root = TrieNode()
        for word in allowed_strings:
            self._insert(word)

    def _insert(self, word: str) -> None:
        node = self.root
        for i, char in enumerate(word):
            remaining_suffix = word[i:]
            node.valid_suffixes.add(remaining_suffix)

            # Precompute all valid token progressions for this node
            for j in range(1, len(remaining_suffix) + 1):
                node.valid_prefixes.add(remaining_suffix[:j])

            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True
        # Allow exact closure from an end node
        node.valid_suffixes.add("")

    def get_node(self, prefix: str) -> Optional[TrieNode]:
        """Traverses the prefix and returns the ending node."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_allowed_next_chars(self, prefix: str) -> Set[str]:
        node = self.get_node(prefix)
        if not node:
            return set()

        allowed = set(node.children.keys())
        if node.is_end:
            allowed.add('"')
        return allowed

    def is_valid_suffix(self, start_node: TrieNode, suffix: str) -> bool:
        """
        Validates a suffix in O(1) time using precomputed sets,
        completely bypassing character-by-character traversal.
        """
        if not suffix:
            return False

        if suffix.endswith('"'):
            # Valid ONLY if the token finishes a complete word
            return suffix[:-1] in start_node.valid_suffixes

        # Valid if the token is an acceptable partial progression
        return suffix in start_node.valid_prefixes
