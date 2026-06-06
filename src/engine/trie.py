from typing import Set, List, Optional


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False


class SchemaTrie:
    """A character-level Prefix Tree for dynamic token masking."""

    def __init__(self, allowed_strings: List[str]) -> None:
        self.root = TrieNode()
        for word in allowed_strings:
            self._insert(word)

    def _insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

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
        Validates a suffix starting from a specific node,
        skipping prefix traversal.
        """
        node = start_node
        for i, char in enumerate(suffix):
            if char == '"':
                # Valid ONLY if it's the final character and the word
                # is complete
                is_last_char = (i == len(suffix) - 1)
                if node.is_end and is_last_char:
                    return True
                return False
            if char not in node.children:
                return False
            node = node.children[char]
        return True
