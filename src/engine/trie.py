from typing import Set, List

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

    def get_allowed_next_chars(self, prefix: str) -> Set[str]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]

        allowed = set(node.children.keys())
        if node.is_end:
            allowed.add('"')

        return allowed

    def is_valid_path(self, path: str) -> bool:
        """Strictly validates if a multi-character sequence exists in the tree."""
        node = self.root
        for i, char in enumerate(path):
            if char == '"':
                # Valid ONLY if it's the final character and the word is complete
                if node.is_end and i == len(path) - 1:
                    return True
                return False
            if char not in node.children:
                return False
            node = node.children[char]
        return True
