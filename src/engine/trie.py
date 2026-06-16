"""
Phase 1 & 2 engine module: Optimized Prefix Tree (Trie).

This module implements a custom Trie data structure designed specifically
to map out the valid character paths for JSON schema keys and function
names. It relies heavily on pre-computation (caching) to guarantee O(1)
validation times during the critical LLM generation loop.
"""

from typing import Set, List, Optional


class TrieNode:
    """
    A single node within the SchemaTrie.

    This node extends the standard Trie architecture by including
    pre-computed memory caches. Rather than just storing pointers to
    the next character, each node knows every valid remaining string
    fragment that can follow it.

    Attributes:
        children (dict[str, 'TrieNode']): Map of next available characters.
        is_end (bool): True if this node completes a valid allowed string.
        valid_suffixes (Set[str]): O(1) cache of all valid exact completions.
        valid_prefixes (Set[str]): O(1) cache of all valid partial completions.
    """
    def __init__(self) -> None:
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False

        # ------------------------------------------------------------------
        # [CACHING OPTIMIZATION]: O(1) Validation Sets
        # Instead of traversing the tree character-by-character every time
        # the engine needs to validate a multi-byte token, we cache all
        # possible valid continuations directly on the node.
        # ------------------------------------------------------------------
        self.valid_suffixes: Set[str] = set()
        self.valid_prefixes: Set[str] = set()


class SchemaTrie:
    """
    A Prefix Tree with O(1) set-based lookups for fast token masking.

    The Trie takes a list of allowed strings (like function names or
    JSON keys) and builds a traversable graph. It pre-computes valid
    token progressions upon initialization to eliminate CPU-heavy
    string traversals during the live matrix masking phases.
    """

    def __init__(self, allowed_strings: List[str]) -> None:
        """
        Initializes the Trie and populates it with allowed strings.

        Args:
            allowed_strings (List[str]): The exact strings (e.g., schema
                keys or function names) that the model is allowed to generate.
        """
        self.root = TrieNode()
        for word in allowed_strings:
            self._insert(word)

    def _insert(self, word: str) -> None:
        """
        Inserts a word into the tree and builds the O(1) memory caches.

        Args:
            word (str): The valid string to insert.
        """
        node = self.root
        for i, char in enumerate(word):
            # --------------------------------------------------------------
            # [CACHING OPTIMIZATION]: Structural Pre-computation
            # As we build the tree, we aggressively pre-compute and store
            # every single valid string fragment that can possibly follow
            # the current character. This trades initialization memory for
            # massive speed gains during the autoregressive loop.
            # --------------------------------------------------------------
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
        """
        Traverses the prefix and returns the ending node.

        Args:
            prefix (str): The string prefix to follow through the tree.

        Returns:
            Optional[TrieNode]: The final node if the prefix exists,
            otherwise None.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_allowed_next_chars(self, prefix: str) -> Set[str]:
        """
        Returns the set of valid single characters that can immediately
        follow the given prefix.

        Args:
            prefix (str): The currently generated string.

        Returns:
            Set[str]: Allowed next characters, potentially including a quote
            if the string is allowed to terminate here.
        """
        node = self.get_node(prefix)
        if not node:
            return set()

        allowed = set(node.children.keys())
        if node.is_end:
            allowed.add('"')
        return allowed

    def is_valid_suffix(self, start_node: TrieNode, suffix: str) -> bool:
        """
        Validates a suffix in O(1) time using precomputed sets.

        [CACHING OPTIMIZATION]: Cache Exploitation
        Because we pre-computed the `valid_suffixes` and `valid_prefixes`
        sets during initialization, validating a complex multi-character
        LLM token against the schema requires zero tree traversal. We just
        perform a lightning-fast Python set lookup.

        Args:
            start_node (TrieNode): The node representing the current prefix.
            suffix (str): The proposed next token string to validate.

        Returns:
            bool: True if the suffix is structurally allowed, False otherwise.
        """
        if not suffix:
            return False

        if suffix.endswith('"'):
            # Valid ONLY if the token finishes a complete word
            return suffix[:-1] in start_node.valid_suffixes

        # Valid if the token is an acceptable partial progression
        return suffix in start_node.valid_prefixes
