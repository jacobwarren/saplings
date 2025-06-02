from __future__ import annotations

"""
Tokenizer utilities for Saplings.

This module provides utilities for working with tokens and tokenizers
in a model-agnostic way. It includes functions for getting token counts,
splitting text by token count, and other token-related operations.
"""


def count_tokens(text: str, tokenizer: object | None = None) -> int:
    """
    Count tokens in a string.

    Args:
    ----
        text: Text to count tokens for
        tokenizer: Optional tokenizer to use (falls back to whitespace approximation)

    Returns:
    -------
        int: Approximate token count

    """
    if not text:
        return 0

    if tokenizer is not None:
        # If a tokenizer is provided, use it to get the token count
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))  # type: ignore[attr-defined]
        if hasattr(tokenizer, "tokenize"):
            return len(tokenizer.tokenize(text))  # type: ignore[attr-defined]

    # Basic fallback approximation (4 chars per token is a reasonable approximation)
    # This is used when no tokenizer is available
    return max(1, len(text) // 4)


def split_text_by_tokens(
    text: str, max_tokens: int, tokenizer: object | None = None, overlap_tokens: int = 0
) -> list[str]:
    """
    Split text into chunks of approximately max_tokens tokens.

    Args:
    ----
        text: Text to split
        max_tokens: Maximum tokens per chunk
        tokenizer: Optional tokenizer to use (falls back to approximation)
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
    -------
        List[str]: Text chunks

    """
    if not text:
        return []

    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_chunk_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, tokenizer)

        if line_tokens > max_tokens:
            # Handle long lines by splitting on spaces
            words = line.split(" ")
            word_chunk = []
            word_chunk_tokens = 0

            for word in words:
                word_tokens = count_tokens(word, tokenizer)

                if word_chunk_tokens + word_tokens > max_tokens:
                    if word_chunk:
                        # Add the accumulated words as a chunk
                        chunks.append(" ".join(word_chunk))

                    # Start a new chunk with overlap if needed
                    if overlap_tokens > 0 and word_chunk:
                        overlap_text = " ".join(word_chunk[-overlap_tokens:])
                        word_chunk = word_chunk[-overlap_tokens:] + [word]
                        word_chunk_tokens = count_tokens(overlap_text, tokenizer) + word_tokens
                    else:
                        word_chunk = [word]
                        word_chunk_tokens = word_tokens
                else:
                    word_chunk.append(word)
                    word_chunk_tokens += word_tokens

            if word_chunk:
                # Add any remaining words as a chunk
                chunks.append(" ".join(word_chunk))

            # Reset for next line
            current_chunk = []
            current_chunk_tokens = 0

        elif current_chunk_tokens + line_tokens > max_tokens:
            # Add the current chunk and start a new one
            chunks.append("\n".join(current_chunk))

            # Start a new chunk with overlap if needed
            if overlap_tokens > 0 and current_chunk:
                # Calculate how many lines to include for overlap
                overlap_lines = []
                overlap_tokens_count = 0

                for prev_line in reversed(current_chunk):
                    prev_line_tokens = count_tokens(prev_line, tokenizer)
                    if overlap_tokens_count + prev_line_tokens <= overlap_tokens:
                        overlap_lines.insert(0, prev_line)
                        overlap_tokens_count += prev_line_tokens
                    else:
                        break

                current_chunk = [*overlap_lines, line]
                current_chunk_tokens = overlap_tokens_count + line_tokens
            else:
                current_chunk = [line]
                current_chunk_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_chunk_tokens += line_tokens

    if current_chunk:
        # Add the final chunk
        chunks.append("\n".join(current_chunk))

    return chunks


def truncate_text_tokens(
    text: str, max_tokens: int, tokenizer: object | None = None, truncate_direction: str = "end"
) -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
    ----
        text: Text to truncate
        max_tokens: Maximum number of tokens
        tokenizer: Optional tokenizer to use (falls back to approximation)
        truncate_direction: Direction to truncate ("start" or "end")

    Returns:
    -------
        str: Truncated text

    """
    if not text:
        return ""

    token_count = count_tokens(text, tokenizer)

    if token_count <= max_tokens:
        return text

    if truncate_direction.lower() == "end":
        # Truncate from the end
        chunks = split_text_by_tokens(text, max_tokens, tokenizer, overlap_tokens=0)
        return chunks[0]
    # Truncate from the start
    chunks = split_text_by_tokens(text, max_tokens, tokenizer, overlap_tokens=0)
    return chunks[-1]


def get_tokens_remaining(
    prompt: str | list[dict[str, str]], max_tokens: int, tokenizer: object | None = None
) -> int:
    """
    Calculate how many tokens are remaining after a prompt.

    Args:
    ----
        prompt: The prompt text or chat messages
        max_tokens: Maximum tokens allowed
        tokenizer: Optional tokenizer to use (falls back to approximation)

    Returns:
    -------
        int: Number of tokens remaining

    """
    if isinstance(prompt, str):
        # For string prompts
        prompt_tokens = count_tokens(prompt, tokenizer)
    else:
        # For chat message prompts
        prompt_tokens = 0
        for message in prompt:
            if "content" in message:
                prompt_tokens += count_tokens(message["content"], tokenizer)

    return max(0, max_tokens - prompt_tokens)
