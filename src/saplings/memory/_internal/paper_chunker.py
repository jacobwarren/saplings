from __future__ import annotations

"""
Paper chunking module for Saplings memory.

This module provides specialized functions for chunking research papers
into logical sections and building dependency graphs between them.
"""

import re

from saplings.memory._internal.document import Document, DocumentMetadata

# Magic values as constants
MIN_PAPER_LENGTH = 1000


# Helper to safely get custom metadata
def get_custom_metadata(metadata) -> dict:
    if (
        isinstance(metadata, DocumentMetadata)
        and hasattr(metadata, "custom")
        and isinstance(metadata.custom, dict)
    ):
        return metadata.custom
    return {}


def identify_paper_sections(content: str) -> list[tuple[str, str, int, int]]:
    """
    Identify logical sections in a research paper.

    Args:
    ----
        content: The content of the paper

    Returns:
    -------
        List[Tuple[str, str, int, int]]: List of (section_title, section_content, start_pos, end_pos)

    """
    # Common section titles in research papers
    section_patterns = [
        r"(?:^|\n)(?:ABSTRACT|Abstract)(?:\s|:)",
        r"(?:^|\n)(?:INTRODUCTION|Introduction|1\.?\s+Introduction)(?:\s|:)",
        r"(?:^|\n)(?:RELATED WORK|Related Work|2\.?\s+Related Work)(?:\s|:)",
        r"(?:^|\n)(?:BACKGROUND|Background|Preliminaries)(?:\s|:)",
        r"(?:^|\n)(?:METHODOLOGY|Methodology|METHOD|Method|APPROACH|Approach|3\.?\s+Method)(?:\s|:)",
        r"(?:^|\n)(?:IMPLEMENTATION|Implementation)(?:\s|:)",
        r"(?:^|\n)(?:EXPERIMENT|Experiments|EVALUATION|Evaluation|4\.?\s+Experiments)(?:\s|:)",
        r"(?:^|\n)(?:RESULTS|Results|FINDINGS|Findings|5\.?\s+Results)(?:\s|:)",
        r"(?:^|\n)(?:DISCUSSION|Discussion|6\.?\s+Discussion)(?:\s|:)",
        r"(?:^|\n)(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusions|7\.?\s+Conclusion)(?:\s|:)",
        r"(?:^|\n)(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)(?:\s|:)",
        r"(?:^|\n)(?:APPENDIX|Appendix)(?:\s|:)",
    ]

    # If the content is very short or doesn't look like a paper, use a simpler approach
    if len(content) < MIN_PAPER_LENGTH or ("Title:" in content and "Abstract:" in content):
        # This might be a formatted paper with explicit sections
        result = []

        # Try to extract title
        title_match = re.search(r"Title:(.*?)(?:\n\n|\nAuthors:|\nAbstract:)", content, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            result.append(("Title", title, title_match.start(), title_match.end()))

        # Try to extract abstract
        abstract_match = re.search(
            r"Abstract:(.*?)(?:\n\n|\nIntroduction:|\nPaper Content:)", content, re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            result.append(("Abstract", abstract, abstract_match.start(), abstract_match.end()))

        # Try to extract paper content
        content_match = re.search(r"Paper Content:(.*?)(?:\Z)", content, re.DOTALL)
        if content_match:
            paper_content = content_match.group(1).strip()

            # Further split paper content by page markers if they exist
            page_sections = re.split(r"--- Page \d+ ---", paper_content)
            if len(page_sections) > 1:
                for i, page_content in enumerate(page_sections):
                    if page_content.strip():
                        page_start = content_match.start() + content_match.group(0).find(
                            page_content
                        )
                        page_end = page_start + len(page_content)
                        result.append((f"Page {i + 1}", page_content.strip(), page_start, page_end))
            else:
                result.append(
                    ("Content", paper_content, content_match.start(), content_match.end())
                )

        # If we found sections, return them
        if result:
            return result

    # Find all section matches
    sections = []
    for pattern in section_patterns:
        for match in re.finditer(pattern, content):
            sections.append((match.group().strip(), match.start()))

    # Sort sections by position
    sections.sort(key=lambda x: x[1])

    # Extract section content
    result = []
    for i, (section_title, start_pos) in enumerate(sections):
        # Clean up section title
        clean_title = re.sub(r"^\d+\.?\s*", "", section_title)
        clean_title = re.sub(r"[:\s]+$", "", clean_title)

        # Determine end position (start of next section or end of content)
        end_pos = sections[i + 1][1] if i < len(sections) - 1 else len(content)

        # Extract section content
        section_content = content[start_pos:end_pos].strip()

        result.append((clean_title, section_content, start_pos, end_pos))

    # If no sections were found or the first section doesn't start at the beginning,
    # add a "Header" section for the content before the first identified section
    if not sections or sections[0][1] > 0:
        first_pos = 0 if not sections else sections[0][1]
        header_content = content[:first_pos].strip()
        if header_content:
            result.insert(0, ("Header", header_content, 0, first_pos))

    # If we still have no sections, create a single section with all content
    if not result:
        result.append(("Content", content, 0, len(content)))

    return result


def chunk_paper(document: Document, max_chunk_size: int = 1000) -> list[Document]:
    """
    Chunk a paper document into logical sections.

    Args:
    ----
        document: The paper document to chunk
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
    -------
        List[Document]: List of document chunks representing paper sections

    """
    # Ensure metadata is a DocumentMetadata instance
    if not isinstance(document.metadata, DocumentMetadata):
        document.metadata = DocumentMetadata(
            source="", content_type="text", language="en", author="unknown", tags=[], custom={}
        )

    # Identify paper sections
    sections = identify_paper_sections(document.content)

    # Extract paper_id from document.id or metadata
    paper_id = document.id
    if paper_id.startswith("paper_"):
        paper_id = paper_id[6:]  # Remove "paper_" prefix
    elif "paper_id" in get_custom_metadata(document.metadata):
        paper_id = get_custom_metadata(document.metadata).get("paper_id", paper_id)

    chunks = []
    for section_title, section_content, start_pos, end_pos in sections:
        # Skip empty sections
        if not section_content.strip():
            continue

        # Clean section title for use in ID
        clean_section_name = (
            section_title.lower().replace(" ", "_").replace(":", "").replace(",", "")
        )

        # If section is too large, split it further
        if len(section_content) > max_chunk_size:
            # Split into paragraphs
            paragraphs = re.split(r"\n\s*\n", section_content)

            # Group paragraphs into chunks of appropriate size
            current_chunk = ""
            paragraph_chunks = []

            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        paragraph_chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"

            if current_chunk:
                paragraph_chunks.append(current_chunk.strip())

            # Create a document for each paragraph chunk
            for i, chunk_content in enumerate(paragraph_chunks):
                chunk_id = f"{document.id}_{clean_section_name}_{i}"

                # Create metadata for the chunk
                if isinstance(document.metadata, DocumentMetadata):
                    tags = (
                        document.metadata.tags.copy() if hasattr(document.metadata, "tags") else []
                    )
                    chunk_metadata = DocumentMetadata(
                        source=getattr(document.metadata, "source", ""),
                        content_type=getattr(document.metadata, "content_type", "text"),
                        language=getattr(document.metadata, "language", "en"),
                        author=getattr(document.metadata, "author", "unknown"),
                        tags=tags,
                        custom={
                            **get_custom_metadata(document.metadata),
                            "parent_id": document.id,
                            "paper_id": paper_id,
                            "section": section_title,
                            "subsection_index": i if "i" in locals() else 0,
                            "start_char": start_pos,
                            "end_char": end_pos,
                        },
                    )
                else:
                    chunk_metadata = DocumentMetadata(
                        source="",
                        content_type="text",
                        language="en",
                        author="unknown",
                        tags=[],
                        custom={
                            "parent_id": document.id,
                            "paper_id": paper_id,
                            "section": section_title,
                            "subsection_index": i if "i" in locals() else 0,
                            "start_char": start_pos,
                            "end_char": end_pos,
                        },
                    )

                chunk = Document(
                    id=chunk_id,
                    content=chunk_content,
                    metadata=chunk_metadata,
                    embedding=None,
                )

                chunks.append(chunk)
        else:
            # Create a document for the entire section
            chunk_id = f"{document.id}_{clean_section_name}"

            # Create metadata for the chunk
            chunk_metadata = DocumentMetadata(
                source=document.metadata.source,
                content_type=document.metadata.content_type,
                language=document.metadata.language,
                author=document.metadata.author,
                tags=document.metadata.tags.copy(),
                custom={
                    **get_custom_metadata(document.metadata),
                    "parent_id": document.id,
                    "paper_id": paper_id,
                    "section": section_title,
                    "start_char": start_pos,
                    "end_char": end_pos,
                },
            )

            chunk = Document(
                id=chunk_id,
                content=section_content,
                metadata=chunk_metadata,
                embedding=None,
            )

            chunks.append(chunk)

    # If no sections were found, fall back to standard chunking
    if not chunks:
        # Create a single chunk with the entire content
        chunk_id = f"{document.id}_content"

        chunk_metadata = DocumentMetadata(
            source=document.metadata.source,
            content_type=document.metadata.content_type,
            language=document.metadata.language,
            author=document.metadata.author,
            tags=document.metadata.tags.copy(),
            custom={
                **get_custom_metadata(document.metadata),
                "parent_id": document.id,
                "paper_id": paper_id,
                "section": "Content",
                "start_char": 0,
                "end_char": len(document.content),
            },
        )

        chunk = Document(
            id=chunk_id,
            content=document.content,
            metadata=chunk_metadata,
            embedding=None,
        )

        chunks.append(chunk)

    document.chunks = chunks
    return chunks


def build_section_relationships(chunks: list[Document]) -> list[dict]:
    """
    Build relationships between paper sections based on logical structure.

    Args:
    ----
        chunks: List of document chunks representing paper sections

    Returns:
    -------
        List[Dict]: List of relationship dictionaries

    """
    relationships = []

    # If there are no chunks or only one chunk, return empty relationships
    if len(chunks) <= 1:
        return relationships

    # Map of section titles to chunk IDs
    section_map = {}
    for chunk in chunks:
        section = get_custom_metadata(chunk.metadata).get("section", "")
        if not section:
            # Try to extract section from the ID
            id_parts = chunk.id.split("_")
            if len(id_parts) > 2:
                section = id_parts[-2]  # Assuming format like paper_id_section_index

        if section:
            if section not in section_map:
                section_map[section] = []
            section_map[section].append(chunk.id)

    # If we couldn't extract sections, create sequential relationships
    if not section_map:
        # Sort chunks by ID (assuming they have sequential IDs)
        sorted_chunks = sorted(chunks, key=lambda x: x.id)

        # Create sequential relationships
        for i in range(len(sorted_chunks) - 1):
            source_id = sorted_chunks[i].id
            target_id = sorted_chunks[i + 1].id
            relationships.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relationship_type": "follows",
                    "metadata": {"confidence": 1.0},
                }
            )

        return relationships

    # Define the typical flow of a research paper
    section_flow = [
        "Title",
        "Header",
        "Abstract",
        "Introduction",
        "Related Work",
        "Background",
        "Methodology",
        "Method",
        "Approach",
        "Implementation",
        "Experiments",
        "Evaluation",
        "Results",
        "Findings",
        "Discussion",
        "Conclusion",
        "Conclusions",
        "References",
        "Appendix",
        "Page 1",
        "Page 2",
        "Page 3",
        "Page 4",
        "Page 5",
        "Content",
    ]

    # Create relationships based on section flow
    for i, section in enumerate(section_flow):
        if section in section_map:
            # Connect to previous section
            # Threshold for i
            I_THRESHOLD = 0
            if i > I_THRESHOLD:
                for prev_i in range(i - 1, -1, -1):
                    prev_section = section_flow[prev_i]
                    if prev_section in section_map:
                        for source_id in section_map[prev_section]:
                            for target_id in section_map[section]:
                                relationships.append(
                                    {
                                        "source_id": source_id,
                                        "target_id": target_id,
                                        "relationship_type": "follows",
                                        "metadata": {"confidence": 0.9},
                                    }
                                )
                        break

            # Connect to next section
            if i < len(section_flow) - 1:
                for next_i in range(i + 1, len(section_flow)):
                    next_section = section_flow[next_i]
                    if next_section in section_map:
                        for source_id in section_map[section]:
                            for target_id in section_map[next_section]:
                                relationships.append(
                                    {
                                        "source_id": source_id,
                                        "target_id": target_id,
                                        "relationship_type": "precedes",
                                        "metadata": {"confidence": 0.9},
                                    }
                                )
                        break

    # Connect subsections within the same section
    for section, chunk_ids in section_map.items():
        if len(chunk_ids) > 1:
            # Sort chunks by subsection index or ID if index not available
            def get_sort_key(chunk_id):
                chunk = next((c for c in chunks if c.id == chunk_id), None)
                if chunk:
                    index = get_custom_metadata(chunk.metadata).get("subsection_index")
                    if index is not None:
                        return index
                    # Extract index from ID if possible
                    parts = chunk_id.split("_")
                    if len(parts) > 1 and parts[-1].isdigit():
                        return int(parts[-1])
                return 0

            sorted_chunk_ids = sorted(chunk_ids, key=get_sort_key)

            # Connect adjacent subsections
            for i in range(len(sorted_chunk_ids) - 1):
                source_id = sorted_chunk_ids[i]
                target_id = sorted_chunk_ids[i + 1]
                relationships.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_type": "continues",
                        "metadata": {"confidence": 1.0},
                    }
                )

    # Add special relationships for certain sections
    if "Abstract" in section_map and "Conclusion" in section_map:
        for source_id in section_map["Abstract"]:
            for target_id in section_map["Conclusion"]:
                relationships.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_type": "summarizes",
                        "metadata": {"confidence": 0.8},
                    }
                )

    if "Introduction" in section_map and "Related Work" in section_map:
        for source_id in section_map["Introduction"]:
            for target_id in section_map["Related Work"]:
                relationships.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_type": "references",
                        "metadata": {"confidence": 0.7},
                    }
                )

    if "Method" in section_map and "Results" in section_map:
        for source_id in section_map["Method"]:
            for target_id in section_map["Results"]:
                relationships.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_type": "produces",
                        "metadata": {"confidence": 0.8},
                    }
                )

    return relationships
