"""Deterministic helpers for comparison answers.

The local LLM occasionally merges all comparison headers into the first cell.
The backend already knows the compared products, so it can safely rebuild only
the table header without changing any factual cell produced from RAG context.
"""
from __future__ import annotations

import re
from typing import Iterable, List


_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


def _cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def _escape_cell(value: str) -> str:
    return value.replace("|", r"\|").strip()


def repair_comparison_table(answer: str, product_names: Iterable[str]) -> str:
    """Repair a malformed comparison-table header using trusted product names.

    Only the header and separator are replaced. Body rows and prose remain
    untouched, so this function cannot invent or move product facts.
    """
    names = [_escape_cell(name) for name in product_names if name.strip()]
    if len(names) < 2 or not answer:
        return answer

    lines = answer.splitlines()
    for index in range(len(lines) - 1):
        if not lines[index].lstrip().startswith("|"):
            continue
        if not lines[index + 1].lstrip().startswith("|"):
            continue
        if not _is_separator(lines[index + 1]):
            continue

        lines[index] = "| Tiêu chí | " + " | ".join(names) + " |"
        lines[index + 1] = "|---|" + "|".join("---" for _ in names) + "|"
        return "\n".join(lines)

    return answer
