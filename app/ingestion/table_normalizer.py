"""General Table Normalizer for RAG chunking."""
from typing import List, Dict, Any, Optional

class TableNormalizer:
    """Normalizes 2-column key-value tables and multi-column tables into clean prose/lists."""

    @staticmethod
    def normalize(table_data: List[List[str]], max_tokens: int = 500) -> List[str]:
        if not table_data or not any(table_data):
            return []

        # Remove completely empty rows
        clean_rows = []
        for r in table_data:
            cells = [c.strip() for c in r if c and c.strip()]
            if cells:
                clean_rows.append(cells)

        if not clean_rows:
            return []

        # Check if 2-column attribute-value table
        is_kv = all(len(r) == 2 for r in clean_rows)

        if is_kv:
            lines = []
            for r in clean_rows:
                k, v = r[0], r[1]
                lines.append(f"{k}: {v}")
            text = "\n".join(lines)
            if len(text.split()) <= max_tokens:
                return [text]
            else:
                # Split KV table lines into smaller chunks if large
                chunks = []
                curr = []
                curr_len = 0
                for line in lines:
                    line_len = len(line.split())
                    if curr_len + line_len > max_tokens and curr:
                        chunks.append("\n".join(curr))
                        curr = [line]
                        curr_len = line_len
                    else:
                        curr.append(line)
                        curr_len += line_len
                if curr:
                    chunks.append("\n".join(curr))
                return chunks
        else:
            # Multi-column table
            headers = clean_rows[0]
            data_rows = clean_rows[1:] if len(clean_rows) > 1 else clean_rows
            
            lines = []
            for r in data_rows:
                row_parts = []
                for idx, cell in enumerate(r):
                    header_label = headers[idx] if idx < len(headers) else f"Col{idx+1}"
                    row_parts.append(f"{header_label}: {cell}")
                lines.append(" | ".join(row_parts))

            text = "\n".join(lines)
            if len(text.split()) <= max_tokens:
                return [text]
            else:
                chunks = []
                curr = []
                curr_len = 0
                for line in lines:
                    line_len = len(line.split())
                    if curr_len + line_len > max_tokens and curr:
                        chunks.append("\n".join(curr))
                        curr = [line]
                        curr_len = line_len
                    else:
                        curr.append(line)
                        curr_len += line_len
                if curr:
                    chunks.append("\n".join(curr))
                return chunks
