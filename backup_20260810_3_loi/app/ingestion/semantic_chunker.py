"""Refactored general Semantic Chunker supporting all 10 Vhomenex products."""
import re
import os
from typing import List, Dict, Any, Optional
from app.ingestion.section_parser import SectionParser, SectionNode
from app.ingestion.table_normalizer import TableNormalizer
from app.ingestion.content_classifier import ContentTypeClassifier
from app.ingestion.document_metadata import DocumentMetadataExtractor
from app.models.chunk import Chunk

INTERNAL_SECTION_PATTERNS = [
    r"metadata sản phẩm",
    r"quy tắc tạo chunk",
    r"mẫu metadata chunk",
    r"hướng dẫn biên soạn",
    r"hướng dẫn chuẩn hóa",
    r"quy ước dữ liệu",
    r"phụ lục dành cho người soạn",
    r"nội dung nội bộ",
    r"do_not_ingest"
]

PLACEHOLDER_LINES = [
    r"\[nội dung chunk\]",
    r"\[bổ sung khi chunking\]",
    r"\[điền nội dung\]",
    r"\[vd:",
    r"bên anh",
    r"bên kinh doanh chốt",
    r"^todo$",
    r"^tbd$"
]

def build_embedding_text(chunk: Chunk) -> str:
    parts = []
    if chunk.product_name:
        parts.append(f"Sản phẩm: {chunk.product_name}")
    if chunk.title:
        parts.append(f"Mục: {chunk.title}")
    if chunk.content:
        parts.append(chunk.content)
    return "\n".join(parts)

class SemanticChunker:
    def __init__(self, target_tokens: int = 400, max_tokens: int = 750, overlap_tokens: int = 50):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.parser = SectionParser()
        self.meta_extractor = DocumentMetadataExtractor()

    def _is_internal_section(self, title: str) -> bool:
        t_clean = title.lower()
        return any(re.search(pat, t_clean) for pat in INTERNAL_SECTION_PATTERNS)

    def _clean_content(self, text: str) -> str:
        lines = text.splitlines()
        clean = []
        for line in lines:
            l_strip = line.strip()
            l_lower = l_strip.lower()
            if any(re.search(pat, l_lower) for pat in PLACEHOLDER_LINES):
                continue
            clean.append(line)
        return "\n".join(clean).strip()

    def chunk(
        self,
        elements: list,
        source_file: str = "",
        product_id: str = "",
        product_name: str = "",
        product_group: str = "",
        model: str = ""
    ) -> List[Chunk]:
        filename = os.path.basename(source_file)
        doc_meta = self.meta_extractor.get_metadata_for_file(filename, elements)
        
        # Override with explicit metadata if passed
        if product_id: doc_meta['product_id'] = product_id
        if product_name: doc_meta['product_name'] = product_name
        if product_group: doc_meta['product_group'] = product_group
        if model: doc_meta['model'] = model

        root_node = self.parser.parse(elements)
        chunks: List[Chunk] = []

        def traverse(node: SectionNode):
            if self._is_internal_section(node.title):
                return

            if node.elements:
                sec_title = node.title or "Mô tả sản phẩm"
                parent_title = node.parent.title if node.parent else ""
                content_type = ContentTypeClassifier.classify(sec_title, "", parent_title)

                body_parts = []
                for elem in node.elements:
                    if elem.kind in ["paragraph", "heading", "bullet", "numbered", "caption"]:
                        cleaned_p = self._clean_content(elem.text)
                        if cleaned_p:
                            body_parts.append(cleaned_p)
                    elif elem.kind == "table" and hasattr(elem, "raw_rows") and elem.raw_rows:
                        normalized_table_parts = TableNormalizer.normalize(elem.raw_rows, max_tokens=self.max_tokens)
                        for tp in normalized_table_parts:
                            cleaned_tp = self._clean_content(tp)
                            if cleaned_tp:
                                body_parts.append(cleaned_tp)
                    elif elem.kind == "table" and hasattr(elem, "text") and elem.text:
                        cleaned_tp = self._clean_content(elem.text)
                        if cleaned_tp:
                            body_parts.append(cleaned_tp)

                if body_parts:
                    full_text = "\n\n".join(body_parts).strip()
                    tokens = len(full_text.split())

                    if tokens <= self.max_tokens:
                        cid = f"{doc_meta['product_id']}_{len(chunks)+1:03d}"
                        chunk_obj = Chunk(
                            chunk_id=cid,
                            product_id=doc_meta['product_id'],
                            product_name=doc_meta['product_name'],
                            product_group=doc_meta['product_group'],
                            model=doc_meta['model'],
                            content_type=content_type,
                            title=sec_title,
                            content=full_text,
                            source_file=filename,
                            source_document=filename,
                            source_section=sec_title,
                            token_count=tokens
                        )
                        chunks.append(chunk_obj)
                    else:
                        sub_parts = full_text.split("\n\n")
                        curr_text = []
                        curr_len = 0
                        for part in sub_parts:
                            p_tokens = len(part.split())
                            if curr_len + p_tokens > self.target_tokens and curr_text:
                                cid = f"{doc_meta['product_id']}_{len(chunks)+1:03d}"
                                body_c = "\n\n".join(curr_text)
                                chunk_obj = Chunk(
                                    chunk_id=cid,
                                    product_id=doc_meta['product_id'],
                                    product_name=doc_meta['product_name'],
                                    product_group=doc_meta['product_group'],
                                    model=doc_meta['model'],
                                    content_type=content_type,
                                    title=sec_title,
                                    content=body_c,
                                    source_file=filename,
                                    source_document=filename,
                                    source_section=sec_title,
                                    token_count=len(body_c.split())
                                )
                                chunks.append(chunk_obj)
                                curr_text = [part]
                                curr_len = p_tokens
                            else:
                                curr_text.append(part)
                                curr_len += p_tokens

                        if curr_text:
                            cid = f"{doc_meta['product_id']}_{len(chunks)+1:03d}"
                            body_c = "\n\n".join(curr_text)
                            chunk_obj = Chunk(
                                chunk_id=cid,
                                product_id=doc_meta['product_id'],
                                product_name=doc_meta['product_name'],
                                product_group=doc_meta['product_group'],
                                model=doc_meta['model'],
                                content_type=content_type,
                                title=sec_title,
                                content=body_c,
                                source_file=filename,
                                source_document=filename,
                                source_section=sec_title,
                                token_count=len(body_c.split())
                            )
                            chunks.append(chunk_obj)

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return chunks
