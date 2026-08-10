"""Step 18 & 19: Evaluation script evaluating 100 benchmark questions and 10-product specification query test."""
import sys, os
sys.path.insert(0, '/home/vconnex/vhomenex_hybrid_rag_v2')

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_rag")

from app.config import get_settings
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.chatbot_service import ChatbotService
from app.generation.ollama_generator import OllamaGenerator

def is_pid_match(pid1: str, pid2: str) -> bool:
    if not pid1 or not pid2:
        return False
    p1 = pid1.lower().strip()
    p2 = pid2.lower().strip()
    return (p1 == p2) or (p1 in p2) or (p2 in p1)

def run_eval():
    settings = get_settings()
    qdrant = QdrantStore(settings)
    bm25 = BM25Store(settings)
    bm25.load_index()
    embedding = OllamaEmbedding(settings)
    generator = OllamaGenerator(settings)
    
    retriever = HybridRetriever(settings, qdrant, bm25, embedding)
    chatbot = ChatbotService(settings, retriever, generator)

    eval_json_path = Path("/home/vconnex/vhomenex_hybrid_rag_v2/data/evaluation/questions.json")
    if not eval_json_path.exists():
        print(f"[-] Evaluation file not found: {eval_json_path}")
        return

    with open(eval_json_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Evaluating %d benchmark questions...", len(questions))

    product_matches = 0
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    rr_sum = 0.0
    cross_product_contaminations = 0

    results_detail = []

    for idx, q in enumerate(questions, 1):
        q_text = q["question"]
        expected_pid = q["expected_product_id"]

        ret_res = retriever.retrieve(q_text, top_k=5)
        res_list = ret_res.get("results", [])
        resolved_pid = ret_res.get("resolved_product", {}).get("product_id", "")

        p_correct = is_pid_match(resolved_pid, expected_pid)
        if p_correct:
            product_matches += 1

        found_rank = None
        top1_pid = ""
        if res_list:
            top1_payload = res_list[0].get("payload", {})
            top1_pid = top1_payload.get("product_id", "")

        # Check if top1 is from another product
        if top1_pid and expected_pid and not is_pid_match(top1_pid, expected_pid):
            cross_product_contaminations += 1

        for r_idx, item in enumerate(res_list, 1):
            payload = item.get("payload", {})
            pid = payload.get("product_id", "")
            if is_pid_match(pid, expected_pid):
                if found_rank is None:
                    found_rank = r_idx

        if found_rank == 1:
            hit_1 += 1
        if found_rank and found_rank <= 3:
            hit_3 += 1
        if found_rank and found_rank <= 5:
            hit_5 += 1

        if found_rank:
            rr_sum += 1.0 / found_rank

        results_detail.append({
            "id": idx,
            "question": q_text,
            "expected_product_id": expected_pid,
            "resolved_product_id": resolved_pid,
            "hit_rank": found_rank,
            "top1_product_id": top1_pid
        })

    total = len(questions)
    p_acc = (product_matches / total) * 100
    h1_acc = (hit_1 / total) * 100
    h3_acc = (hit_3 / total) * 100
    h5_acc = (hit_5 / total) * 100
    mrr = (rr_sum / total)
    contam_rate = (cross_product_contaminations / total) * 100

    report = f"""# RAG EVALUATION BENCHMARK METRICS

- Total Questions Evaluated: {total}
- Product Resolution Accuracy: {p_acc:.2f}% (Threshold >= 95%)
- Hit@1: {h1_acc:.2f}%
- Hit@3: {h3_acc:.2f}% (Threshold >= 90%)
- Hit@5: {h5_acc:.2f}%
- Mean Reciprocal Rank (MRR): {mrr:.4f}
- Cross-Product Contamination Rate: {contam_rate:.2f}% (Threshold <= 2%)
"""
    print(report)

    # Save metrics report
    report_path = Path("/home/vconnex/vhomenex_hybrid_rag_v2/data/evaluation/eval_metrics.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ----------------------------------------------------
    # Step 19: All 10-Product Specification Test
    # ----------------------------------------------------
    logger.info("Running Step 19: All 10-Product Specification Test...")

    catalog_path = Path("/home/vconnex/vhomenex_hybrid_rag_v2/data/product_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    spec_tests = []
    for item in catalog:
        pname = item["product_name"]
        pid = item["product_id"]
        q_spec = f"cho tôi tất cả thông số của {pname}"
        
        chat_res = chatbot.chat(q_spec)
        spec_tests.append({
            "product_id": pid,
            "product_name": pname,
            "query": q_spec,
            "answer": chat_res.answer,
            "sources_count": len(chat_res.sources),
            "sources": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in chat_res.sources]
        })

    out_path = Path("/home/vconnex/vhomenex_hybrid_rag_v2/data/evaluation/all_products_specification_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec_tests, f, ensure_ascii=False, indent=2)

    logger.info("Saved Step 19 specification test results to %s", out_path)

if __name__ == "__main__":
    run_eval()
