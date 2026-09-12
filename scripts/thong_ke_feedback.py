"""Thống kê đánh giá 👍/👎 mà người dùng để lại trên câu trả lời.

Mục đích chính: tìm ra NHỮNG CÂU HỎI hay bị đánh giá sai, để biết chỗ nào
cần cải thiện độ chính xác — chứ không chỉ đếm tổng số like/dislike.

Cách chạy:
    python3 scripts/thong_ke_feedback.py
    python3 scripts/thong_ke_feedback.py ui/chat_sessions/feedback.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "ui" / "chat_sessions" / "feedback.jsonl"


def main(log_path: Path) -> None:
    if not log_path.exists():
        print(f"Chưa có dữ liệu đánh giá tại: {log_path}")
        print("(File chỉ được tạo sau khi có người bấm 👍 hoặc 👎 lần đầu.)")
        return

    entries = []
    for line_no, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # Dòng hỏng (ví dụ ghi dở khi mất điện) — bỏ qua đúng dòng đó,
            # không làm hỏng toàn bộ thống kê.
            print(f"  [!] Bỏ qua dòng {line_no} bị lỗi định dạng.")

    if not entries:
        print("Chưa có đánh giá nào.")
        return

    verdicts = Counter(e.get("verdict") for e in entries)
    total = len(entries)
    up = verdicts.get("up", 0)
    down = verdicts.get("down", 0)

    print("=" * 70)
    print(f"TỔNG SỐ ĐÁNH GIÁ: {total}")
    print(f"  👍 Chính xác   : {up:4d}  ({up / total * 100:.0f}%)")
    print(f"  👎 Bị báo sai  : {down:4d}  ({down / total * 100:.0f}%)")
    print("=" * 70)

    if down:
        print("\nCÁC CÂU HỎI BỊ BÁO SAI (cần rà soát):\n")
        # Gom theo câu hỏi: cùng 1 câu bị báo sai nhiều lần là dấu hiệu
        # mạnh hơn nhiều so với một lần đơn lẻ.
        by_question = Counter(
            e.get("question", "(không rõ câu hỏi)")
            for e in entries
            if e.get("verdict") == "down"
        )
        for question, count in by_question.most_common():
            suffix = f"  ({count} lần)" if count > 1 else ""
            print(f"  • {question}{suffix}")

    # Góp ý bằng chữ được tách riêng và in ĐẦY ĐỦ, kể cả khi kèm 👍: đây là
    # phần có giá trị chẩn đoán cao nhất -- một lượt 👎 chỉ nói rằng câu trả
    # lời sai, còn lời góp ý nói rõ sai CHỖ NÀO nên sửa được ngay.
    with_comments = [e for e in entries if (e.get("comment") or "").strip()]
    if with_comments:
        print(f"\n{'=' * 70}")
        print(f"GÓP Ý BẰNG CHỮ TỪ NGƯỜI DÙNG ({len(with_comments)} góp ý)")
        print("=" * 70)
        for e in with_comments:
            icon = "👍" if e.get("verdict") == "up" else "👎"
            print(f"\n{icon} {e.get('timestamp', '')[:16].replace('T', ' ')}")
            print(f"   Câu hỏi : {e.get('question', '')}")
            print(f"   Góp ý   : {e.get('comment', '')}")

    if down:
        print("\n--- Chi tiết câu trả lời bị báo sai gần nhất ---")
        latest = [e for e in entries if e.get("verdict") == "down"][-1]
        print(f"Thời gian: {latest.get('timestamp', '')}")
        print(f"Câu hỏi  : {latest.get('question', '')}")
        print(f"Trả lời  :\n{latest.get('answer', '')[:600]}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    main(path)
