"""So sánh hai lần chạy test để phát hiện câu trả lời nào đã thay đổi.

Dùng để kiểm chứng một bản vá có làm hỏng câu nào đang đúng hay không —
thay vì đoán, so trực tiếp kết quả trước và sau.

Cách chạy:
    python3 scripts/so_sanh_ket_qua.py ket_qua_cu.txt ket_qua_moi.txt

In ra:
- Số câu giữ nguyên / đã đổi
- Danh sách câu đã đổi, kèm trích đoạn cũ và mới để đối chiếu
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def doc_ket_qua(path: Path) -> dict[str, str]:
    """Tách file kết quả thành {câu hỏi: câu trả lời}."""
    text = path.read_text(encoding="utf-8", errors="replace")
    ket_qua: dict[str, str] = {}

    # Mỗi khối bắt đầu bằng dòng "    Hỏi: ..." rồi tới dòng gạch ngang,
    # phần trả lời chạy tới khi gặp mục kế tiếp hoặc dấu phân cách sản phẩm.
    khoi = re.split(r"\n(?=\[\d+\.\d+\])", text)
    for k in khoi:
        m = re.search(r"^\s*Hỏi:\s*(.+?)\s*$", k, re.M)
        if not m:
            continue
        cau_hoi = m.group(1).strip()
        # Bỏ phần tiêu đề, lấy phần sau dòng gạch ngang
        parts = re.split(r"-{20,}\n", k, maxsplit=1)
        tra_loi = parts[1].strip() if len(parts) > 1 else ""
        # Cắt bỏ phần thừa nếu khối chứa cả tiêu đề sản phẩm kế tiếp
        tra_loi = re.split(r"\n={20,}", tra_loi)[0].strip()
        ket_qua[cau_hoi] = tra_loi
    return ket_qua


def rut_gon(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def main(cu: Path, moi: Path) -> None:
    if not cu.exists():
        print(f"Không tìm thấy file cũ: {cu}")
        return
    if not moi.exists():
        print(f"Không tìm thấy file mới: {moi}")
        return

    kq_cu = doc_ket_qua(cu)
    kq_moi = doc_ket_qua(moi)

    chung = set(kq_cu) & set(kq_moi)
    chi_co_cu = set(kq_cu) - set(kq_moi)
    chi_co_moi = set(kq_moi) - set(kq_cu)

    giong = [q for q in chung if kq_cu[q].strip() == kq_moi[q].strip()]
    khac = [q for q in chung if kq_cu[q].strip() != kq_moi[q].strip()]

    print("=" * 78)
    print(f"Câu có ở cả hai lần chạy : {len(chung)}")
    print(f"  - Giữ nguyên           : {len(giong)}")
    print(f"  - ĐÃ ĐỔI               : {len(khac)}")
    if chi_co_cu:
        print(f"Chỉ có ở lần chạy cũ     : {len(chi_co_cu)}")
    if chi_co_moi:
        print(f"Chỉ có ở lần chạy mới    : {len(chi_co_moi)}  (mục mới được test)")
    print("=" * 78)

    if khac:
        print("\nCÁC CÂU ĐÃ ĐỔI — cần đọc kỹ xem đổi theo hướng tốt hay xấu:\n")
        for i, q in enumerate(sorted(khac), 1):
            print(f"[{i}] {q}")
            print(f"    CŨ : {rut_gon(kq_cu[q])}")
            print(f"    MỚI: {rut_gon(kq_moi[q])}")
            print()

    if chi_co_moi:
        print("\nCÂU MỚI XUẤT HIỆN (trước đây script chưa trích được mục này):\n")
        for q in sorted(chi_co_moi):
            print(f"  • {q}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
