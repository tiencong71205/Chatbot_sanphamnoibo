"""Vconnex Smart Assistant — chat UI.

Tuân thủ nghiêm ngặt 2 nguyên tắc kỹ thuật:
1. Không can thiệp hoặc ẩn header / toolbar / sidebar-toggle mặc định của Streamlit.
2. Đồng bộ toàn bộ bảng màu với .streamlit/config.toml qua khối biến CSS :root.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
LOGO_PATH = Path(__file__).parent / "assets" / "vconnex_logo.png"
HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", str(Path(__file__).parent / "chat_sessions")))
REQUEST_TIMEOUT = 180
TITLE_MAX_CHARS = 46

_SOURCE_FOOTER_RE = re.compile(r"\n{0,3}📚[^\n]*Nguồn tham khảo\s*:.*$", re.IGNORECASE | re.DOTALL)
_FOOTER_MARKER = "📚"


def strip_source_footer(text: str) -> str:
    return _SOURCE_FOOTER_RE.sub("", text or "").rstrip()


# ─────────────────────── Chat history persistence ───────────────────────

def _history_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def _conversation_path(conversation_id: str) -> Path:
    return _history_dir() / f"{conversation_id}.json"


def _derive_title(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("content", "").strip():
            text = " ".join(m["content"].split())
            return text[:TITLE_MAX_CHARS] + ("…" if len(text) > TITLE_MAX_CHARS else "")
    return "Cuộc trò chuyện mới"


def _list_conversations() -> List[Dict[str, Any]]:
    items = []
    for path in _history_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items.append({
            "id": path.stem,
            "title": data.get("title") or "Cuộc trò chuyện mới",
            "updated_at": data.get("updated_at", ""),
        })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def _load_conversation(conversation_id: str) -> List[Dict[str, Any]]:
    path = _conversation_path(conversation_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("messages", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_conversation(conversation_id: str, messages: List[Dict[str, Any]]) -> None:
    if not messages:
        return
    payload = {
        "title": _derive_title(messages),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    try:
        _conversation_path(conversation_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _delete_conversation(conversation_id: str) -> None:
    path = _conversation_path(conversation_id)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _new_conversation() -> None:
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []


FEEDBACK_LOG_NAME = "feedback.jsonl"


def _record_feedback(
    conversation_id: str,
    messages: List[Dict[str, Any]],
    message_index: int,
    verdict: str,
) -> None:
    """Persist a 👍/👎 on one answer, in two places for two different uses.

    1. Onto the message itself inside the conversation JSON, so reopening a
       past conversation shows which answers were already rated instead of
       silently resetting to unrated.
    2. Appended to feedback.jsonl with the question that produced the answer
       and the answer text — that pairing is the point: a bare thumbs-down
       count says nothing actionable, but "these are the questions people
       marked wrong" is directly usable for accuracy work.

    Written append-only as JSON Lines so a crash mid-write can at worst lose
    the last line rather than corrupt the whole history, and so the file can
    grow without being rewritten each time.
    """
    if not (0 <= message_index < len(messages)):
        return

    messages[message_index]["feedback"] = verdict
    _save_conversation(conversation_id, messages)

    question = ""
    for earlier in reversed(messages[:message_index]):
        if earlier.get("role") == "user":
            question = earlier.get("content", "")
            break

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "verdict": verdict,
        "question": question,
        "answer": messages[message_index].get("content", ""),
    }
    try:
        # _history_dir() creates the folder if missing, same as conversations.
        with (_history_dir() / FEEDBACK_LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Losing a feedback line must never break the chat itself.
        pass


# ─────────────────────────── Backend calls ───────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_products() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/products", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=15, show_spinner=False)
def _backend_is_online() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False


def _stream_answer(
    question: str, product_id: Optional[str], history: List[Dict[str, str]]
) -> Generator[str, None, None]:
    payload = {"question": question, "product_id": product_id, "history": history, "debug": False}
    with requests.post(
        f"{BACKEND_URL}/api/chat/stream", json=payload, stream=True, timeout=REQUEST_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        event_type = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                event_type = None
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if event_type == "done":
                _stream_answer.final_response = data  # type: ignore[attr-defined]
                return
            if event_type == "error":
                _stream_answer.final_response = None  # type: ignore[attr-defined]
                raise RuntimeError(data.get("message") or "Lỗi không xác định từ máy chủ.")
            if event_type in ("delta", "replace") and data.get("text"):
                yield data["text"]


def _extract_images(sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen: set = set()
    images: List[Dict[str, str]] = []
    for src in sources or []:
        for data_uri in src.get("images", []) or []:
            if data_uri and data_uri not in seen:
                seen.add(data_uri)
                images.append({
                    "uri": data_uri,
                    "section": src.get("source_section", "") or "",
                    "document": src.get("document_title", "") or "",
                })
    return images


def _decode_image(data_uri: str) -> Optional[bytes]:
    try:
        _, b64_part = data_uri.split(",", 1)
        return base64.b64decode(b64_part)
    except (ValueError, TypeError):
        return None


def _call_chat_sync(question: str, product_id: Optional[str], history: List[Dict[str, str]]) -> Dict[str, Any]:
    resp = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={"question": question, "product_id": product_id, "history": history, "debug": False},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _render_answer(question: str, product_id: Optional[str], history: List[Dict[str, str]]) -> Dict[str, Any]:
    placeholder = st.empty()
    started_at = time.time()
    try:
        accumulated = ""
        for delta in _stream_answer(question, product_id, history):
            accumulated += delta
            marker_at = accumulated.find(_FOOTER_MARKER)
            visible = accumulated[:marker_at].rstrip() if marker_at != -1 else accumulated
            placeholder.markdown(visible + " ▌")

        final_response = getattr(_stream_answer, "final_response", None)
        if final_response is not None:
            answer = strip_source_footer(final_response.get("answer", accumulated))
            latency_ms = final_response.get("latency_ms", (time.time() - started_at) * 1000)
            images = _extract_images(final_response.get("sources", []))
        else:
            answer = strip_source_footer(accumulated)
            latency_ms = (time.time() - started_at) * 1000
            images = []

        placeholder.markdown(answer)
        return {"answer": answer, "latency_ms": latency_ms, "images": images}

    except Exception:
        placeholder.markdown("_Đang xử lý kết nối lại..._")
        data = _call_chat_sync(question, product_id, history)
        answer = strip_source_footer(data.get("answer", "Không tìm thấy câu trả lời phù hợp."))
        placeholder.markdown(answer)
        return {
            "answer": answer,
            "latency_ms": data.get("latency_ms", (time.time() - started_at) * 1000),
            "images": _extract_images(data.get("sources", [])),
        }


# ────────────────────────────── Page setup ──────────────────────────────

st.set_page_config(page_title="Vhomenex AI Assistant", page_icon="🔷", layout="wide")

if "conversation_id" not in st.session_state:
    _new_conversation()

st.markdown(
    """
    <style>
        :root {
            --accent: #E8834A;
            --accent-soft: #FDF4EC;
            --accent-dark: #C96A34;
            --ink: #1F242D;
            --text-muted: #64748B;
            --border: #E2E8F0;
            --surface-soft: #F8FAFC;
            --warn-bg: #FFFBEB;
            --warn-border: #FCD34D;
            --warn-ink: #92400E;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--ink);
        }
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 7.5rem;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.2rem 0 1rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }
        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--ink);
            line-height: 1.2;
            letter-spacing: -0.01em;
        }
        .brand-subtitle {
            font-size: 0.72rem;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .sidebar-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 1.2rem 0 0.4rem 0.2rem;
        }

        .empty-state {
            text-align: center;
            padding: 14vh 1rem 2rem 1rem;
        }
        .empty-state h1 {
            font-size: clamp(1.8rem, 3.5vw, 2.3rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--ink);
            margin-bottom: 0.5rem;
        }
        .empty-state p {
            font-size: 0.95rem;
            color: var(--text-muted);
            max-width: 480px;
            margin: 0 auto;
        }

        /* =======================================================
           CÂU HỎI USER: CĂN LỆCH PHẢI (Right-aligned)
           ======================================================= */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            max-width: 82% !important;
            background-color: var(--accent-soft) !important;
            border: 1px solid rgba(232, 131, 74, 0.3) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 0.5rem 1rem !important;
            box-shadow: var(--shadow-sm);
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
            text-align: left;
        }

        /* =======================================================
           CÂU TRẢ LỜI BOT: CĂN LỆCH TRÁI (Left-aligned)
           ======================================================= */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
            flex-direction: row !important;
            margin-left: 0 !important;
            margin-right: auto !important;
            max-width: 88% !important;
            background-color: #FFFFFF !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 0.6rem 1.1rem !important;
            box-shadow: var(--shadow-sm);
        }

        /* Thẻ ảnh sơ đồ kỹ thuật */
        .doc-card-container {
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0 0.8rem 0;
            background: #FFFFFF;
            box-shadow: var(--shadow-sm);
        }
        .doc-card-caption {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            background: var(--surface-soft);
            border-top: 1px solid var(--border);
            padding: 0.45rem 0.75rem;
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .response-meta {
            color: var(--text-muted);
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.4rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            background: var(--surface-soft);
            font-size: 0.78rem;
            color: var(--text-muted);
            border: 1px solid var(--border);
            width: 100%;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.online { background: #10B981; box-shadow: 0 0 0 2px rgba(16,185,129,0.2); }
        .status-dot.offline { background: #EF4444; }

        .disclaimer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-top: 0.6rem;
        }

        /* Tối ưu màn hình thiết bị di động */
        @media (max-width: 768px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                max-width: 92% !important;
            }
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
                max-width: 96% !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _group_conversations_by_age(
    conversations: List[Dict[str, Any]]
) -> List[tuple]:
    now = datetime.now(timezone.utc)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "HÔM NAY": [], "7 NGÀY QUA": [], "30 NGÀY QUA": [], "CŨ HƠN": [],
    }
    for conv in conversations:
        try:
            updated = datetime.fromisoformat(conv.get("updated_at", ""))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            days = (now - updated).days
        except (ValueError, TypeError):
            days = 9999
        if days < 1:
            buckets["HÔM NAY"].append(conv)
        elif days < 7:
            buckets["7 NGÀY QUA"].append(conv)
        elif days < 30:
            buckets["30 NGÀY QUA"].append(conv)
        else:
            buckets["CŨ HƠN"].append(conv)
    return [(label, items) for label, items in buckets.items() if items]


def _group_products_by_category(products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    categories: Dict[str, List[Dict[str, Any]]] = {
        "Công tắc & Ổ cắm": [], "Khóa điện tử": [], "Cảm biến": [],
        "Gateway & Trung tâm": [], "Điều khiển & Giám sát": [], "Khác": [],
    }
    for product in products:
        name = str(product.get("product_name", "")).lower()
        if "công tắc" in name or "ổ cắm" in name:
            categories["Công tắc & Ổ cắm"].append(product)
        elif "khóa" in name and "gateway" not in name:
            categories["Khóa điện tử"].append(product)
        elif "cảm biến" in name:
            categories["Cảm biến"].append(product)
        elif "hồng ngoại" in name or "rèm" in name or "giám sát" in name or "công tơ" in name:
            categories["Điều khiển & Giám sát"].append(product)
        elif "gateway" in name or "trung tâm" in name:
            categories["Gateway & Trung tâm"].append(product)
        else:
            categories["Khác"].append(product)
    return {label: items for label, items in categories.items() if items}


products = _fetch_products()
backend_online = _backend_is_online()

# ────────────────────────── Sidebar ──────────────────────────

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

    st.markdown(
        """
        <div class="brand-container">
            <div>
                <div class="brand-title">Vhomenex</div>
                <div class="brand-subtitle">AI Technical Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋ Đoạn chat mới", use_container_width=True, type="primary"):
        _new_conversation()
        st.rerun()

    conversations = _list_conversations()
    if not conversations:
        st.caption("Chưa có phiên hội thoại nào.")

    for group_label, group_items in _group_conversations_by_age(conversations):
        st.markdown(f'<div class="sidebar-label">{group_label}</div>', unsafe_allow_html=True)
        for conv in group_items:
            is_active = conv["id"] == st.session_state.conversation_id
            row = st.columns([5.5, 1.2])
            with row[0]:
                if st.button(
                    conv["title"],
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = _load_conversation(conv["id"])
                    st.rerun()
            with row[1]:
                if st.button("✕", key=f"del_{conv['id']}", help="Xóa hội thoại"):
                    _delete_conversation(conv["id"])
                    if is_active:
                        _new_conversation()
                    st.rerun()

    st.markdown('<div class="sidebar-label">Kho tài liệu sản phẩm</div>', unsafe_allow_html=True)
    for category, items in _group_products_by_category(products).items():
        with st.expander(f"📁 {category} ({len(items)})", expanded=False):
            for product in items:
                st.caption(f"• {product.get('product_name', product.get('product_id', ''))}")

    st.markdown('<div class="sidebar-label">Phạm vi tra cứu</div>', unsafe_allow_html=True)
    product_options = ["Tất cả tài liệu"] + [
        p.get("product_name", p.get("product_id", "")) for p in products
    ]
    selected_product_label = st.selectbox("Sản phẩm mục tiêu", product_options, label_visibility="collapsed")
    selected_product_id = None
    if selected_product_label != "Tất cả tài liệu" and products:
        idx = product_options.index(selected_product_label) - 1
        if 0 <= idx < len(products):
            selected_product_id = products[idx].get("product_id")

    st.markdown('<div class="sidebar-label">Trạng thái hệ thống</div>', unsafe_allow_html=True)
    status_dot = "online" if backend_online else "offline"
    status_label = "Máy chủ sẵn sàng" if backend_online else "Mất kết nối máy chủ"
    st.markdown(
        f"""
        <div class="status-badge">
            <span class="status-dot {status_dot}"></span>
            <span>{status_label} ({len(products)} thiết bị)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_message_images(images: List[Any]) -> None:
    if not images:
        return
    normalized = [
        img if isinstance(img, dict) else {"uri": img, "section": "", "document": ""}
        for img in images
    ]
    cols = st.columns(min(len(normalized), 2))
    for i, img in enumerate(normalized):
        img_bytes = _decode_image(img.get("uri", ""))
        if not img_bytes:
            continue
        with cols[i % len(cols)]:
            st.image(img_bytes, use_container_width=True)
            caption = img.get("section") or img.get("document") or ""
            if caption:
                st.markdown(
                    f'<div class="doc-card-caption"><span>{caption}</span>'
                    f'<span>Hình {i + 1}</span></div>',
                    unsafe_allow_html=True,
                )


# ────────────────────────── Main Chat Flow ──────────────────────────

new_question_to_process: Optional[str] = None

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h1>Tra cứu kỹ thuật Vhomenex</h1>
            <p>Hỏi về cách đấu nối dây, thông số kỹ thuật hoặc mã lỗi thiết bị trong kho tài liệu nội bộ.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg_idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🔷"):
            st.markdown(message["content"])
            _render_message_images(message.get("images", []))
            
            footer_cols = st.columns([5, 1, 1])
            with footer_cols[0]:
                if message.get("latency_ms"):
                    st.markdown(
                        f'<div class="response-meta">⚡ Hoàn thành trong {message["latency_ms"] / 1000:.1f}s</div>',
                        unsafe_allow_html=True,
                    )
            # Reads the verdict back off the message so reopening a past
            # conversation shows what was already rated, and so the buttons
            # visibly change state instead of looking like they did nothing.
            verdict = message.get("feedback")
            with footer_cols[1]:
                if st.button(
                    "👍",
                    key=f"like_{msg_idx}",
                    help="Thông tin chính xác",
                    type="primary" if verdict == "up" else "secondary",
                    disabled=verdict == "up",
                ):
                    _record_feedback(
                        st.session_state.conversation_id,
                        st.session_state.messages,
                        msg_idx,
                        "up",
                    )
                    st.rerun()
            with footer_cols[2]:
                if st.button(
                    "👎",
                    key=f"dislike_{msg_idx}",
                    help="Báo cáo sai lệch",
                    type="primary" if verdict == "down" else "secondary",
                    disabled=verdict == "down",
                ):
                    _record_feedback(
                        st.session_state.conversation_id,
                        st.session_state.messages,
                        msg_idx,
                        "down",
                    )
                    st.rerun()
            if verdict:
                st.caption(
                    "✓ Đã ghi nhận đánh giá của bạn."
                    if verdict == "up"
                    else "✓ Đã ghi nhận. Nội dung này sẽ được rà soát lại."
                )

if user_input := st.chat_input("Nhập mã lỗi hoặc sơ đồ cần tìm...", disabled=not backend_online):
    new_question_to_process = user_input

st.markdown(
    '<div class="disclaimer">Thông tin mang tính tham khảo nội bộ — luôn kiểm tra nguồn điện và sơ đồ đính kèm trước khi đấu nối.</div>',
    unsafe_allow_html=True,
)

if not backend_online:
    st.error("Không thể kết nối đến máy chủ nội bộ. Vui lòng kiểm tra dịch vụ backend.")

if new_question_to_process:
    question = new_question_to_process
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-8:-1]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    with st.chat_message("assistant", avatar="🔷"):
        try:
            result = _render_answer(question, selected_product_id, history)
            _render_message_images(result.get("images", []))
            st.markdown(
                f'<div class="response-meta">⚡ Hoàn thành trong {result["latency_ms"] / 1000:.1f}s</div>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "latency_ms": result["latency_ms"],
                    "images": result.get("images", []),
                }
            )
            _save_conversation(st.session_state.conversation_id, st.session_state.messages)
            st.rerun()
        except requests.exceptions.ConnectionError:
            st.error("Mất kết nối tới dịch vụ API.")
        except requests.exceptions.Timeout:
            st.error("Hệ thống mất quá nhiều thời gian để phản hồi. Vui lòng thử lại.")
        except Exception as error:
            st.error(f"Đã xảy ra lỗi: {error}")