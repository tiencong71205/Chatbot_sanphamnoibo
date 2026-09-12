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

import auth

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
LOGO_PATH = Path(__file__).parent / "assets" / "vconnex_logo.png"
HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", str(Path(__file__).parent / "chat_sessions")))
REQUEST_TIMEOUT = 180
TITLE_MAX_CHARS = 30   # ngắn cho vừa sidebar hẹp -- 46 ký tự bị cắt cụt giữa chữ

_SOURCE_FOOTER_RE = re.compile(r"\n{0,3}📚[^\n]*Nguồn tham khảo\s*:.*$", re.IGNORECASE | re.DOTALL)
_FOOTER_MARKER = "📚"


def strip_source_footer(text: str) -> str:
    return _SOURCE_FOOTER_RE.sub("", text or "").rstrip()


# ─────────────────────── Chat history persistence ───────────────────────

def _history_dir() -> Path:
    """Thư mục lưu hội thoại của ĐÚNG người dùng đang đăng nhập.

    Mỗi tài khoản có thư mục riêng (chat_sessions/<tên đăng nhập>/), nên
    không ai đọc được lịch sử của người khác — trước đây mọi hội thoại nằm
    chung một chỗ và ai mở web cũng thấy hết.

    Nếu chưa đăng nhập thì rơi về thư mục "_khach": chỉ xảy ra ở các đường
    chạy phụ, còn giao diện chính đã chặn không cho vào khi chưa đăng nhập.
    """
    username = st.session_state.get("auth_user", {}).get("username", "_khach")
    # Chặn ký tự có thể thoát khỏi thư mục (../) — tên đăng nhập đã được
    # chuẩn hoá về chữ thường khi tạo, nhưng vẫn lọc thêm cho chắc.
    safe = re.sub(r"[^a-z0-9_-]", "_", str(username).lower()) or "_khach"
    path = HISTORY_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


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
            # Ưu tiên tên người dùng tự đặt; chỉ khi chưa đặt mới lấy câu
            # hỏi đầu tiên. Nếu không, tên tự đặt sẽ bị ghi đè mỗi lần
            # người dùng hỏi thêm trong cùng đoạn chat.
            "title": data.get("custom_title") or data.get("title") or "Cuộc trò chuyện mới",
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
    path = _conversation_path(conversation_id)

    # Giữ lại tên người dùng tự đặt: hàm này ghi đè cả file, nên nếu không
    # đọc lại trước thì tên tự đặt sẽ biến mất ngay khi người dùng hỏi thêm
    # một câu trong cùng đoạn chat.
    custom_title = ""
    if path.exists():
        try:
            custom_title = json.loads(path.read_text(encoding="utf-8")).get(
                "custom_title", ""
            )
        except (OSError, json.JSONDecodeError):
            pass

    payload = {
        "title": _derive_title(messages),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    if custom_title:
        payload["custom_title"] = custom_title
    try:
        path.write_text(
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


def _rename_conversation(conversation_id: str, new_title: str) -> None:
    """Đặt tên do người dùng chọn cho một đoạn chat.

    Tên được ghi thẳng vào file hội thoại. `_list_conversations()` ưu tiên
    tên này nếu có, nếu không mới lấy câu hỏi đầu tiên làm tên — nên tên tự
    đặt không bị ghi đè khi người dùng hỏi thêm câu mới.
    """
    path = _conversation_path(conversation_id)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["custom_title"] = new_title[:TITLE_MAX_CHARS]
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError):
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
    comment: str = "",
) -> None:
    """Persist a 👍/👎 (and any written comment) on one answer, in two places
    for two different uses.

    1. Onto the message itself inside the conversation JSON, so reopening a
       past conversation shows which answers were already rated instead of
       silently resetting to unrated.
    2. Appended to feedback.jsonl with the question that produced the answer
       and the answer text — that pairing is the point: a bare thumbs-down
       count says nothing actionable, but "these are the questions people
       marked wrong" is directly usable for accuracy work.

    `comment` is where the real diagnostic value lives: a thumbs-down alone
    says an answer was wrong, but not HOW — missing a step, wrong product,
    outdated spec. An installer's own words ("thiếu bước đấu dây COM") point
    straight at the fix, so the comment is stored even when it arrives after
    the verdict was already recorded.

    Written append-only as JSON Lines so a crash mid-write can at worst lose
    the last line rather than corrupt the whole history, and so the file can
    grow without being rewritten each time.
    """
    if not (0 <= message_index < len(messages)):
        return

    messages[message_index]["feedback"] = verdict
    if comment:
        messages[message_index]["feedback_comment"] = comment
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
        "comment": comment,
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

        /* Nút hội thoại trong sidebar: chữ nhỏ, gọn, và cắt bằng dấu "…"
           ở cuối thay vì bị xén ngang giữa chữ như trước. Chỉ nhắm vào
           nút bên trong sidebar nên không ảnh hưởng nút ở khung chat. */
        /* Hai nút (tên đoạn chat + ⋮) nằm sát nhau thay vì cách một khoảng
           rộng như mặc định của Streamlit. */
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            gap: 0.15rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
            div[data-testid="column"] {
            padding: 0 !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            font-size: 0.8rem;
            padding: 0.3rem 0.6rem;
            min-height: 0;
            line-height: 1.35;
            text-align: left;
            justify-content: flex-start;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Nút "＋ Đoạn chat mới" vẫn căn giữa như bình thường */
        section[data-testid="stSidebar"] .stButton > button[kind="primaryFormSubmit"],
        section[data-testid="stSidebar"] > div > div > div > div:first-child .stButton > button {
            justify-content: center;
            text-align: center;
        }

        /* Nút "⋯" mở menu: gọn, canh giữa, không kéo dài như nút tiêu đề.
           Mặc định mờ đi, chỉ hiện rõ khi rê chuột vào hàng hội thoại —
           giống cách Claude/ChatGPT làm: danh sách trông sạch, nút xoá
           không đập vào mắt cho tới lúc thực sự cần.

           Dùng opacity chứ không dùng display:none — phần tử vẫn chiếm
           chỗ, nên hàng không bị giật/nhảy layout khi chuột đi qua. */
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
            button[data-testid="stPopoverButton"] {
            font-size: 1rem;
            padding: 0.3rem 0 !important;
            min-height: 0;
            height: 100%;              /* cao bằng nút tên bên cạnh */
            line-height: 1.35;
            justify-content: center;
            text-align: center;
            color: var(--text-muted);
            /* Nền và viền giống hệt nút tên đoạn chat bên cạnh, để hai nút
               trông như một khối liền mạch. `transparent` trước đây để lộ
               nền xám của sidebar, khiến nút ⋮ nổi lên khác màu. */
            background: #FFFFFF !important;
            border: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
            button[data-testid="stPopoverButton"]:hover {
            /* Giữ nền trắng cho khớp ô tên bên cạnh, chỉ đậm viền và chữ
               để biết đang trỏ đúng nút. */
            color: var(--ink);
            background: #FFFFFF !important;
            border-color: var(--text-muted) !important;
        }
        /* Streamlit tự thêm mũi tên chỉ xuống vào nút mở menu. Ở đây nút đã
           là biểu tượng ⋮ quen thuộc rồi, thêm mũi tên vừa thừa vừa làm nút
           rộng ra gấp đôi. */
        /* Ẩn mũi tên chỉ xuống mà Streamlit tự chèn vào nút mở menu.
           Nhắm theo aria-hidden="true" — xác định từ cấu trúc HTML thật:
               <button data-testid="stPopoverButton">
                   <div>...⋮...</div>              <- phần chữ
                   <div aria-hidden="true">▾</div>  <- mũi tên cần ẩn
               </button>
           KHÔNG nhắm theo class (st-emotion-cache-xxxxx): những chuỗi đó do
           Streamlit sinh tự động và đổi mỗi lần nâng cấp. aria-hidden là
           thuộc tính ngữ nghĩa, ổn định hơn nhiều.

           Các bản vá trước nhắm vào svg / class chứa "icon" nên hụt hoàn
           toàn — mũi tên không phải svg, cũng không có class nào tên "icon". */
        button[data-testid="stPopoverButton"] > div[aria-hidden="true"],
        button[data-testid="stPopoverButton"] div[aria-hidden="true"] {
            display: none !important;
        }

        /* Các mục trong menu ⋮ (Đổi tên / Xóa): nền trắng, viền mảnh, chữ
           và icon căn trái — theo đúng kiểu menu ngữ cảnh quen thuộc, thay
           vì nút bấm to căn giữa như mặc định của Streamlit. */
        div[data-testid="stPopoverBody"] .stButton > button {
            background: #FFFFFF !important;
            border: 1px solid var(--ink) !important;
            color: var(--ink) !important;
            justify-content: flex-start !important;
            text-align: left !important;
            font-size: 0.85rem !important;
            padding: 0.45rem 0.7rem !important;
            min-height: 0 !important;
        }
        div[data-testid="stPopoverBody"] .stButton > button:hover {
            background: var(--surface-soft) !important;
            border-color: var(--ink) !important;
        }
        /* Icon trong menu: mảnh, cùng màu chữ, không tô đậm như emoji */
        div[data-testid="stPopoverBody"] .stButton > button span[data-testid] {
            font-size: 1.05rem !important;
            color: var(--ink) !important;
            opacity: 0.85;
        }

        /* Sidebar rộng cố định, bỏ thanh kéo giãn: bề rộng thay đổi được
           khiến tên đoạn chat lúc đủ chỗ lúc bị cắt, và trên điện thoại rất
           dễ kéo nhầm khi định cuộn. */
        section[data-testid="stSidebar"] {
            width: 300px !important;
            min-width: 300px !important;
            max-width: 300px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"],
        section[data-testid="stSidebar"] [data-testid="stSidebarResizer"] {
            display: none !important;
        }

        /* Trên thiết bị cảm ứng không có "rê chuột" — luôn hiện nút, nếu
           không người dùng điện thoại sẽ không bao giờ xoá được hội thoại */
        @media (hover: none) {
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
            div[data-testid="stHorizontalBlock"] button[data-testid="stPopoverButton"] {
                opacity: 1;
            }
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

            /* iOS Safari TỰ PHÓNG TO cả trang khi chạm vào ô nhập có cỡ chữ
               dưới 16px — không tắt được bằng thẻ viewport trên iOS đời mới.
               Cách duy nhất còn hiệu lực là để cỡ chữ đúng 16px. */
            .stChatInput textarea,
            .stChatInput input,
            section[data-testid="stSidebar"] input,
            .stTextInput input,
            .stTextArea textarea {
                font-size: 16px !important;
            }

            /* Hàng hội thoại trong sidebar: Streamlit mặc định xếp DỌC các
               cột khi màn hình hẹp, khiến nút ⋮ rơi xuống hàng riêng bên
               dưới tên hội thoại.
               CHỈ chặn việc xếp dọc, KHÔNG đụng vào bề rộng từng cột — lần
               trước ép width cho first/last-child làm tỉ lệ cột lệch hẳn,
               tên hội thoại bị đẩy ra ngoài khung. Để Streamlit tự chia
               theo tỉ lệ đã khai báo là đủ. */
            section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _dang_nhap_thanh_cong(user: Dict[str, Any]) -> None:
    """Lưu trạng thái đăng nhập và ghi mã phiên vào địa chỉ trang.

    Mã phiên nằm trên URL nên tải lại trang (F5) vẫn giữ được đăng nhập —
    Streamlit không có cookie, bộ nhớ phiên bị xoá sạch mỗi lần tải lại.

    Đánh đổi cần biết: mã phiên hiện trên thanh địa chỉ, ai nhìn màn hình
    hoặc xem lịch sử trình duyệt đều thấy. Trong mạng nội bộ, đổi lấy việc
    không phải đăng nhập lại mỗi lần F5 là hợp lý; nếu sau này mở ra ngoài
    thì nên thay bằng cookie thật.
    """
    st.session_state.auth_user = user
    st.session_state.messages = []
    st.session_state.conversation_id = str(uuid.uuid4())
    try:
        st.query_params["phien"] = auth.create_session(user["username"])
    except Exception:  # noqa: BLE001 — không tạo được phiên vẫn phải vào được
        pass
    st.rerun()


def _khoi_phuc_phien() -> bool:
    """Đọc mã phiên trên URL, khôi phục đăng nhập nếu còn hiệu lực."""
    token = st.query_params.get("phien", "")
    if not token:
        return False
    user = auth.verify_session(token)
    if not user:
        # Mã hỏng hoặc hết hạn — dọn khỏi URL để không hiện mãi mã vô dụng.
        try:
            del st.query_params["phien"]
        except (KeyError, Exception):  # noqa: BLE001
            pass
        return False
    st.session_state.auth_user = user
    return True


def _render_login_screen() -> None:
    """Màn hình đăng nhập / đăng ký. Chặn ứng dụng cho tới khi đăng nhập xong."""
    col_left, col_mid, col_right = st.columns([1, 1.4, 1])
    with col_mid:
        if LOGO_PATH.exists():
            # Nhúng thẳng bằng HTML thay vì st.image: st.image không căn giữa
            # được, và cách cũ (lồng thêm 3 cột con để căn) làm logo bị ép hẹp
            # hơn bề rộng yêu cầu rồi méo chữ. Cách này kiểm soát chính xác cả
            # kích thước lẫn khoảng cách phía trên.
            try:
                logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
                st.markdown(
                    f'<div style="text-align:center;margin:6vh 0 1.2rem 0;">'
                    f'<img src="data:image/png;base64,{logo_b64}" '
                    f'style="width:240px;max-width:80%;height:auto;"></div>',
                    unsafe_allow_html=True,
                )
            except OSError:
                pass  # Thiếu logo không được làm hỏng màn hình đăng nhập
        st.markdown(
            '<div class="empty-state" style="padding:2vh 0 1rem 0;">'
            "<h1>Tra cứu kỹ thuật</h1></div>",
            unsafe_allow_html=True,
        )

        tab_dang_nhap, tab_dang_ky = st.tabs(["Đăng nhập", "Đăng ký"])

        with tab_dang_nhap:
            with st.form("dang_nhap", clear_on_submit=False):
                username = st.text_input("Tên đăng nhập", key="login_user")
                password = st.text_input("Mật khẩu", type="password", key="login_pass")
                submitted = st.form_submit_button(
                    "Đăng nhập", use_container_width=True, type="primary"
                )

            if submitted:
                user = auth.verify_user(username, password)
                if user:
                    # Xoá hội thoại đang mở của phiên trước để người vừa đăng
                    # nhập không thấy nội dung của người trước trên cùng máy.
                    _dang_nhap_thanh_cong(user)
                else:
                    # Thông báo chung, không nói rõ sai tên hay sai mật khẩu --
                    # nếu phân biệt, người dò mật khẩu biết được tài khoản nào có thật.
                    st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        with tab_dang_ky:
            with st.form("dang_ky", clear_on_submit=False):
                new_user = st.text_input(
                    "Tên đăng nhập", key="reg_user",
                    placeholder="viết liền, không dấu — vd: nguyenvanan",
                )
                new_name = st.text_input(
                    "Họ và tên", key="reg_name", placeholder="Nguyễn Văn An"
                )
                new_pass = st.text_input("Mật khẩu", type="password", key="reg_pass")
                new_pass2 = st.text_input(
                    "Nhập lại mật khẩu", type="password", key="reg_pass2"
                )
                registered = st.form_submit_button(
                    "Tạo tài khoản", use_container_width=True, type="primary"
                )

            if registered:
                if new_pass != new_pass2:
                    st.error("Hai lần nhập mật khẩu không khớp.")
                elif not re.fullmatch(r"[a-z0-9_]{3,20}", (new_user or "").strip().lower()):
                    # Giới hạn ký tự ngay từ đầu: tên đăng nhập được dùng làm
                    # tên thư mục lưu lịch sử chat, nên chỉ cho chữ thường,
                    # số và gạch dưới để không phát sinh tên thư mục lạ.
                    st.error(
                        "Tên đăng nhập chỉ gồm chữ thường không dấu, số và dấu "
                        "gạch dưới, dài 3–20 ký tự."
                    )
                else:
                    ok, msg = auth.create_user(new_user, new_pass, new_name)
                    if ok:
                        # Đăng nhập luôn thay vì bắt người dùng gõ lại đúng
                        # thông tin vừa nhập xong ở ngay phía trên.
                        user = auth.verify_user(new_user, new_pass)
                        if user:
                            _dang_nhap_thanh_cong(user)
                        else:
                            st.success(msg + " Chuyển sang tab Đăng nhập để vào hệ thống.")
                    else:
                        st.error(msg)


# Chặn cửa: chưa đăng nhập thì chỉ hiện màn hình đăng nhập, không render
# gì khác. Đặt TRƯỚC mọi thứ liên quan tới dữ liệu để lịch sử chat và danh
# sách sản phẩm không bị lộ ra khi chưa xác thực.
if "auth_user" not in st.session_state and not _khoi_phuc_phien():
    _render_login_screen()
    st.stop()


products = _fetch_products()
backend_online = _backend_is_online()

# ────────────────────────── Sidebar ──────────────────────────

with st.sidebar:
    if LOGO_PATH.exists():
        # Bề rộng cố định 180px: ảnh gốc chỉ 204×67px, để "giãn full" sẽ bị
        # kéo vượt kích thước thật và trông mờ/vỡ nét.
        st.image(str(LOGO_PATH), width=180)

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
    else:
        st.markdown(
            '<div class="sidebar-label">Lịch sử trò chuyện</div>',
            unsafe_allow_html=True,
        )

    # Vùng cuộn chiều cao cố định: lịch sử dài bao nhiêu cũng chỉ chiếm
    # đúng phần này, cuộn bên trong để xem tiếp. Trước đây danh sách đổ
    # thẳng ra sidebar nên chỉ cần chục hội thoại là Kho tài liệu và Phạm
    # vi tra cứu bị đẩy xuống tận đáy, phải cuộn cả sidebar mới thấy.
    #
    # Danh sách phẳng, KHÔNG chia nhóm theo ngày: trong một vùng cuộn hẹp
    # như thế này, các tiêu đề nhóm ("HÔM NAY", "7 NGÀY QUA"...) chiếm mất
    # phần lớn không gian vốn đã ít, đẩy chính các hội thoại ra ngoài tầm
    # nhìn. Danh sách đã sắp sẵn theo thứ tự mới nhất trước.
    with st.container(height=230):
        for conv in conversations:
            is_active = conv["id"] == st.session_state.conversation_id
            row = st.columns([6.5, 1], gap="small")
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
                # Menu "⋯" thay cho nút xoá trực tiếp: xoá là thao tác không
                # hoàn tác được, mà nút xoá đặt ngay cạnh nút mở hội thoại thì
                # rất dễ bấm nhầm — nhất là khi hai nút sát nhau trong sidebar
                # hẹp. Bọc trong menu bắt phải qua hai bước có chủ đích.
                with st.popover("⋮", use_container_width=True):
                    new_title = st.text_input(
                        "Đổi tên",
                        value=conv["title"],
                        key=f"rename_input_{conv['id']}",
                        label_visibility="collapsed",
                        placeholder="Tên đoạn chat...",
                    )
                    if st.button(
                        "Đổi tên",
                        icon=":material/edit:",
                        key=f"rename_{conv['id']}",
                        use_container_width=True,
                    ):
                        moi = new_title.strip()
                        if moi and moi != conv["title"]:
                            _rename_conversation(conv["id"], moi)
                            st.rerun()

                    st.divider()
                    if st.button(
                        "Xóa đoạn chat",
                        icon=":material/delete:",
                        key=f"del_{conv['id']}",
                        use_container_width=True,
                    ):
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

    # ── Tài khoản đang đăng nhập ──
    st.divider()
    current_user = st.session_state.auth_user
    st.markdown(
        f'<div class="sidebar-label">Tài khoản</div>'
        f'<div style="font-size:0.85rem;font-weight:600;color:var(--ink);">'
        f'{current_user["full_name"]}</div>'
        f'<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.5rem;">'
        f'@{current_user["username"]}'
        + (" · Quản trị" if current_user.get("is_admin") else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.popover("⚙️ Tài khoản", use_container_width=True):
        st.caption("Đổi mật khẩu")
        old_pw = st.text_input("Mật khẩu hiện tại", type="password", key="pw_old")
        new_pw = st.text_input("Mật khẩu mới", type="password", key="pw_new")
        if st.button("Cập nhật mật khẩu", use_container_width=True):
            ok, msg = auth.change_password(current_user["username"], old_pw, new_pw)
            (st.success if ok else st.error)(msg)

    if st.button("Đăng xuất", use_container_width=True):
        # Huỷ phiên đã lưu TRƯỚC khi xoá trạng thái: nếu chỉ xoá bộ nhớ
        # phiên, mã trên URL vẫn còn hiệu lực và người sau mở lại đúng địa
        # chỉ đó sẽ vào thẳng tài khoản này.
        try:
            auth.destroy_session(st.query_params.get("phien", ""))
            del st.query_params["phien"]
        except (KeyError, Exception):  # noqa: BLE001
            pass
        # Xoá sạch trạng thái phiên, không chỉ riêng auth_user: nếu chỉ xoá
        # mỗi thông tin đăng nhập thì hội thoại đang mở vẫn nằm trong bộ nhớ
        # và người đăng nhập tiếp theo trên cùng trình duyệt sẽ thấy nó.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


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

            # Ô góp ý bằng chữ. Chỉ mở sau khi đã bấm 👍/👎 -- hiện sẵn với
            # mọi câu trả lời sẽ làm khung chat rối và gần như không ai dùng;
            # đúng lúc vừa bấm 👎 mới là lúc người dùng đang có sẵn trong đầu
            # câu trả lời sai chỗ nào.
            existing_comment = message.get("feedback_comment", "")
            if verdict and not existing_comment:
                with st.expander("✍️ Góp ý thêm (không bắt buộc)", expanded=False):
                    comment_key = f"comment_{msg_idx}"
                    comment_text = st.text_area(
                        "Câu trả lời sai/thiếu chỗ nào? Mô tả giúp đội kỹ thuật sửa nhanh hơn:",
                        key=comment_key,
                        placeholder="VD: Thiếu bước đấu dây COM; hoặc: Trả lời nhầm sang sản phẩm khác...",
                        height=90,
                        label_visibility="visible",
                    )
                    if st.button("Gửi góp ý", key=f"send_comment_{msg_idx}"):
                        if comment_text.strip():
                            _record_feedback(
                                st.session_state.conversation_id,
                                st.session_state.messages,
                                msg_idx,
                                verdict,
                                comment_text.strip(),
                            )
                            st.rerun()
                        else:
                            st.warning("Bạn chưa nhập nội dung góp ý.")
            elif existing_comment:
                st.caption(f"💬 Góp ý của bạn: *{existing_comment}*")

if user_input := st.chat_input("Nhập mã lỗi hoặc sơ đồ cần tìm...", disabled=not backend_online):
    new_question_to_process = user_input

# Trên điện thoại, bàn phím vẫn bật sau khi gửi và che mất câu trả lời đang
# hiện dần. Streamlit không có cách tắt bàn phím, nên phải bỏ tiêu điểm khỏi
# ô nhập bằng JS — bàn phím tự đóng theo.
#
# Chỉ chạy trên thiết bị cảm ứng: trên máy tính, bỏ tiêu điểm sau mỗi lần
# gửi sẽ bắt người dùng bấm lại vào ô mỗi lần muốn hỏi tiếp — khó chịu hơn
# là giúp ích.
st.markdown(
    """
    <script>
    (function () {
        if (window.matchMedia && window.matchMedia('(hover: hover)').matches) {
            return;  // máy tính có chuột — không đụng vào
        }
        const doc = window.parent ? window.parent.document : document;
        const input = doc.querySelector('[data-testid="stChatInput"] textarea');
        if (input && doc.activeElement === input) {
            input.blur();
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

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