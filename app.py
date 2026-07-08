import streamlit as st
import os
import re
import time
import hashlib
import pandas as pd
from datetime import datetime
from io import BytesIO

# --- PDF GENERATION ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# --- LANGCHAIN IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# --- RERANKER (CrossEncoder) ---
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()


class HybridRetriever:
    def __init__(self, retrievers: list, weights: list, c: int = 60):
        self.retrievers = retrievers
        self.weights    = weights
        self.c          = c
    def invoke(self, query: str) -> list:
        all_results = [r.invoke(query) for r in self.retrievers]
        scores, doc_map = {}, {}
        for docs, weight in zip(all_results, self.weights):
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (self.c + rank + 1))
                doc_map[key] = doc
        return [doc_map[k] for k in sorted(scores, key=lambda k: scores[k], reverse=True)]


def split_large_pasal(pasal_chunk: str, max_len: int = 1200) -> list[str]:
    if len(pasal_chunk) <= max_len:
        return [pasal_chunk]
    lines = pasal_chunk.split("\n")
    header_lines = []
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("* "):
        header_lines.append(lines[i])
        i += 1
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[i:])
    if not body.strip():
        return [pasal_chunk]
    segments = re.split(r'\n(?=\* )', body)
    segments = [s.strip() for s in segments if s.strip()]
    return [f"{header}\n{seg}" for seg in segments]


def build_chunks_from_text(raw_text: str, pasal_split_threshold: int = 1200) -> list[str]:
    parts = re.split(r'\n(?=###\s+Pasal)', raw_text)
    chunks = []
    current_bab = "KETENTUAN UMUM"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        bab_match = re.search(r'\n##\s+(BAB\s+[^\n]+)', "\n" + part)
        next_bab = None
        if bab_match:
            part = part[:bab_match.start()].strip()
            next_bab = bab_match.group(1).strip()
        full_chunk = f"[{current_bab}]\n{part}"
        chunks.extend(split_large_pasal(full_chunk, max_len=pasal_split_threshold))
        if next_bab:
            current_bab = next_bab
    return [c for c in chunks if len(c.strip()) > 10]


@st.cache_resource
def get_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )
    file_path = "perdes_sampah_optimize.txt"
    chunks = []
    if not os.path.exists("faiss_index"):
        if not os.path.exists(file_path):
            return None, None, None
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = build_chunks_from_text(raw_text, pasal_split_threshold=RAG_CONFIG["pasal_split_threshold"])
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        with open("chunks_cache.txt", "w", encoding="utf-8") as f:
            f.write("\n<<<CHUNK_SEPARATOR>>>\n".join(chunks))
    else:
        if os.path.exists("chunks_cache.txt"):
            with open("chunks_cache.txt", "r", encoding="utf-8") as f:
                chunks = f.read().split("\n<<<CHUNK_SEPARATOR>>>\n")
        elif os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            chunks = build_chunks_from_text(raw_text, pasal_split_threshold=RAG_CONFIG["pasal_split_threshold"])
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    documents = [Document(page_content=c) for c in chunks]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RAG_CONFIG["top_k_retrieval"]
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": RAG_CONFIG["top_k_retrieval"]})
    hybrid_retriever = HybridRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4])
    return embeddings, vector_db, hybrid_retriever


@st.cache_resource
def get_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def _deduplicate_chunks(docs: list) -> list:
    seen_content, result = [], []
    for doc in docs:
        content = doc.page_content.strip()
        is_dup = any(content in existing or existing in content for existing in seen_content)
        if not is_dup:
            seen_content.append(content)
            result.append(doc)
    return result


def _smart_truncate(context_str: str, max_chars: int) -> str:
    if len(context_str) <= max_chars:
        return context_str
    cutoff = context_str.rfind("\n\n---\n\n", 0, max_chars)
    if cutoff > 0:
        return context_str[:cutoff]
    cutoff = context_str.rfind("\n", 0, max_chars)
    if cutoff > 0:
        return context_str[:cutoff]
    return context_str[:max_chars]


def retrieve_and_rerank(question: str, hybrid_retriever, reranker: CrossEncoder) -> tuple[list, str]:
    candidate_docs = hybrid_retriever.invoke(question)
    if not candidate_docs:
        return [], ""
    candidate_docs = _deduplicate_chunks(candidate_docs)
    pairs = [(question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)
    threshold = RAG_CONFIG["rerank_threshold"]
    scored_docs = [(s, d) for s, d in zip(scores, candidate_docs) if s >= threshold]
    if not scored_docs:
        ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in ranked[:3]]
    else:
        ranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in ranked[:RAG_CONFIG["top_n_rerank"]]]
    contexts = [doc.page_content for doc in top_docs]
    context_str = "\n\n---\n\n".join(contexts)
    context_str = _smart_truncate(context_str, RAG_CONFIG["max_context_chars"])
    return contexts, context_str


@st.cache_resource
def get_shared_cache() -> dict:
    return {}


def get_cache_key(question: str, context: str, has_history: bool) -> str:
    scope = "with_history" if has_history else "fresh"
    combined = f"{scope}||{question.strip().lower()}||{context[:200]}"
    return hashlib.md5(combined.encode()).hexdigest()


def get_cached_response(cache_key: str, has_history: bool) -> str | None:
    if has_history:
        cache = st.session_state.get("response_cache", {})
    else:
        cache = get_shared_cache()
    return cache.get(cache_key)


def set_cached_response(cache_key: str, response: str, has_history: bool):
    if has_history:
        if "response_cache" not in st.session_state:
            st.session_state["response_cache"] = {}
        st.session_state["response_cache"][cache_key] = response
    else:
        get_shared_cache()[cache_key] = response


RAG_CONFIG = {
    "top_k_retrieval"      : 10,
    "top_n_rerank"         : 4,
    "rerank_threshold"     : 0.0,
    "max_context_chars"    : 2500,
    "pasal_split_threshold": 1200,
}

MAX_HISTORY_TURNS = 2

def build_history(messages: list) -> list:
    recent = messages[:-1][-(MAX_HISTORY_TURNS * 2):]
    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


MODEL_LIMITS = {
    "lite" : {"name": "gemini-2.5-flash-lite", "rpd": 1000, "max_tokens": 600},
    "flash": {"name": "gemini-2.5-flash",      "rpd": 20,   "max_tokens": 1200},
}


def _get_secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def load_api_keys() -> list[str]:
    keys = []
    for i in range(1, 8):
        env_name = f"GOOGLE_API_KEY_{i}"
        k = os.getenv(env_name) or _get_secret(env_name)
        if k and k.strip():
            keys.append(k.strip())
    if not keys:
        raw = os.getenv("GOOGLE_API_KEYS") or _get_secret("GOOGLE_API_KEYS")
        if raw:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        single = os.getenv("GOOGLE_API_KEY") or _get_secret("GOOGLE_API_KEY")
        if single and single.strip():
            keys = [single.strip()]
    return keys


def _parse_retry_delay(err_str: str) -> int | None:
    import re as _re
    match = _re.search(r"retry in ([\d\.]+)s", err_str, _re.IGNORECASE)
    if match:
        return int(float(match.group(1))) + 3
    return None


def _classify_rate_limit(err_str: str) -> str:
    low = err_str.lower()
    if "perday" in low.replace(" ", "") or "requests per day" in low or "daily" in low:
        return "day"
    if "perminute" in low.replace(" ", "") or "requests per minute" in low:
        return "minute"
    if any(k in low for k in ["429", "quota", "resource_exhausted", "rate"]):
        return "unknown"
    return "none"


def call_model_tier(prompt_template, payload: dict, api_keys: list[str],
                     model_type: str, start_idx: int) -> tuple[str, int]:
    cfg = MODEL_LIMITS[model_type]
    n = len(api_keys)
    last_err: Exception | None = None

    for offset in range(n):
        idx = (start_idx + offset) % n
        if not is_key_available(idx, model_type):
            continue

        model = ChatGoogleGenerativeAI(
            model=cfg["name"],
            temperature=0.0,
            max_output_tokens=cfg["max_tokens"],
            google_api_key=api_keys[idx],
        )
        chain = prompt_template | model | StrOutputParser()

        def _stream_once() -> str:
            result = ""
            for chunk in chain.stream(payload):
                result += chunk
            return result

        try:
            result = _stream_once()
            increment_usage(model_type, idx)
            st.session_state["active_key_idx"] = idx
            return result, idx
        except Exception as e:
            err_str = str(e)
            kind = _classify_rate_limit(err_str)
            if kind == "minute":
                wait = _parse_retry_delay(err_str) or 15
                st.info(f"⏳ API key #{idx + 1} kena limit per-menit, tunggu {wait}s ...")
                time.sleep(wait)
                try:
                    result = _stream_once()
                    increment_usage(model_type, idx)
                    st.session_state["active_key_idx"] = idx
                    return result, idx
                except Exception as e2:
                    mark_key_exhausted(idx, model_type)
                    last_err = e2
                    continue
            elif kind in ("day", "unknown"):
                mark_key_exhausted(idx, model_type)
                last_err = e
                continue
            else:
                raise
    raise RuntimeError(f"Semua {n} API key kehabisan kuota / gagal untuk model {model_type}.") from last_err


def init_usage_tracker(num_keys: int):
    today = datetime.now().strftime("%Y-%m-%d")
    if (st.session_state.get("usage_date") != today
            or len(st.session_state.get("key_states", [])) != num_keys):
        st.session_state["usage_date"] = today
        st.session_state["key_states"] = [
            {"lite_used": 0, "flash_used": 0, "lite_exhausted": False, "flash_exhausted": False}
            for _ in range(num_keys)
        ]
        st.session_state["usage_cache_hit"] = 0
        st.session_state["active_key_idx"] = 0


def increment_usage(model_type: str, key_idx: int | None = None):
    if model_type == "cache":
        st.session_state["usage_cache_hit"] = st.session_state.get("usage_cache_hit", 0) + 1
        return
    states = st.session_state.get("key_states", [])
    if key_idx is not None and 0 <= key_idx < len(states):
        states[key_idx][f"{model_type}_used"] += 1


def mark_key_exhausted(key_idx: int, model_type: str):
    states = st.session_state.get("key_states", [])
    if 0 <= key_idx < len(states):
        states[key_idx][f"{model_type}_exhausted"] = True


def is_key_available(key_idx: int, model_type: str) -> bool:
    states = st.session_state.get("key_states", [])
    if key_idx >= len(states):
        return False
    state = states[key_idx]
    if state.get(f"{model_type}_exhausted"):
        return False
    limit = MODEL_LIMITS[model_type]["rpd"]
    return state.get(f"{model_type}_used", 0) < limit


def generate_chat_pdf(messages: list) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Riwayat Percakapan Chatbot Desa Tieng", author="Chatbot Perdes Tieng",
    )
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "ChatTitle", parent=styles["Title"], fontSize=16, textColor=colors.HexColor("#1F4E79"),
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    style_subtitle = ParagraphStyle("ChatSubtitle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=2, alignment=TA_CENTER)
    style_meta = ParagraphStyle("ChatMeta", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888888"), spaceAfter=16, alignment=TA_CENTER)
    style_label_user = ParagraphStyle("LabelUser", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#1A5276"), fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=2)
    style_label_bot = ParagraphStyle("LabelBot", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#1E8449"), fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=2)
    style_bubble_user = ParagraphStyle("BubbleUser", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#1A1A1A"), leading=14, alignment=TA_LEFT)
    style_bubble_bot = ParagraphStyle("BubbleBot", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#1A1A1A"), leading=14, alignment=TA_JUSTIFY)
    style_model_tag = ParagraphStyle("ModelTag", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#999999"), spaceAfter=2)

    story = []
    story.append(Paragraph("Chatbot Peraturan Desa Tieng", style_title))
    story.append(Paragraph("Informasi Pengelolaan Sampah &amp; Bank Sampah", style_subtitle))
    story.append(Paragraph(f"Riwayat Percakapan &mdash; Dicetak pada {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", style_meta))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1F4E79"), spaceAfter=12))

    chat_messages = [m for m in messages if m["role"] in ("user", "assistant")]
    if not chat_messages:
        story.append(Paragraph("Belum ada percakapan.", styles["Normal"]))
    else:
        for i, msg in enumerate(chat_messages, 1):
            role    = msg["role"]
            content = msg.get("content", "").strip()
            model   = msg.get("model_used", "")
            content_safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

            if role == "user":
                story.append(Paragraph(f"Warga #{(i + 1) // 2}", style_label_user))
                tbl = Table([[Paragraph(content_safe, style_bubble_user)]], colWidths=[doc.width])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D6EAF8")),
                    ("ROUNDEDCORNERS", [6]), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
                ]))
                story.append(tbl)
            else:
                story.append(Paragraph("🤖 Asisten Desa", style_label_bot))
                tbl = Table([[Paragraph(content_safe, style_bubble_bot)]], colWidths=[doc.width])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D5F5E3")),
                    ("ROUNDEDCORNERS", [6]), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#A9DFBF")),
                ]))
                story.append(tbl)
                if model:
                    story.append(Paragraph(f"Model: {model}", style_model_tag))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceAfter=6))
        story.append(Paragraph(f"Total percakapan: {len(chat_messages)} pesan &nbsp;|&nbsp; Peraturan Desa Tieng No. 02 Tahun 2024", style_meta))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def get_chat_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", """Anda adalah asisten resmi Desa Tieng yang membantu warga memahami \
Peraturan Desa Nomor 02 Tahun 2024 tentang Pengelolaan Sampah.

KONTEKS PERATURAN DESA (SUMBER JAWABAN ANDA):
{context}

═══════════════════════════════════════════
ATURAN WAJIB — PATUHI TANPA PENGECUALIAN:
═══════════════════════════════════════════
1. HANYA jawab berdasarkan KONTEKS di atas. JANGAN menambahkan informasi dari luar.
2. Untuk pertanyaan DEFINISI (mis. "Apa yang dimaksud dengan X"):
   • Cari langsung kalimat "X adalah ..." atau "X merupakan ..." di konteks.
   • Jika ditemukan, sampaikan definisi tersebut lalu sebutkan pasalnya.
   • Jika TIDAK ditemukan secara eksplisit, nyatakan: "Hal tersebut belum diatur \
secara eksplisit dalam Peraturan Desa Tieng Nomor 02 Tahun 2024."
3. Setiap pernyataan tentang aturan, larangan, kewajiban, atau sanksi WAJIB \
diakhiri referensi pasal. Contoh: "...sesuai Pasal 12 ayat (1)."
4. Jika jawaban Anda menyebut istilah kategori sampah (mis. sampah organik, \
anorganik, residu, spesifik/B3) atau istilah teknis lain yang definisinya \
tersedia di KONTEKS (termasuk pada bagian "[Definisi istilah terkait]" jika \
ada), jelaskan singkat artinya dalam bahasa sederhana disertai 1-2 contoh \
sehari-hari (mis. organik = sisa makanan/dedaunan, anorganik = plastik/kaleng/ \
botol, residu = sampah yang tidak bisa didaur ulang/dikompos seperti puntung \
rokok atau popok). JANGAN berasumsi warga awam sudah paham istilah tersebut.
5. Jawaban ringkas namun lengkap, bahasa ramah dan mudah dipahami warga awam. \
Boleh memakai poin-poin singkat jika perlu menjelaskan lebih dari satu istilah \
agar mudah dibaca. Selesaikan kalimat terakhir hingga tanda titik.
6. JANGAN mengarang nomor pasal yang tidak ada dalam konteks."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def local_css():
    st.markdown("""
        <style>
            [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stFooter"] {display: none !important;}
            .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; max-width: 800px !important; }
            .stChatMessage { border-radius: 12px; }

            .app-hero {
                background: linear-gradient(135deg, #1E8449 0%, #229954 100%);
                padding: 22px 24px;
                border-radius: 16px;
                color: white;
                margin-bottom: 6px;
            }
            .app-hero h1 { font-size: 1.5rem; margin: 0 0 4px 0; }
            .app-hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; }

            .st-key-top_actions[data-testid="stVerticalBlock"],
            .st-key-top_actions div[data-testid="stVerticalBlock"],
            div[data-testid="stVerticalBlock"].st-key-top_actions,
            html body div[class*="st-key-top_actions"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                justify-content: flex-end !important;
                align-items: center !important;
                gap: 0.5rem !important;
                width: 100% !important;
            }
            .st-key-top_actions div[data-testid="stElementContainer"],
            .st-key-top_actions div.element-container {
                flex: 0 0 auto !important;
                width: auto !important;
                min-width: 0 !important;
            }
            .st-key-top_actions button,
            .st-key-top_actions div[data-testid="stPopover"] button {
                border-radius: 12px !important;
                height: 3.1rem !important;
                width: 3.2rem !important;
                min-width: 3.2rem !important;
                padding: 0 !important;
                font-size: 1.1rem !important;
                white-space: nowrap !important;
            }

            .st-key-example_questions div[data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap !important;
                gap: 0.6rem !important;
            }
            .st-key-example_questions div[data-testid="column"] {
                min-width: 0 !important;
            }
            .example-chip button {
                width: 100%;
                border-radius: 20px !important;
                border: 1px solid #A9DFBF !important;
                background-color: #F0FBF4 !important;
                color: #145A32 !important;
                font-size: 0.85rem !important;
                padding: 6px 14px !important;
                white-space: normal !important;
                word-break: break-word !important;
            }
            .example-chip button:hover {
                background-color: #D5F5E3 !important;
                border-color: #1E8449 !important;
            }

            @media (max-width: 640px) {
                .st-key-example_questions div[data-testid="stHorizontalBlock"] {
                    flex-direction: column !important;
                    flex-wrap: wrap !important;
                }
                .st-key-example_questions div[data-testid="column"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                }
            }

            .disclaimer-box {
                background-color: #FEF9E7;
                border: 1px solid #F9E79F;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 0.82rem;
                color: #7D6608;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)


def execute_llm_processing(prompt: str, hybrid_retriever, reranker, api_keys: list[str]):
    """Memproses jawaban LLM di latar belakang dan memasukkannya langsung ke session state."""
    _, context_string = retrieve_and_rerank(prompt, hybrid_retriever, reranker)
    langchain_history = build_history(st.session_state.messages)
    has_history = bool(langchain_history)
    
    cache_key = get_cache_key(prompt, context_string, has_history)
    cached = get_cached_response(cache_key, has_history)

    if cached:
        full_response = cached
        increment_usage("cache")
        model_used = "Cache"
    else:
        prompt_template = get_chat_prompt_template()
        payload = {
            "context": context_string,
            "chat_history": langchain_history,
            "question": prompt
        }
        start_idx = st.session_state.get("active_key_idx", 0)

        try:
            full_response, used_idx = call_model_tier(prompt_template, payload, api_keys, "lite", start_idx)
            model_used = f"Flash-Lite (key #{used_idx + 1})"
        except Exception:
            try:
                full_response, used_idx = call_model_tier(prompt_template, payload, api_keys, "flash", start_idx)
                model_used = f"Flash (key #{used_idx + 1})"
            except Exception:
                full_response = (
                    "Mohon maaf, seluruh kuota API hari ini sudah habis. Silakan coba lagi "
                    "besok, atau hubungi perangkat Desa Tieng untuk bantuan langsung."
                )
                model_used = "Error"

        if full_response and "seluruh kuota API" not in full_response:
            set_cached_response(cache_key, full_response, has_history)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "context_retrieved": context_string,
        "model_used": model_used
    })


CONTOH_PERTANYAAN = [
    "Apa itu bank sampah?",
    "Apa sanksi jika buang sampah sembarangan?",
    "Bagaimana cara memilah sampah rumah tangga?",
    "Siapa yang bertanggung jawab mengelola sampah desa?",
]


def main():
    st.set_page_config(
        page_title="Asisten Warga Desa Tieng",
        page_icon="♻️",
        initial_sidebar_state="collapsed",
    )
    local_css()

    api_keys = load_api_keys()
    if not api_keys:
        st.error("Layanan belum siap: API Key belum dikonfigurasi. Silakan hubungi admin.")
        return

    init_usage_tracker(len(api_keys))

    embeddings, vector_db, hybrid_retriever = get_resources()
    if vector_db is None:
        st.error("Layanan belum siap: dokumen peraturan desa belum tersedia. Silakan hubungi admin.")
        return

    reranker = get_reranker()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── 1. intercept input di paling atas ──
    typed_prompt = st.chat_input("Tulis pertanyaan Anda di sini, mis. \"Apa itu bank sampah?\"")
    queued_prompt = st.session_state.pop("queued_prompt", None)
    prompt = typed_prompt or queued_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Mencari jawaban di Peraturan Desa..."):
            execute_llm_processing(prompt, hybrid_retriever, reranker, api_keys)
        st.rerun()

    # ── 2. render elemen top UI bar ──
    with st.container(key="top_actions"):
        if st.button("🆕", help="Mulai percakapan baru", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.session_state.messages:
            pdf_bytes = generate_chat_pdf(st.session_state.messages)
            st.download_button(
                "📥", data=pdf_bytes,
                file_name=f"riwayat_chatbot_tieng_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", help="Unduh percakapan (PDF)", use_container_width=True,
            )
        else:
            st.button("📥", help="Belum ada percakapan untuk diunduh", disabled=True, use_container_width=True)
        
        about_container = st.popover("❓", use_container_width=True) if hasattr(st, "popover") else st.expander("❓ Tentang", expanded=False)
        with about_container:
            st.markdown("**Tentang chatbot ini**\n\nChatbot ini membantu warga Desa Tieng mencari informasi seputar **Peraturan Desa No. 02 Tahun 2024 tentang Pengelolaan Sampah**.")

    st.markdown("""
        <div class="app-hero">
            <h1>♻️ Asisten Warga Desa Tieng</h1>
            <p>Tanya apa saja seputar aturan pengelolaan sampah dan bank sampah desa.
            Jawaban diambil langsung dari Peraturan Desa Tieng No. 02 Tahun 2024.</p>
        </div>
    """, unsafe_allow_html=True)

    # ── 3. render contoh pertanyaan jika belum ada obrolan ──
    if not st.session_state.messages:
        st.markdown("**👋 Belum tahu mau tanya apa? Coba salah satu ini:**")
        with st.container(key="example_questions"):
            cols = st.columns(2)
            for i, contoh in enumerate(CONTOH_PERTANYAAN):
                with cols[i % 2]:
                    st.markdown('<div class="example-chip">', unsafe_allow_html=True)
                    if st.button(contoh, key=f"contoh_{i}", use_container_width=True):
                        st.session_state["queued_prompt"] = contoh
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── 4. render riwayat obrolan secara kronologis ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("context_retrieved"):
                with st.expander("📄 Lihat pasal yang dirujuk AI", expanded=False):
                    st.markdown(f"```\n{message['context_retrieved']}\n```")

    # ── 5. render alert disclaimer di bagian paling bawah ──
    st.markdown(
        '<div class="disclaimer-box">ℹ️ Jawaban chatbot ini dihasilkan otomatis '
        'berdasarkan isi Peraturan Desa dan dapat memuat kekeliruan. Untuk keperluan '
        'resmi/hukum, mohon konfirmasi ke perangkat Desa Tieng.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()