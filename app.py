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


# ================================================================
# BATAS FREE TIER (PER JUNI 2026) — PER API KEY
# ┌──────────────────────┬──────┬────────┬──────────┬────────────┐
# │ Model                │ RPM  │ RPD    │ TPM      │ Prioritas  │
# ├──────────────────────┼──────┼────────┼──────────┼────────────┤
# │ gemini-2.5-flash-lite│  15  │ 1.000  │ 250.000  │ UTAMA ✅   │
# │ gemini-2.5-flash     │  10  │    20  │ 250.000  │ DARURAT ⚠️ │
# └──────────────────────┴──────┴────────┴──────────┴────────────┘
# CATATAN: Limit RPD di atas berlaku PER API KEY. Chatbot ini mendukung
#          hingga 7 API key sekaligus (total: 7.000 RPD Flash-Lite +
#          140 RPD Flash) dengan rotasi otomatis — begitu satu key
#          kehabisan kuota harian, sistem langsung pindah ke key
#          berikutnya tanpa mengganggu warga yang sedang bertanya.
#
# CARA SET API KEY (pilih salah satu):
#   1) 7 variabel terpisah di .env / Secrets:
#        GOOGLE_API_KEY_1=xxxx
#        GOOGLE_API_KEY_2=xxxx
#        ... sampai GOOGLE_API_KEY_7=xxxx
#   2) Satu variabel berisi daftar dipisah koma:
#        GOOGLE_API_KEYS=xxxx,yyyy,zzzz,...
#   3) Mode lama (1 key saja, tetap didukung):
#        GOOGLE_API_KEY=xxxx
#
# STRATEGI EFISIENSI FREE TIER YANG DITERAPKAN:
# 1. Response Cache      → Pertanyaan identik tidak memanggil API ulang
# 2. Token Budgeting     → Potong context + histori agar tidak mubazir TPM
# 3. Flash-Lite First    → Semua query → Flash-Lite; Flash hanya darurat
# 4. Smart Backoff       → Baca retryDelay dari Google, bukan exponential buta
# 5. Rotasi Multi API Key→ Key habis kuota → otomatis pindah ke key berikutnya
# 6. Daily Usage Tracker → Pantau sisa RPD tiap key (Flash-Lite & Flash)
# ================================================================



# ================================================================
# HYBRID RETRIEVER (pengganti EnsembleRetriever)
# Reciprocal Rank Fusion: FAISS 60% + BM25 40%
# ================================================================
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


# ================================================================
# CHUNKING v2 — selaras dengan generate_ragas_dataset.py
#
# Perbaikan dibanding versi lama (hasil evaluasi RAGAS / hit-rate):
# 1. BAB I (ketentuan umum) dipecah PER DEFINISI, bukan satu blok
#    besar — soal "apa yang dimaksud dengan X" jadi jauh lebih akurat
#    ke-retrieve.
# 2. Ayat panjang yang berisi daftar huruf (a, b, c, ...) TIDAK LAGI
#    dipecah per-huruf satuan (itu menghilangkan konteks ayat induk).
#    Sekarang split per ayat (1)(2)(3); jika masih >800 karakter baru
#    dipecah per grup 3 item huruf, dengan kalimat pembuka ayat tetap
#    disertakan di tiap sub-chunk.
# ================================================================
def _split_bab1_definisi(bab1_text: str, bab_header: str) -> list[str]:
    """Pecah blok BAB 1 menjadi chunk per item definisi."""
    parts = re.split(r'\n(?=\d{1,2}\.\s)', bab1_text)

    chunks = []
    pasal_header = ""
    pasal_match = re.search(r'(Pasal\s+\d+)', parts[0], re.IGNORECASE)
    if pasal_match:
        pasal_header = pasal_match.group(1)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        is_definisi = bool(re.match(r'^\d{1,2}\.\s', part))

        if is_definisi:
            header = f"[{bab_header} | {pasal_header} | Definisi]"
            chunks.append(f"{header}\n{part}")
        else:
            if len(part) > 20:
                header = f"[{bab_header} | {pasal_header}]"
                chunks.append(f"{header}\n{part}")

    return chunks


def _split_ayat_with_huruf_list(ayat_text: str) -> list[str]:
    """
    Pecah satu ayat yang sangat panjang dan berisi daftar berlabel huruf
    (a. ... b. ... c. ...) menjadi beberapa sub-chunk lebih kecil, TANPA
    kehilangan konteks ayat induknya (kalimat pembuka ayat ikut disertakan
    di tiap sub-chunk).
    """
    intro_match = re.split(r'\n(?=[a-z]\.\s)', ayat_text, maxsplit=1)
    if len(intro_match) < 2:
        return [ayat_text]

    intro, rest = intro_match[0].strip(), intro_match[1]
    items = re.split(r'\n(?=[a-z]\.\s)', rest)
    items = [i.strip() for i in items if i.strip()]

    GROUP_SIZE = 3
    sub_chunks = []
    for i in range(0, len(items), GROUP_SIZE):
        group = items[i:i + GROUP_SIZE]
        sub_chunks.append(intro + "\n" + "\n".join(group))

    return sub_chunks if sub_chunks else [ayat_text]


def build_chunks_from_text(raw_text: str) -> list[str]:
    """Chunking v2 dengan penanganan khusus untuk BAB 1 (definisi)."""
    pasal_splits = re.split(r'\n(?=Pasal\s+\d+)', raw_text, flags=re.IGNORECASE)
    chunks = []
    current_bab = "BAB I KETENTUAN UMUM"
    in_bab1 = True

    for part in pasal_splits:
        part_cleaned = part.strip()
        if not part_cleaned:
            continue

        bab_match = re.search(r'(BAB\s+(?:[IVXLC]+|\d+)[^\n]*)', part_cleaned, re.IGNORECASE)
        if bab_match:
            current_bab = bab_match.group(1).strip()
            if not re.search(r'BAB\s+(I|1)\b', current_bab, re.IGNORECASE):
                in_bab1 = False

        pasal_match = re.match(r'(Pasal\s+\d+)', part_cleaned, re.IGNORECASE)
        pasal_header = pasal_match.group(1) if pasal_match else ""

        # ── Khusus BAB 1: pecah per definisi ──────────────────────
        has_definisi = bool(re.search(r'\n\d{1,2}\.\s+\S', part_cleaned))
        if in_bab1 and has_definisi:
            definisi_chunks = _split_bab1_definisi(part_cleaned, current_bab)
            if definisi_chunks:
                chunks.extend(definisi_chunks)
                continue

        # ── Pasal biasa ──────────────────────────────────────────
        if len(part_cleaned) <= 600:
            header = f"[{current_bab} | {pasal_header}]" if pasal_header else f"[{current_bab}]"
            chunks.append(f"{header}\n{part_cleaned}")
        else:
            # Sub-split HANYA per ayat (1)(2)(3) — daftar huruf a,b,c
            # tetap menyatu dengan ayat induknya.
            ayat_splits = re.split(r'\n(?=\s*\(\d+\)\s)', part_cleaned)
            for ayat in ayat_splits:
                ayat_cleaned = ayat.strip()
                if len(ayat_cleaned) < 20:
                    continue
                header = f"[{current_bab} | {pasal_header}]" if pasal_header else f"[{current_bab}]"

                if len(ayat_cleaned) > 800:
                    sub_chunks = _split_ayat_with_huruf_list(ayat_cleaned)
                    for sub in sub_chunks:
                        chunks.append(f"{header}\n{sub}")
                else:
                    chunks.append(f"{header}\n{ayat_cleaned}")

    return [c for c in chunks if len(c.strip()) > 10]


# ================================================================
# LOAD RESOURCES (Embedding + FAISS + BM25 + EnsembleRetriever)
# ================================================================
@st.cache_resource
def get_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )

    file_path = "perdes_sampah.txt"
    chunks = []

    if not os.path.exists("faiss_index"):
        if not os.path.exists(file_path):
            return None, None, None
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = build_chunks_from_text(raw_text)
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
            chunks = build_chunks_from_text(raw_text)

    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    documents = [Document(page_content=c) for c in chunks]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RAG_CONFIG["top_k_retrieval"]

    faiss_retriever = vector_db.as_retriever(
        search_kwargs={"k": RAG_CONFIG["top_k_retrieval"]}
    )
    hybrid_retriever = HybridRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return embeddings, vector_db, hybrid_retriever


@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ================================================================
# RETRIEVAL + RERANKING v2 — selaras dengan generate_ragas_dataset.py
#
# Perbaikan dibanding versi lama:
# • rerank_threshold 0.0 (dulu tanpa filter) — buang kandidat yang
#   cuma "kebetulan" mirip secara leksikal (noise).
# • top_n_rerank 7 (dulu 3) — soal yang jawabannya tersebar di 2+
#   pasal butuh slot lebih banyak agar semuanya lolos ke context.
# • Deduplication — hapus chunk duplikat/near-duplicate.
# • Definition boost — chunk BAB 1 bertag "| Definisi]" dapat bonus
#   skor saat pertanyaan berpola "apa yang dimaksud dengan X",
#   supaya tidak kalah saing vs pasal lain yang cuma menyinggung
#   kata kunci yang sama secara sambil lalu.
# • Smart truncate — potong context di batas antar-chunk ("---"),
#   bukan di tengah kalimat.
# ================================================================
_DEFINITION_QUESTION_PATTERN = re.compile(
    r'apa\s+(yang\s+dimaksud|definisi|arti|pengertian)|'
    r'jelaskan\s+(apa\s+itu|pengertian)|'
    r'apa\s+itu\b',
    re.IGNORECASE,
)
_DEFINITION_CHUNK_PATTERN = re.compile(r'\|\s*Definisi\s*\]')


def _is_definition_question(question: str) -> bool:
    return bool(_DEFINITION_QUESTION_PATTERN.search(question))


def _apply_definition_boost(question: str, docs: list, scores) -> list:
    """Beri bonus skor ke chunk definisi BAB 1 saat pertanyaan berpola definisi."""
    BOOST = 4.0
    if not _is_definition_question(question):
        return list(scores)
    boosted = []
    for doc, score in zip(docs, scores):
        if _DEFINITION_CHUNK_PATTERN.search(doc.page_content.split("\n", 1)[0]):
            boosted.append(score + BOOST)
        else:
            boosted.append(score)
    return boosted


def _deduplicate_chunks(docs: list) -> list:
    """Hapus chunk duplikat atau near-duplicate (subset string)."""
    seen_content, result = [], []
    for doc in docs:
        content = doc.page_content.strip()
        is_dup = any(content in existing or existing in content for existing in seen_content)
        if not is_dup:
            seen_content.append(content)
            result.append(doc)
    return result


def _smart_truncate(context_str: str, max_chars: int) -> str:
    """Potong context pada batas chunk '---', bukan di tengah kalimat."""
    if len(context_str) <= max_chars:
        return context_str
    cutoff = context_str.rfind("\n\n---\n\n", 0, max_chars)
    if cutoff > 0:
        return context_str[:cutoff]
    cutoff = context_str.rfind("\n", 0, max_chars)
    if cutoff > 0:
        return context_str[:cutoff]
    return context_str[:max_chars]


# ================================================================
# AUTO-GLOSARIUM ISTILAH SAMPAH
#
# Masalah: warga sering tidak paham istilah kategori sampah (organik,
# anorganik, residu, spesifik/B3, dst.) yang disebut dalam jawaban,
# padahal definisinya ADA di BAB 1 (Ketentuan Umum) perdes — hanya saja
# tidak ikut ter-retrieve karena pertanyaan warga tidak memakai kata
# itu secara eksplisit (mis. "bagaimana cara memilah sampah?" tidak
# mengandung kata "organik").
#
# Solusi: setelah context utama didapat, pindai istilah kategori sampah
# yang MUNCUL di context/pertanyaan, cari definisi resminya lewat
# retrieval terpisah (query "apa yang dimaksud dengan <istilah>"), lalu
# sisipkan ke context sebagai blok "[Definisi istilah terkait]" — supaya
# LLM tetap menjawab HANYA berdasarkan konteks (tidak mengarang), tapi
# konteksnya sekarang sudah memuat penjelasan istilah tersebut.
# ================================================================
GLOSSARY_TERMS = [
    "sampah organik", "sampah anorganik", "sampah residu",
    "sampah spesifik", "sampah B3", "bank sampah",
    "TPS3R", "TPS", "TPA", "daur ulang", "pengomposan", "kompos",
]

MAX_GLOSSARY_EXTRA_CHARS = 1500


def _find_definition_chunk(term: str, hybrid_retriever, reranker: CrossEncoder) -> str | None:
    """Cari 1 chunk definisi terbaik untuk sebuah istilah, kalau ada di dokumen."""
    query = f"apa yang dimaksud dengan {term}"
    candidates = hybrid_retriever.invoke(query)
    if not candidates:
        return None
    candidates = _deduplicate_chunks(candidates)

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scores = _apply_definition_boost(query, candidates, scores)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_score, top_doc = ranked[0]

    # Hanya terima kalau memang chunk definisi & skornya lolos threshold —
    # supaya tidak menyisipkan chunk yang cuma kebetulan mirip.
    is_definisi_chunk = bool(_DEFINITION_CHUNK_PATTERN.search(top_doc.page_content.split("\n", 1)[0]))
    if is_definisi_chunk and top_score >= RAG_CONFIG["rerank_threshold"]:
        return top_doc.page_content
    return None


def augment_context_with_glossary(question: str, base_context: str, hybrid_retriever, reranker: CrossEncoder) -> str:
    """Deteksi istilah kategori sampah yang relevan lalu tambahkan definisinya ke context."""
    text_lower = (question + "\n" + base_context).lower()
    extra_parts, seen_terms, total_extra = [], set(), 0

    for term in GLOSSARY_TERMS:
        tl = term.lower()
        if tl in seen_terms or tl not in text_lower:
            continue
        seen_terms.add(tl)

        def_chunk = _find_definition_chunk(term, hybrid_retriever, reranker)
        if not def_chunk:
            continue
        # Jangan duplikat kalau definisinya kebetulan sudah ikut di context utama.
        if def_chunk in base_context or def_chunk in "\n\n---\n\n".join(extra_parts):
            continue
        if total_extra + len(def_chunk) > MAX_GLOSSARY_EXTRA_CHARS:
            break
        extra_parts.append(def_chunk)
        total_extra += len(def_chunk)

    if not extra_parts:
        return base_context

    glossary_block = "\n\n---\n\n".join(extra_parts)
    return f"{base_context}\n\n---\n\n[Definisi istilah terkait]\n{glossary_block}"


def retrieve_and_rerank(question: str, hybrid_retriever, reranker: CrossEncoder) -> tuple[list, str]:
    """Pipeline retrieval v2: hybrid search → dedup → rerank + boost definisi
    → filter threshold → smart truncate."""
    candidate_docs = hybrid_retriever.invoke(question)
    if not candidate_docs:
        return [], ""

    candidate_docs = _deduplicate_chunks(candidate_docs)

    pairs = [(question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)
    scores = _apply_definition_boost(question, candidate_docs, scores)

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

    # Sisipkan definisi istilah kategori sampah (organik/anorganik/residu/dst.)
    # yang disebut tapi belum ikut ter-retrieve, supaya jawaban ke warga
    # otomatis menjelaskan istilah tersebut.
    context_str = augment_context_with_glossary(question, context_str, hybrid_retriever, reranker)

    return contexts, context_str


# ================================================================
# EFISIENSI 1: RESPONSE CACHE
# Hash pertanyaan + konteks → simpan jawaban di session_state.
# Pertanyaan yang sama persis TIDAK memanggil API → hemat RPD & TPM.
# ================================================================
def get_cache_key(question: str, context: str) -> str:
    combined = f"{question.strip().lower()}||{context[:200]}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(cache_key: str) -> str | None:
    cache = st.session_state.get("response_cache", {})
    return cache.get(cache_key)

def set_cached_response(cache_key: str, response: str):
    if "response_cache" not in st.session_state:
        st.session_state["response_cache"] = {}
    st.session_state["response_cache"][cache_key] = response


# ================================================================
# EFISIENSI 2: TOKEN BUDGETING
# Potong context agar tidak melebihi ~1500 token (~6000 karakter).
# Potong juga histori percakapan: hanya 2 giliran terakhir (bukan semua).
# Ini mencegah pemborosan TPM yang tidak perlu.
# ================================================================
# RAG_CONFIG diselaraskan dengan CONFIG di generate_ragas_dataset.py,
# supaya kualitas retrieval chatbot ini konsisten dengan hasil yang
# sudah divalidasi lewat hit_rate_analysis.py / evaluasi RAGAS.
RAG_CONFIG = {
    "top_k_retrieval"  : 15,   # kandidat awal sebelum reranking
    "top_n_rerank"     : 7,    # chunk final yang dikirim ke LLM
    "rerank_threshold" : 0.0,  # buang kandidat skor CrossEncoder < 0.0
    "max_context_chars": 9000, # cukup untuk ~5-7 chunk penuh
}

MAX_HISTORY_TURNS = 2      # Hanya 2 giliran terakhir (user+assistant)

def build_history(messages: list) -> list:
    recent = messages[:-1][-(MAX_HISTORY_TURNS * 2):]
    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


# ================================================================
# EFISIENSI 3: FLASH-LITE FIRST
# Flash RPD hanya 20/hari → SEMUA query dikirim ke Flash-Lite terlebih
# dahulu. Flash hanya digunakan jika Flash-Lite benar-benar gagal/error.
# Tidak ada lagi routing berdasarkan "kompleksitas" — bedanya tidak
# sebanding dengan penghematan RPD Flash yang hanya 20/hari.
# ================================================================
# (Smart routing dihapus — Flash-Lite dipakai untuk semua query)


# ================================================================
# EFISIENSI 3: MULTI API KEY DENGAN ROTASI OTOMATIS
# Chatbot ini bisa memakai hingga 7 API key Gemini. Setiap query
# dicoba dengan key yang sedang aktif; begitu satu key kehabisan
# kuota harian (RPD), sistem otomatis pindah ke key berikutnya
# tanpa mengganggu pengalaman warga. Flash-Lite tetap dicoba lebih
# dulu di tiap key; Flash dipakai hanya jika seluruh key gagal di
# Flash-Lite.
# ================================================================
MODEL_LIMITS = {
    "lite" : {"name": "gemini-2.5-flash-lite", "rpd": 1000, "max_tokens": 600},
    "flash": {"name": "gemini-2.5-flash",      "rpd": 20,   "max_tokens": 1200},
}


def load_api_keys() -> list[str]:
    """
    Muat hingga 7 API key. Urutan pencarian:
    1. GOOGLE_API_KEY_1 .. GOOGLE_API_KEY_7 (format .env yang disarankan)
    2. GOOGLE_API_KEYS berisi daftar dipisah koma ("key1,key2,...")
    3. GOOGLE_API_KEY tunggal (mode lama, tetap didukung — 1 key saja)
    """
    keys = []
    for i in range(1, 8):
        env_name = f"GOOGLE_API_KEY_{i}"
        k = os.getenv(env_name)
        if not k and env_name in st.secrets:
            k = st.secrets.get(env_name)
        if k and k.strip():
            keys.append(k.strip())

    if not keys:
        raw = os.getenv("GOOGLE_API_KEYS")
        if not raw and "GOOGLE_API_KEYS" in st.secrets:
            raw = st.secrets.get("GOOGLE_API_KEYS")
        if raw:
            keys = [k.strip() for k in raw.split(",") if k.strip()]

    if not keys:
        single = os.getenv("GOOGLE_API_KEY")
        if not single and "GOOGLE_API_KEY" in st.secrets:
            single = st.secrets.get("GOOGLE_API_KEY")
        if single and single.strip():
            keys = [single.strip()]

    return keys


def _parse_retry_delay(err_str: str) -> int | None:
    """Ekstrak retryDelay dari pesan error Google 429."""
    import re as _re
    match = _re.search(r"retry in ([\d\.]+)s", err_str, _re.IGNORECASE)
    if match:
        return int(float(match.group(1))) + 3  # +3 detik buffer
    return None


def _classify_rate_limit(err_str: str) -> str:
    """
    Klasifikasikan error 429 Google:
    • "minute" → limit per-menit, cukup tunggu sebentar & coba key yang sama
    • "day"    → kuota harian key ini habis, langsung pindah ke key lain
    • "unknown"→ error 429/kuota tapi tidak jelas jenisnya → aman utk pindah key
    • "none"   → bukan error rate-limit sama sekali
    """
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
    """
    Coba panggil satu tier model (lite/flash) dengan berputar melalui
    semua API key yang tersedia, mulai dari start_idx. Key yang lokal
    tercatat sudah habis kuotanya untuk tier ini akan dilewati tanpa
    memanggil API sama sekali (hemat waktu). Return (jawaban, index_key
    yang berhasil dipakai). Melempar Exception jika SEMUA key gagal.
    """
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
                # Limit sesaat (per-menit) — tunggu lalu coba key YANG SAMA sekali lagi.
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
                # Kuota harian key ini habis (atau tidak jelas) — langsung ganti key lain.
                mark_key_exhausted(idx, model_type)
                last_err = e
                continue

            else:
                # Error di luar rate-limit (mis. salah key, jaringan) → lempar langsung.
                raise

    raise RuntimeError(
        f"Semua {n} API key kehabisan kuota / gagal untuk model {model_type}."
    ) from last_err


# ================================================================
# EFISIENSI 5: DAILY USAGE TRACKER — PER API KEY
# Pantau pemakaian Flash & Flash-Lite untuk MASING-MASING key,
# karena kuota RPD Google berlaku per API key, bukan gabungan.
# Reset otomatis saat hari berganti atau jumlah key berubah.
# ================================================================
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


# ================================================================
# GENERATE PDF RIWAYAT PERCAKAPAN
# Menghasilkan PDF rapi berformat laporan percakapan chatbot.
# Mendukung teks panjang, wrap otomatis, dan karakter Unicode.
# ================================================================
def generate_chat_pdf(messages: list) -> bytes:
    """
    Mengonversi riwayat percakapan menjadi PDF terformat.
    Mengembalikan bytes PDF siap untuk st.download_button.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Riwayat Percakapan Chatbot Desa Tieng",
        author="Chatbot Perdes Tieng",
    )

    # ── Styles ────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "ChatTitle",
        parent=styles["Title"],
        fontSize=16,
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    style_subtitle = ParagraphStyle(
        "ChatSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    style_meta = ParagraphStyle(
        "ChatMeta",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    style_label_user = ParagraphStyle(
        "LabelUser",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#1A5276"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=2,
    )
    style_label_bot = ParagraphStyle(
        "LabelBot",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#1E8449"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=2,
    )
    style_bubble_user = ParagraphStyle(
        "BubbleUser",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1A1A1A"),
        leading=14,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0,
    )
    style_bubble_bot = ParagraphStyle(
        "BubbleBot",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1A1A1A"),
        leading=14,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        rightIndent=0,
    )
    style_model_tag = ParagraphStyle(
        "ModelTag",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#999999"),
        spaceAfter=2,
    )

    # ── Konten PDF ────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("Chatbot Peraturan Desa Tieng", style_title))
    story.append(Paragraph("Informasi Pengelolaan Sampah &amp; Bank Sampah", style_subtitle))
    story.append(Paragraph(
        f"Riwayat Percakapan &mdash; Dicetak pada {datetime.now().strftime('%d %B %Y, %H:%M WIB')}",
        style_meta
    ))
    story.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.HexColor("#1F4E79"), spaceAfter=12
    ))

    # Hitung hanya pesan yang ditampilkan (user + assistant)
    chat_messages = [m for m in messages if m["role"] in ("user", "assistant")]
    if not chat_messages:
        story.append(Paragraph("Belum ada percakapan.", styles["Normal"]))
    else:
        for i, msg in enumerate(chat_messages, 1):
            role    = msg["role"]
            content = msg.get("content", "").strip()
            model   = msg.get("model_used", "")

            # Escape karakter HTML agar tidak rusak di ReportLab
            content_safe = (
                content
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                # Pertahankan baris baru sebagai <br/>
                .replace("\n", "<br/>")
            )

            if role == "user":
                story.append(Paragraph(f"Warga #{(i + 1) // 2}", style_label_user))
                # Kotak bubble user (biru muda)
                tbl = Table(
                    [[Paragraph(content_safe, style_bubble_user)]],
                    colWidths=[doc.width],
                )
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D6EAF8")),
                    ("ROUNDEDCORNERS", [6]),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
                ]))
                story.append(tbl)

            else:
                story.append(Paragraph("🤖 Asisten Desa", style_label_bot))
                # Kotak bubble asisten (hijau muda)
                tbl = Table(
                    [[Paragraph(content_safe, style_bubble_bot)]],
                    colWidths=[doc.width],
                )
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D5F5E3")),
                    ("ROUNDEDCORNERS", [6]),
                    ("TOPPADDING",    (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#A9DFBF")),
                ]))
                story.append(tbl)
                # Tag model kecil di bawah bubble
                if model:
                    story.append(Paragraph(f"Model: {model}", style_model_tag))

        story.append(Spacer(1, 16))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#CCCCCC"), spaceAfter=6
        ))
        story.append(Paragraph(
            f"Total percakapan: {len(chat_messages)} pesan &nbsp;|&nbsp; "
            f"Peraturan Desa Tieng No. 02 Tahun 2024",
            style_meta
        ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ================================================================
# PROMPT TEMPLATE
# ================================================================
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


# ================================================================
# UI CSS — tampilan ramah untuk warga umum
# ================================================================
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

            .example-chip button {
                width: 100%;
                border-radius: 20px !important;
                border: 1px solid #A9DFBF !important;
                background-color: #F0FBF4 !important;
                color: #145A32 !important;
                font-size: 0.85rem !important;
                padding: 6px 14px !important;
            }
            .example-chip button:hover {
                background-color: #D5F5E3 !important;
                border-color: #1E8449 !important;
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
            .quota-bar { font-size: 0.82em; }
        </style>
    """, unsafe_allow_html=True)


# ================================================================
# LOGIKA MENJAWAB SATU PERTANYAAN
# Dipakai bersama oleh chat_input maupun tombol contoh pertanyaan.
# ================================================================
def answer_question(prompt: str, hybrid_retriever, reranker, api_keys: list[str]):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        context_string = ""
        model_used = ""

        with st.spinner("Mencari jawaban di Peraturan Desa..."):
            # ── Retrieval + Reranking v2 (selaras generate_ragas_dataset.py) ──
            _, context_string = retrieve_and_rerank(prompt, hybrid_retriever, reranker)

            # EFISIENSI 1: Cek cache dulu sebelum panggil API
            cache_key = get_cache_key(prompt, context_string)
            cached = get_cached_response(cache_key)

        if cached:
            full_response = cached
            message_placeholder.markdown(full_response)
            increment_usage("cache")
            model_used = "Cache"
        else:
            langchain_history = build_history(st.session_state.messages)
            prompt_template = get_chat_prompt_template()
            payload = {
                "context": context_string,
                "chat_history": langchain_history,
                "question": prompt
            }

            start_idx = st.session_state.get("active_key_idx", 0)

            # Flash-Lite dicoba lebih dulu di seluruh key (rotasi otomatis);
            # Flash hanya dipakai jika SEMUA key gagal di Flash-Lite.
            try:
                with st.spinner("Menyusun jawaban..."):
                    full_response, used_idx = call_model_tier(
                        prompt_template, payload, api_keys, "lite", start_idx
                    )
                model_used = f"Flash-Lite (key #{used_idx + 1})"

            except Exception:
                try:
                    with st.spinner("Mencoba model cadangan..."):
                        full_response, used_idx = call_model_tier(
                            prompt_template, payload, api_keys, "flash", start_idx
                        )
                    model_used = f"Flash (key #{used_idx + 1})"
                except Exception:
                    full_response = (
                        "Mohon maaf, seluruh kuota API hari ini sudah habis di "
                        f"ke-{len(api_keys)} key yang tersedia. Silakan coba lagi "
                        "besok, atau hubungi perangkat Desa Tieng untuk bantuan langsung."
                    )
                    model_used = "Error"

            if full_response and "seluruh kuota API" not in full_response:
                set_cached_response(cache_key, full_response)

        message_placeholder.markdown(full_response)

        if st.session_state.get("show_context"):
            with st.expander("📄 Lihat pasal yang dirujuk AI", expanded=False):
                st.markdown(f"```\n{context_string}\n```")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "context_retrieved": context_string,
            "model_used": model_used
        })


def handle_prompt(prompt: str, hybrid_retriever, reranker, api_keys: list[str]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer_question(prompt, hybrid_retriever, reranker, api_keys)


CONTOH_PERTANYAAN = [
    "Apa itu bank sampah?",
    "Apa sanksi jika buang sampah sembarangan?",
    "Bagaimana cara memilah sampah rumah tangga?",
    "Siapa yang bertanggung jawab mengelola sampah desa?",
]


# ================================================================
# MAIN
# ================================================================
def main():
    st.set_page_config(
        page_title="Asisten Warga Desa Tieng",
        page_icon="♻️",
        initial_sidebar_state="expanded",
    )
    local_css()

    st.markdown("""
        <div class="app-hero">
            <h1>♻️ Asisten Warga Desa Tieng</h1>
            <p>Tanya apa saja seputar aturan pengelolaan sampah dan bank sampah desa.
            Jawaban diambil langsung dari Peraturan Desa Tieng No. 02 Tahun 2024.</p>
        </div>
    """, unsafe_allow_html=True)

    api_keys = load_api_keys()
    if not api_keys:
        st.error(
            "Layanan belum siap: API Key belum dikonfigurasi. "
            "Silakan hubungi admin."
        )
        return

    init_usage_tracker(len(api_keys))

    embeddings, vector_db, hybrid_retriever = get_resources()
    if vector_db is None:
        st.error("Layanan belum siap: dokumen peraturan desa belum tersedia. Silakan hubungi admin.")
        return

    reranker = get_reranker()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Sambutan + contoh pertanyaan (hanya saat belum ada obrolan) ────
    if not st.session_state.messages:
        st.markdown("**👋 Belum tahu mau tanya apa? Coba salah satu ini:**")
        cols = st.columns(2)
        for i, contoh in enumerate(CONTOH_PERTANYAAN):
            with cols[i % 2]:
                st.markdown('<div class="example-chip">', unsafe_allow_html=True)
                if st.button(contoh, key=f"contoh_{i}", use_container_width=True):
                    st.session_state["queued_prompt"] = contoh
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── Riwayat obrolan ─────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if (message["role"] == "assistant"
                    and message.get("context_retrieved")
                    and st.session_state.get("show_context")):
                with st.expander("📄 Lihat pasal yang dirujuk AI", expanded=False):
                    st.markdown(f"```\n{message['context_retrieved']}\n```")

    # ── Input baru: dari kotak chat ATAU tombol contoh pertanyaan ──────
    typed_prompt = st.chat_input("Tulis pertanyaan Anda di sini, mis. \"Apa itu bank sampah?\"")
    queued_prompt = st.session_state.pop("queued_prompt", None)
    prompt = typed_prompt or queued_prompt

    if prompt:
        handle_prompt(prompt, hybrid_retriever, reranker, api_keys)

    st.markdown(
        '<div class="disclaimer-box">ℹ️ Jawaban chatbot ini dihasilkan otomatis '
        'berdasarkan isi Peraturan Desa dan dapat memuat kekeliruan. Untuk keperluan '
        'resmi/hukum, mohon konfirmasi ke perangkat Desa Tieng.</div>',
        unsafe_allow_html=True,
    )

    # ── SIDEBAR ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ♻️ Menu")

        if st.button("🆕 Mulai Percakapan Baru", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            pdf_bytes = generate_chat_pdf(st.session_state.messages)
            st.download_button(
                label="📥 Simpan Percakapan (PDF)",
                data=pdf_bytes,
                file_name=f"riwayat_chatbot_tieng_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("---")

        with st.expander("❓ Tentang chatbot ini"):
            st.markdown("""
            Chatbot ini membantu warga Desa Tieng mencari informasi seputar
            **Peraturan Desa No. 02 Tahun 2024 tentang Pengelolaan Sampah**,
            termasuk aturan pemilahan sampah, sanksi, dan bank sampah.

            Setiap jawaban diusahakan menyertakan **rujukan pasal** agar
            mudah diverifikasi. Jika ada yang kurang jelas, silakan
            hubungi kantor Desa Tieng secara langsung.
            """)

        st.session_state["show_context"] = st.toggle(
            "Tampilkan pasal rujukan di setiap jawaban",
            value=st.session_state.get("show_context", False),
            help="Aktifkan untuk melihat potongan teks peraturan yang dipakai AI menyusun jawaban."
        )

        st.markdown("---")

        with st.expander("🛠️ Info teknis (untuk admin)"):
            key_states = st.session_state.get("key_states", [])
            cache_hits = st.session_state.get("usage_cache_hit", 0)
            n_keys = len(api_keys)

            total_lite_limit = MODEL_LIMITS["lite"]["rpd"] * n_keys
            total_flash_limit = MODEL_LIMITS["flash"]["rpd"] * n_keys
            total_lite_used = sum(s["lite_used"] for s in key_states)
            total_flash_used = sum(s["flash_used"] for s in key_states)

            st.markdown(f"**{n_keys} API key terpasang** — rotasi otomatis saat kuota habis.")
            st.markdown("**Total kuota gabungan hari ini**")
            st.markdown(f"""
            ⚡ Flash-Lite — {total_lite_used}/{total_lite_limit} RPD
            🚨 Flash — {total_flash_used}/{total_flash_limit} RPD
            💾 Cache hit — {cache_hits} (0 RPD terpakai)
            """)

            st.markdown("**Detail per API key**")
            active_idx = st.session_state.get("active_key_idx", 0)
            for i, s in enumerate(key_states):
                lite_habis = s["lite_exhausted"] or s["lite_used"] >= MODEL_LIMITS["lite"]["rpd"]
                flash_habis = s["flash_exhausted"] or s["flash_used"] >= MODEL_LIMITS["flash"]["rpd"]
                lite_dot = "🔴" if lite_habis else "🟢"
                flash_dot = "🔴" if flash_habis else "🟢"
                aktif = " 👈 *sedang dipakai*" if i == active_idx else ""
                st.markdown(
                    f"- **Key #{i + 1}**{aktif} — "
                    f"Lite {lite_dot} {s['lite_used']}/{MODEL_LIMITS['lite']['rpd']} · "
                    f"Flash {flash_dot} {s['flash_used']}/{MODEL_LIMITS['flash']['rpd']}"
                )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Reset cache"):
                    st.session_state["response_cache"] = {}
                    st.success("Cache jawaban direset.")
            with col_b:
                if st.button("🔓 Reset status key"):
                    for s in key_states:
                        s["lite_exhausted"] = False
                        s["flash_exhausted"] = False
                    st.success("Status 'habis' pada semua key direset.")

            st.markdown("---")
            st.markdown("**Konfigurasi RAG aktif** (selaras `generate_ragas_dataset.py`)")
            st.markdown(f"""
            - Chunking v2: BAB 1 per-definisi, ayat panjang per-grup huruf
            - top_k_retrieval: {RAG_CONFIG['top_k_retrieval']}
            - top_n_rerank: {RAG_CONFIG['top_n_rerank']}
            - rerank_threshold: {RAG_CONFIG['rerank_threshold']}
            - max_context_chars: {RAG_CONFIG['max_context_chars']}
            - Hybrid FAISS (60%) + BM25 (40%)
            - CrossEncoder reranker + boost definisi
            - Dedup chunk + smart truncate
            - Auto-glosarium istilah sampah (organik/anorganik/residu/dll.)
            - Flash-Lite first, Flash darurat, response cache
            - Rotasi otomatis {n_keys} API key saat kuota habis
            """)


if __name__ == "__main__":
    main()