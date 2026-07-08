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
# CHUNKING v4.1 — selaras dengan generate_ragas_dataset_v4.py
#
# Diganti total dari chunking v2 (split "Pasal\s+\d+" + BAB1 per
# definisi + ayat per grup huruf) ke chunking berbasis Markdown
# header ("### Pasal") yang jauh lebih ringan, plus auto-split untuk
# Pasal yang kepanjangan:
# 1. File sumber sekarang versi Markdown (perdes_sampah_optimize.txt).
# 2. Split utama hanya di baris "### Pasal N (...)" — super ringan,
#    tidak perlu parsing ayat/huruf manual.
# 3. Fix label BAB: header "## BAB ..." yang nyangkut di ekor Pasal
#    terakhir suatu BAB dipotong dulu sebelum dipakai sebagai label
#    BAB PASAL BERIKUTNYA (bukan pasal saat ini).
# 4. split_large_pasal(): chunk > pasal_split_threshold (mis. Pasal
#    47 yang berisi banyak sub-topik) dipecah lagi otomatis per bullet
#    level-0 ("* ..."), dengan header [BAB]/Pasal tetap disertakan di
#    tiap pecahan supaya konteksnya tidak hilang.
# ================================================================
def split_large_pasal(pasal_chunk: str, max_len: int = 1200) -> list[str]:
    """Pecah 1 chunk Pasal yang kepanjangan menjadi beberapa sub-chunk
    yang lebih kecil dan presisi, berdasarkan bullet level-0 ("* ...").
    Header "[BAB ...]" dan "### Pasal N (...)" disalin ulang ke tiap
    pecahan supaya konteksnya tetap ada meski chunk-nya sudah kecil.
    Chunk yang masih di bawah `max_len` dibiarkan apa adanya."""
    if len(pasal_chunk) <= max_len:
        return [pasal_chunk]

    lines = pasal_chunk.split("\n")

    # Ambil baris header di awal chunk: "[BAB ...]" dan "### Pasal N (...)"
    header_lines = []
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("* "):
        header_lines.append(lines[i])
        i += 1
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[i:])

    if not body.strip():
        # Tidak ada bullet level-0 yang bisa dipakai sebagai titik pecah
        return [pasal_chunk]

    # Pecah body berdasarkan bullet level-0 baru; sub-bullet ("  - ...")
    # otomatis ikut ke segmen induknya
    segments = re.split(r'\n(?=\* )', body)
    segments = [s.strip() for s in segments if s.strip()]

    return [f"{header}\n{seg}" for seg in segments]


def build_chunks_from_text(raw_text: str, pasal_split_threshold: int = 1200) -> list[str]:
    """Chunking berbasis Markdown Header, dengan auto-split untuk Pasal kepanjangan."""
    # Pecah berdasarkan pola "### Pasal"
    parts = re.split(r'\n(?=###\s+Pasal)', raw_text)

    chunks = []
    current_bab = "KETENTUAN UMUM"

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Fix label BAB: karena split hanya terjadi di "### Pasal" (bukan
        # di "## BAB"), header BAB berikutnya sering ikut nyangkut di EKOR
        # teks Pasal terakhir suatu BAB. Potong dulu baris "## BAB ..." dari
        # isi pasal saat ini, lalu simpan sebagai current_bab untuk PASAL
        # BERIKUTNYA saja.
        bab_match = re.search(r'\n##\s+(BAB\s+[^\n]+)', "\n" + part)
        next_bab = None
        if bab_match:
            part = part[:bab_match.start()].strip()
            next_bab = bab_match.group(1).strip()

        full_chunk = f"[{current_bab}]\n{part}"
        # Pecah lagi kalau chunk-nya kepanjangan (Pasal 47, Pasal 1, dll)
        chunks.extend(split_large_pasal(full_chunk, max_len=pasal_split_threshold))

        if next_bab:
            current_bab = next_bab

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

    # v4: file sumber diganti ke versi Markdown (optimize) — selaras
    # dengan generate_ragas_dataset_v4.py. Chunking lama tidak kompatibel
    # dengan file ini, jadi faiss_index/ & chunks_cache.txt lama harus
    # dihapus dulu supaya dibangun ulang dengan chunking baru.
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
    # v4.1: ganti dari cross-encoder/ms-marco-MiniLM-L-6-v2 (khusus B.Inggris)
    # ke BAAI/bge-reranker-v2-m3 (multibahasa, senasab dgn embedding bge-m3),
    # supaya skor relevansi lebih terkalibrasi untuk teks Bahasa Indonesia.
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


# ================================================================
# RETRIEVAL + RERANKING v4.1 — selaras dengan generate_ragas_dataset_v4.py
#
# Catatan migrasi dari v2: fitur "definition boost" dan glosarium
# otomatis (berbasis tag chunk "| Definisi]") DIHAPUS pada versi ini,
# karena chunking v4 (berbasis Markdown "### Pasal" + split_large_pasal)
# tidak lagi menghasilkan tag "| Definisi]" tsb — fitur itu jadi kode
# mati (tidak pernah match) sekaligus memboroskan panggilan retrieval
# & rerank tambahan per istilah glosarium. Pipeline sekarang murni:
# hybrid search → dedup → rerank → filter threshold → top_n →
# smart truncate, persis seperti retrieve_and_rerank() di v4.
# ================================================================
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


def retrieve_and_rerank(question: str, hybrid_retriever, reranker: CrossEncoder) -> tuple[list, str]:
    """Pipeline retrieval v4.1: hybrid search → dedup → rerank → filter
    threshold → top_n → smart truncate (identik dengan retrieve_and_rerank()
    di generate_ragas_dataset_v4.py, minus query expansion — lihat catatan
    di bawah RAG_CONFIG soal kenapa query expansion tidak dipakai di sini)."""
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


# ================================================================
# EFISIENSI 1: RESPONSE CACHE — DUA TINGKAT
#
# Masalah cache lama: disimpan di st.session_state (per-sesi browser),
# jadi tidak pernah "nular" ke warga lain meski pertanyaannya identik —
# potensi hemat terbesar (pertanyaan umum yang ditanya BANYAK warga
# berbeda) tidak pernah kepakai.
#
# Tapi cache tidak bisa asal di-share, karena jawaban LLM juga
# dipengaruhi `chat_history` (2 giliran terakhir obrolan) yang TIDAK
# ikut jadi bagian cache key. Kalau di-share apa adanya, warga lain
# bisa kebagian jawaban yang sebenarnya "diracik" mengikuti alur
# obrolan orang lain.
#
# Solusi — pisahkan berdasarkan ADA/TIDAKNYA riwayat obrolan:
# • Pertanyaan giliran PERTAMA (chat_history kosong) → jawabannya murni
#   berdasar context pasal, tidak dipengaruhi obrolan siapa pun → AMAN
#   di-share ke semua warga lewat SHARED_CACHE (st.cache_resource,
#   hidup selama proses Streamlit berjalan, dipakai bersama semua sesi).
# • Pertanyaan LANJUTAN (chat_history ada isinya) → tetap per-sesi
#   seperti sebelumnya (st.session_state), karena jawabannya memang
#   khusus untuk alur obrolan orang itu.
# ================================================================
@st.cache_resource
def get_shared_cache() -> dict:
    """Dict jawaban BERSAMA lintas sesi/warga — HANYA untuk pertanyaan
    tanpa riwayat obrolan. Dibuat sekali lewat st.cache_resource sehingga
    objek dict yang sama dipakai oleh semua pengguna aplikasi ini selama
    proses Streamlit-nya tidak restart."""
    return {}


def get_cache_key(question: str, context: str, has_history: bool) -> str:
    # Prefix "fresh"/"with_history" memastikan 2 skenario ini TIDAK
    # pernah dianggap sama, meski teks pertanyaan & context-nya identik.
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


# ================================================================
# EFISIENSI 2: TOKEN BUDGETING
# Potong context agar tidak melebihi ~1500 token (~6000 karakter).
# Potong juga histori percakapan: hanya 2 giliran terakhir (bukan semua).
# Ini mencegah pemborosan TPM yang tidak perlu.
# ================================================================
# RAG_CONFIG diselaraskan dengan CONFIG di generate_ragas_dataset_v4.py,
# supaya kualitas retrieval chatbot ini konsisten dengan hasil yang
# sudah divalidasi lewat evaluasi RAGAS v4 (chunking Markdown + auto-split
# pasal panjang + reranker multibahasa bge-reranker-v2-m3).
#
# Catatan: "use_query_expansion" di v4 TIDAK diikutsertakan di sini secara
# sengaja — fitur itu menambah 1 panggilan LLM per pertanyaan warga, yang
# bertentangan dengan strategi efisiensi kuota free-tier chatbot ini
# (lihat komentar EFISIENSI di atas). v4 dipakai untuk generate dataset
# evaluasi offline sehingga biaya token tambahan itu tidak masalah;
# di chatbot interaktif ini, tiap panggilan LLM ekstra mengurangi RPD
# yang tersedia untuk warga.
RAG_CONFIG = {
    "top_k_retrieval"      : 10,    # kandidat awal sebelum reranking (v4: turun dari 15)
    "top_n_rerank"         : 4,     # chunk final yang dikirim ke LLM (v4: turun dari 7)
    "rerank_threshold"     : 0.0,   # buang kandidat skor CrossEncoder < 0.0
    "max_context_chars"    : 2500,  # v4: turun drastis dari 9000 (chunk kini lebih kecil & presisi)
    "pasal_split_threshold": 1200,  # chunk > 1200 char dipecah otomatis per bullet level-0
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


def _get_secret(name: str) -> str | None:
    """
    Ambil satu nilai dari st.secrets dengan aman.
    Jika file .streamlit/secrets.toml tidak ada sama sekali (mis. saat
    dijalankan lokal tanpa Streamlit Cloud), st.secrets akan melempar
    StreamlitSecretNotFoundError bahkan hanya untuk mengecek keberadaan
    key ("in st.secrets"). Fungsi ini menangkap kasus itu dan cukup
    mengembalikan None, supaya aplikasi tetap bisa jalan dengan
    environment variable / .env saja.
    """
    try:
        return st.secrets.get(name)
    except Exception:
        return None


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

            /* ── Baris tombol atas (New chat / Unduh / Tentang) ───────────
               Tidak lagi memakai st.columns() (lihat komentar di Python),
               jadi di sini kita paksa flex-row langsung pada blok vertikal
               bawaan Streamlit yang membungkus ke-3 tombol tsb. Pendekatan
               ini menghindari CSS bawaan Streamlit yang khusus menumpuk
               elemen [data-testid="column"] di layar sempit — karena kita
               sudah sama sekali tidak memakai elemen "column". */
            .st-key-top_actions[data-testid="stVerticalBlock"],
            .st-key-top_actions div[data-testid="stVerticalBlock"],
            div[data-testid="stVerticalBlock"].st-key-top_actions,
            html body div[class*="st-key-top_actions"],
            .st-key-top_actions.st-key-top_actions div[data-testid="stVerticalBlock"],
            div[data-testid="stVerticalBlock"].st-key-top_actions.st-key-top_actions {
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
            /* Tombol ikon di pojok atas (pengganti sidebar) — lebar tetap
               supaya tidak melebar mengikuti kolom Streamlit. */
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

            /* ── Baris tombol contoh pertanyaan (di bawah app-hero) ───────
               Desktop: 2 kolom berdampingan (default st.columns(2)).
               Mobile: dipaksa 1 kolom/bertumpuk lewat media query di
               bawah, supaya teks pertanyaan yang panjang tidak memaksa
               tombol melebar dan bikin layar bisa discroll ke kanan. */
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

            /* ══════════════════════ MOBILE (≤640px) ══════════════════════ */
            @media (max-width: 640px) {
                /* Tombol atas TETAP di kanan (sama seperti desktop) — tidak
                   ada perubahan layout khusus untuk .st-key-top_actions di sini. */

                /* Tombol contoh pertanyaan: 1 kolom, bertumpuk ke bawah. */
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

            # Hitung riwayat obrolan LEBIH DULU — dipakai untuk menentukan
            # apakah pertanyaan ini boleh masuk SHARED cache (tanpa histori)
            # atau harus tetap per-sesi (ada histori). Lihat catatan di
            # bagian "EFISIENSI 1: RESPONSE CACHE — DUA TINGKAT" di atas.
            langchain_history = build_history(st.session_state.messages)
            has_history = bool(langchain_history)

            # EFISIENSI 1: Cek cache dulu sebelum panggil API
            cache_key = get_cache_key(prompt, context_string, has_history)
            cached = get_cached_response(cache_key, has_history)

        if cached:
            full_response = cached
            message_placeholder.markdown(full_response)
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
                set_cached_response(cache_key, full_response, has_history)

        message_placeholder.markdown(full_response)

        # Pasal rujukan SELALU ditampilkan (tidak ada lagi opsi untuk menyembunyikan).
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
        initial_sidebar_state="collapsed",
    )
    local_css()

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

    # ── Baris tombol: New chat / Unduh PDF / Tentang ───────────────────
    # (menggantikan sidebar — ditaruh DI ATAS banner hero)
    # PENTING: sengaja TIDAK memakai st.columns() di sini. Streamlit punya
    # CSS bawaan yang memaksa elemen [data-testid="column"] menumpuk penuh
    # (width 100%) di layar sempit, dan itu kadang mengalahkan override kita
    # sehingga urutan/lebar tombol jadi berantakan di mobile. Dengan menaruh
    # 3 widget berurutan (tanpa columns) lalu memaksa flex-row lewat CSS pada
    # container-nya (.st-key-top_actions di local_css), kita menghindari
    # aturan responsif bawaan tsb sepenuhnya.
    with st.container(key="top_actions"):
        if st.button("🆕", help="Mulai percakapan baru", use_container_width=True):
            st.session_state.messages = []
            # Tidak perlu st.rerun() manual — klik tombol sudah otomatis
            # memicu rerun. Memanggilnya lagi di sini menyebabkan DOUBLE
            # rerun yang membuat elemen lama (jawaban/disclaimer) tertinggal
            # di tampilan — itulah sumber bug tampilan sebelumnya.
        if st.session_state.messages:
            pdf_bytes = generate_chat_pdf(st.session_state.messages)
            st.download_button(
                "📥", data=pdf_bytes,
                file_name=f"riwayat_chatbot_tieng_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                help="Unduh percakapan (PDF)",
                use_container_width=True,
            )
        else:
            st.button("📥", help="Belum ada percakapan untuk diunduh", disabled=True, use_container_width=True)
        about_container = st.popover("❓", use_container_width=True) if hasattr(st, "popover") \
            else st.expander("❓ Tentang", expanded=False)
        with about_container:
            st.markdown("""
            **Tentang chatbot ini**

            Chatbot ini membantu warga Desa Tieng mencari informasi seputar
            **Peraturan Desa No. 02 Tahun 2024 tentang Pengelolaan Sampah**,
            termasuk aturan pemilahan sampah, sanksi, dan bank sampah.

            Setiap jawaban selalu menyertakan **rujukan pasal** agar mudah
            diverifikasi. Jika ada yang kurang jelas, silakan hubungi kantor
            Desa Tieng secara langsung.
            """)

    st.markdown("""
        <div class="app-hero">
            <h1>♻️ Asisten Warga Desa Tieng</h1>
            <p>Tanya apa saja seputar aturan pengelolaan sampah dan bank sampah desa.
            Jawaban diambil langsung dari Peraturan Desa Tieng No. 02 Tahun 2024.</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Sambutan + contoh pertanyaan (hanya saat belum ada obrolan) ────
    if not st.session_state.messages:
        st.markdown("**👋 Belum tahu mau tanya apa? Coba salah satu ini:**")
        # Dibungkus st.container(key=...) supaya baris ini bisa ditarget
        # CSS terpisah dari baris tombol atas (lihat .st-key-example_questions
        # di local_css): 2 kolom di desktop, 1 kolom (bertumpuk) di mobile.
        with st.container(key="example_questions"):
            cols = st.columns(2)
            for i, contoh in enumerate(CONTOH_PERTANYAAN):
                with cols[i % 2]:
                    st.markdown('<div class="example-chip">', unsafe_allow_html=True)
                    if st.button(contoh, key=f"contoh_{i}", use_container_width=True):
                        st.session_state["queued_prompt"] = contoh
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── Riwayat obrolan ─────────────────────────────────────────────
    # Pasal rujukan SELALU ditampilkan di setiap jawaban (tidak ada lagi
    # opsi untuk menyembunyikannya).
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("context_retrieved"):
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


if __name__ == "__main__":
    main()