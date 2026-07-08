"""
╔══════════════════════════════════════════════════════════════════╗
║       BATCH GENERATOR — Dataset Evaluasi RAGAS  v4.1 (Optimized) ║
║       Chatbot Perdes Tieng No. 02/2024                           ║
╠══════════════════════════════════════════════════════════════════╣
║  PERBAIKAN v4.0 (Efisiensi Token & Kecepatan):                   ║
║  • File sumber diganti ke versi Markdown (optimize).             ║
║  • Logika chunking v2 yang berat diganti dengan split            ║
║    berbasis header (### Pasal) yang super ringan.                ║
║  • max_context_chars diturunkan drastis dari 9000 ke 4000.       ║
║  • top_k_retrieval (10) & top_n_rerank (4) diturunkan.           ║
║  • System prompt dikompresi agar hemat token (TPM).              ║
╠══════════════════════════════════════════════════════════════════╣
║  PERBAIKAN v4.1 (Fix Retrieval — Pasal Kepanjangan):             ║
║  • MASALAH: Pasal 47 (4895 char, 13 sub-topik bank sampah) dan   ║
║    Pasal 1 (2960 char, 23 definisi) jadi 1 chunk raksasa —       ║
║    embedding-nya "encer" (mewakili banyak topik sekaligus)       ║
║    sehingga kalah rank saat retrieval, dan bagian belakangnya    ║
║    kepotong oleh max_context_chars=4000.                         ║
║  • FIX: split_large_pasal() — chunk yang > pasal_split_threshold ║
║    dipecah lagi otomatis per bullet level-0 ("* ..."), dengan    ║
║    header [BAB]/Pasal tetap disertakan di tiap pecahan supaya    ║
║    konteksnya tidak hilang. Berlaku generik untuk Pasal manapun  ║
║    yang kepanjangan, bukan cuma Pasal 47/1.                      ║
║  • reranker diganti ke BAAI/bge-reranker-v2-m3 (multibahasa,     ║
║    senasab dgn embedding bge-m3) — model sebelumnya              ║
║    (ms-marco-MiniLM, dilatih B. Inggris) kurang akurat           ║
║    mengurutkan skor relevansi untuk teks Bahasa Indonesia.       ║
╠══════════════════════════════════════════════════════════════════╣
║  CARA PAKAI:                                                     ║
║  ⚠️  Hapus folder faiss_index/ dan file chunks_cache.txt dulu!   ║
║     (chunking baru tidak kompatibel dengan index lama)           ║
║  python generate_ragas_dataset_v4.py                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import re
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# CrossEncoder Reranker
from sentence_transformers import CrossEncoder

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# KONFIGURASI (HEMAT TOKEN)
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    # File input
    "perdes_file"      : "perdes_sampah_optimize.txt",
    "faiss_dir"        : "faiss_index",
    "chunks_cache"     : "chunks_cache.txt",
    "dataset_xlsx"     : "ragas_dataset_perdes_tieng.xlsx",
    "dataset_sheet"    : "Dataset RAGAS",

    # File output
    "output_csv"       : "ragas_answers.csv",
    "log_file"         : "generate_log.jsonl",

    # Model
    "model_primary"    : "gemini-2.5-flash-lite",
    "model_fallback"   : "gemini-2.5-flash-lite",
    "temperature"      : 0.0,
    "max_output_tokens": 1024,

    # Reranker — v4.1: ganti ke model multibahasa (senasab dgn embedding bge-m3)
    "reranker_model"     : "BAAI/bge-reranker-v2-m3",   # sebelumnya: cross-encoder/ms-marco-MiniLM-L-6-v2 (khusus B.Inggris)

    # Chunking — v4.1: batas ukuran chunk sebelum dipecah lagi per sub-topik
    "pasal_split_threshold": 1200,  # chunk > 1200 char (mis. Pasal 47, Pasal 1) dipecah otomatis

    # Retrieval — DIOPTIMASI UNTUK TEKS PADAT
    "top_k_retrieval"    : 10,     # Turun dari 15
    "top_n_rerank"       : 4,      # Turun dari 7
    "rerank_threshold"   : 0.0,    
    "max_context_chars"  : 2500,   # Turun dari 9000 (Pangkas token >50%). Chunk kini lebih kecil & presisi
                                    # setelah split_large_pasal(), jadi angka ini bisa diturunkan lagi
                                    # (mis. ke 2500-3000) kalau ingin hemat token lebih jauh.

    # Query expansion
    "use_query_expansion": True,

    # Rate limit & retry
    "rpm_limit"       : 10,
    "delay_between"   : 6.5,
    "max_retries"     : 4,
    "backoff_base"    : 8,

    # Resume
    "resume"          : True,
}


# ══════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER (Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════════
class HybridRetriever:
    def __init__(self, retrievers: list, weights: list, c: int = 60):
        self.retrievers = retrievers
        self.weights    = weights
        self.c          = c

    def invoke(self, query: str) -> list:
        all_results = [r.invoke(query) for r in self.retrievers]
        scores: dict = {}
        doc_map: dict = {}
        for docs, weight in zip(all_results, self.weights):
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (self.c + rank + 1))
                doc_map[key] = doc
        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in ranked]


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 1: CHUNKING BERBASIS MARKDOWN (SANGAT SEDERHANA & CEPAT)
# ══════════════════════════════════════════════════════════════════════
def split_large_pasal(pasal_chunk: str, max_len: int = 1200) -> list[str]:
    """
    v4.1: Pecah 1 chunk Pasal yang kepanjangan (mis. Pasal 47 = 4895 char,
    berisi 13 sub-topik bank sampah; atau Pasal 1 = 2960 char, berisi 23
    definisi) menjadi beberapa sub-chunk yang lebih kecil dan presisi.

    Kenapa perlu: chunk yang mewakili banyak topik sekaligus punya embedding
    yang "encer" sehingga kalah rank saat retrieval dibanding chunk lain yang
    lebih spesifik, dan kalau chunk-nya melebihi max_context_chars, bagian
    belakangnya (mis. "Sistem Bagi Hasil", "Wadah Sampah" di Pasal 47) bisa
    kepotong oleh _smart_truncate() sebelum sempat dibaca LLM.

    Cara kerja: split berdasarkan bullet level-0 ("* ...") — setiap sub-topik
    dalam Pasal biasanya diawali bullet level-0 baru (mis. "* Jenis Tabungan
    diatur sebagai berikut:" atau "* Bank sampah: tempat pemilahan ..."),
    dengan sub-bullet ("  - ...") di bawahnya ikut ke induknya. Header
    "[BAB ...]" dan "### Pasal N (...)" disalin ulang ke tiap pecahan supaya
    konteksnya (pasal & bab berapa) tetap ada meski chunk-nya sudah kecil.

    Chunk yang masih di bawah `max_len` dibiarkan apa adanya (tidak dipecah),
    supaya Pasal pendek/sedang tidak ikut terpecah tanpa perlu.
    """
    if len(pasal_chunk) <= max_len:
        return [pasal_chunk]

    lines = pasal_chunk.split("\n")

    # Ambil baris header di awal chunk: "[BAB ...]" dan "### Pasal N (...)"
    # (semua baris sebelum bullet level-0 pertama dianggap header)
    header_lines = []
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("* "):
        header_lines.append(lines[i])
        i += 1
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[i:])

    if not body.strip():
        # Tidak ada bullet level-0 yang bisa dipakai sebagai titik pecah
        # (jarang terjadi) — kembalikan chunk asli saja daripada dipaksakan.
        return [pasal_chunk]

    # Pecah body berdasarkan bullet level-0 baru; sub-bullet ("  - ...")
    # otomatis ikut ke segmen induknya karena regex hanya match "* " di awal baris
    segments = re.split(r'\n(?=\* )', body)
    segments = [s.strip() for s in segments if s.strip()]

    sub_chunks = [f"{header}\n{seg}" for seg in segments]
    log.debug(
        "  Pasal kepanjangan (%d char) dipecah jadi %d sub-chunk", 
        len(pasal_chunk), len(sub_chunks)
    )
    return sub_chunks


def build_chunks_from_text(raw_text: str, pasal_split_threshold: int = 1200) -> list[str]:
    """Chunking berbasis Markdown Header, dengan auto-split untuk Pasal kepanjangan (v4.1)."""
    # Pecah berdasarkan pola "### Pasal"
    parts = re.split(r'\n(?=###\s+Pasal)', raw_text)
    
    chunks = []
    current_bab = "KETENTUAN UMUM"
    
    for part in parts:
        part = part.strip()
        if not part: continue
        
        # v4.1 FIX BUG LABEL BAB: karena split hanya terjadi di "### Pasal"
        # (bukan di "## BAB"), header BAB berikutnya sering ikut nyangkut di
        # EKOR teks Pasal terakhir suatu BAB (mis. "## BAB XII ..." nyangkut
        # di ekor Pasal 47, padahal Pasal 47 masih milik BAB XI). Versi lama
        # langsung memakai bab_match ini untuk melabeli chunk saat ini —
        # akibatnya Pasal 38, 43, 44, 45, 47, 48 (semua Pasal terakhir di
        # BAB masing-masing) salah label, kebawa nama BAB SETELAHNYA.
        # Fix: potong dulu baris "## BAB ..." dari isi pasal saat ini (bukan
        # bagian dari pasal ini), lalu simpan sebagai current_bab untuk
        # PASAL BERIKUTNYA saja.
        bab_match = re.search(r'\n##\s+(BAB\s+[^\n]+)', "\n" + part)
        next_bab = None
        if bab_match:
            part = part[:bab_match.start()].strip()
            next_bab = bab_match.group(1).strip()

        full_chunk = f"[{current_bab}]\n{part}"
        # v4.1: pecah lagi kalau chunk-nya kepanjangan (Pasal 47, Pasal 1, dll)
        chunks.extend(split_large_pasal(full_chunk, max_len=pasal_split_threshold))

        if next_bab:
            current_bab = next_bab
        
    result = [c for c in chunks if len(c.strip()) > 10]
    log.info("Total chunk dihasilkan: %d (Markdown chunker + auto-split pasal panjang)", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 2: QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════
_EXPANSION_PROMPT = """Tugas: Buat 2 variasi pertanyaan yang BERBEDA KALIMAT tapi SAMA MAKNA.
Gunakan kata kunci alternatif agar mencakup sudut pandang berbeda.
Gunakan Bahasa Indonesia. Jawab HANYA 2 baris variasi, tanpa penomoran atau penjelasan.

Pertanyaan asli: {question}"""


def expand_query(question: str, llm) -> list[str]:
    try:
        prompt = ChatPromptTemplate.from_messages([("human", _EXPANSION_PROMPT)])
        chain  = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})
        variants = [v.strip() for v in result.strip().split("\n") if v.strip()]
        return [question] + variants[:2]
    except Exception as e:
        log.warning("Query expansion gagal, pakai pertanyaan asli: %s", e)
        return [question]


def multi_query_retrieve(queries: list[str], hybrid_retriever) -> list[Document]:
    seen   = set()
    merged = []
    for q in queries:
        for doc in hybrid_retriever.invoke(q):
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 3: INISIALISASI RESOURCES
# ══════════════════════════════════════════════════════════════════════
def init_resources(cfg: dict):
    log.info("Memuat model embedding BAAI/bge-m3 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )

    chunks = []

    if not os.path.exists(cfg["faiss_dir"]):
        log.info("faiss_index belum ada — membangun dari %s ...", cfg["perdes_file"])
        if not os.path.exists(cfg["perdes_file"]):
            raise FileNotFoundError(
                f"File '{cfg['perdes_file']}' tidak ditemukan."
            )
        with open(cfg["perdes_file"], "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = build_chunks_from_text(raw_text, pasal_split_threshold=cfg["pasal_split_threshold"])

        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(cfg["faiss_dir"])
        with open(cfg["chunks_cache"], "w", encoding="utf-8") as f:
            f.write("\n<<<CHUNK_SEPARATOR>>>\n".join(chunks))
        log.info("FAISS index disimpan ke '%s'", cfg["faiss_dir"])
    else:
        log.info("Memuat FAISS index dari '%s' ...", cfg["faiss_dir"])
        if os.path.exists(cfg["chunks_cache"]):
            with open(cfg["chunks_cache"], "r", encoding="utf-8") as f:
                chunks = f.read().split("\n<<<CHUNK_SEPARATOR>>>\n")
            log.info("Chunk cache dimuat: %d chunks", len(chunks))
        elif os.path.exists(cfg["perdes_file"]):
            with open(cfg["perdes_file"], "r", encoding="utf-8") as f:
                raw_text = f.read()
            chunks = build_chunks_from_text(raw_text, pasal_split_threshold=cfg["pasal_split_threshold"])

    vector_db = FAISS.load_local(
        cfg["faiss_dir"], embeddings, allow_dangerous_deserialization=True
    )
    log.info("FAISS index dimuat (%d vektor).", vector_db.index.ntotal)

    documents = [Document(page_content=c) for c in chunks]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = cfg["top_k_retrieval"]

    faiss_retriever = vector_db.as_retriever(
        search_kwargs={"k": cfg["top_k_retrieval"]}
    )
    hybrid_retriever = HybridRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    log.info("HybridRetriever (FAISS 60%% + BM25 40%%) siap.")

    log.info("Memuat CrossEncoder reranker (%s) ...", cfg["reranker_model"])
    # v4.1: ganti dari cross-encoder/ms-marco-MiniLM-L-6-v2 (B.Inggris) ke
    # BAAI/bge-reranker-v2-m3 (multibahasa, senasab dgn embedding bge-m3),
    # supaya skor relevansi lebih terkalibrasi untuk teks Bahasa Indonesia.
    reranker = CrossEncoder(cfg["reranker_model"], max_length=512)
    log.info("CrossEncoder siap.")

    return hybrid_retriever, reranker


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 4: PROMPT TEMPLATE (VERSI KOMPRESI)
# ══════════════════════════════════════════════════════════════════════
def get_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", """Anda Asisten Desa Tieng (Perdes No. 02/2024 tentang Sampah).
JAWAB HANYA BERDASARKAN KONTEKS BERIKUT:
{context}

ATURAN WAJIB:
1. Jangan mengarang info/pasal di luar konteks. Jika tidak ada di konteks, jawab: "Belum diatur secara eksplisit dalam Perdes No.02/2024."
2. Sebutkan dasar aturan/pasalnya di akhir kalimat (Contoh: "...sesuai Pasal 12 ayat 1.").
3. Gunakan bahasa yang ramah, sederhana, dan mudah dimengerti warga awam.
4. Jika memakai istilah teknis (Organik, Residu, dll), berikan 1-2 contoh sederhana."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 5: RETRIEVAL + RERANKING
# ══════════════════════════════════════════════════════════════════════
def _deduplicate_chunks(docs: list[Document]) -> list[Document]:
    seen_content = []
    result = []
    for doc in docs:
        content = doc.page_content.strip()
        is_dup = any(
            content in existing or existing in content
            for existing in seen_content
        )
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

def retrieve_and_rerank(
    question: str,
    hybrid_retriever,
    reranker: CrossEncoder,
    llm,
    top_n: int,
    max_chars: int,
    use_expansion: bool = True,
    rerank_threshold: float = -3.0,
) -> tuple[list[str], str]:
    queries = expand_query(question, llm) if use_expansion else [question]
    candidate_docs = multi_query_retrieve(queries, hybrid_retriever)
    
    if not candidate_docs:
        return [], ""

    candidate_docs = _deduplicate_chunks(candidate_docs)

    pairs  = [(question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)

    scored_docs = [
        (score, doc)
        for score, doc in zip(scores, candidate_docs)
        if score >= rerank_threshold
    ]

    if not scored_docs:
        ranked   = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:3]]
    else:
        ranked   = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:top_n]]

    contexts    = [doc.page_content for doc in top_docs]
    context_str = "\n\n---\n\n".join(contexts)
    context_str = _smart_truncate(context_str, max_chars)

    return contexts, context_str


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 6: PANGGIL LLM DENGAN EXPONENTIAL BACKOFF
# ══════════════════════════════════════════════════════════════════════
END_CHARS = set('.!?)"')

def is_answer_complete(text: str) -> bool:
    t = text.strip()
    return bool(t) and t[-1] in END_CHARS

def call_llm_with_backoff(
    chain,
    payload: dict,
    max_retries: int,
    backoff_base: int,
    llm_primary=None,
    api_key: str = "",
) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser

    for attempt in range(max_retries):
        try:
            result = chain.invoke(payload)
            if not is_answer_complete(result) and attempt < max_retries - 1:
                log.warning(
                    "Jawaban terpotong percobaan %d/%d (akhir: ...%s), retry ...",
                    attempt + 1, max_retries, result.strip()[-40:]
                )
                boosted_tokens = 1024 + (attempt + 1) * 512
                if llm_primary is not None and api_key:
                    boosted_llm = ChatGoogleGenerativeAI(
                        model=CONFIG["model_primary"],
                        temperature=CONFIG["temperature"],
                        max_output_tokens=boosted_tokens,
                        google_api_key=api_key,
                    )
                    chain = get_prompt_template() | boosted_llm | StrOutputParser()
                time.sleep(3)
                continue
            return result
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in ["429", "quota", "rate", "resource_exhausted"])
            if is_rate_limit and attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                log.warning("Rate limit — percobaan %d/%d, tunggu %ds ...", attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            raise


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 7: MUAT DATASET PERTANYAAN
# ══════════════════════════════════════════════════════════════════════
def load_questions(cfg: dict) -> pd.DataFrame:
    path = cfg["dataset_xlsx"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset '{path}' tidak ditemukan.")
    df = pd.read_excel(path, sheet_name=cfg["dataset_sheet"], header=1)
    df.columns = ["no", "kategori", "question", "ground_truth", "referensi", "tipe", "catatan"]
    df = df.dropna(subset=["question", "ground_truth"]).reset_index(drop=True)
    log.info("Dataset dimuat: %d pertanyaan dari '%s'", len(df), path)
    return df


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 8: RESUME
# ══════════════════════════════════════════════════════════════════════
def load_existing_results(output_csv: str) -> dict[int, dict]:
    if not os.path.exists(output_csv):
        return {}
    existing = pd.read_csv(output_csv)
    return {int(row["no"]): row.to_dict() for _, row in existing.iterrows()}


# ══════════════════════════════════════════════════════════════════════
# BAGIAN 9: MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  BATCH GENERATOR v4.0 (Optimized) — Dataset Evaluasi RAGAS")
    print("  Chatbot Perdes Tieng No. 02/2024")
    print("=" * 65)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY tidak ditemukan.")

    hybrid_retriever, reranker = init_resources(CONFIG)

    llm_primary = ChatGoogleGenerativeAI(
        model=CONFIG["model_primary"],
        temperature=CONFIG["temperature"],
        max_output_tokens=CONFIG["max_output_tokens"],
        google_api_key=api_key,
    )
    llm_fallback = ChatGoogleGenerativeAI(
        model=CONFIG["model_fallback"],
        temperature=CONFIG["temperature"],
        max_output_tokens=CONFIG["max_output_tokens"],
        google_api_key=api_key,
    )

    prompt_template = get_prompt_template()
    chain_primary   = prompt_template | llm_primary  | StrOutputParser()
    chain_fallback  = prompt_template | llm_fallback | StrOutputParser()

    df_questions = load_questions(CONFIG)

    existing = {}
    if CONFIG["resume"]:
        existing = load_existing_results(CONFIG["output_csv"])
        if existing:
            log.info("Resume mode: %d soal sudah selesai.", len(existing))

    log_fh = Path(CONFIG["log_file"]).open("a", encoding="utf-8")

    results = []
    stats   = {"success": 0, "fallback_used": 0, "error": 0, "skipped": 0}
    pbar    = tqdm(df_questions.iterrows(), total=len(df_questions), desc="Generating")

    for idx, row in pbar:
        no        = int(row["no"])
        question  = str(row["question"]).strip()
        gt        = str(row["ground_truth"]).strip()
        kategori  = str(row["kategori"])
        referensi = str(row["referensi"])
        tipe      = str(row["tipe"])

        pbar.set_postfix({"no": no, "ok": stats["success"], "err": stats["error"]})

        if CONFIG["resume"] and no in existing:
            prev = str(existing[no].get("answer", "")).strip()
            if prev and prev[-1] in END_CHARS and not prev.startswith("[ERROR]"):
                results.append(existing[no])
                stats["skipped"] += 1
                continue

        try:
            contexts, context_str = retrieve_and_rerank(
                question,
                hybrid_retriever,
                reranker,
                llm              = llm_primary,
                top_n            = CONFIG["top_n_rerank"],
                max_chars        = CONFIG["max_context_chars"],
                use_expansion    = CONFIG["use_query_expansion"],
                rerank_threshold = CONFIG["rerank_threshold"],
            )
        except Exception as e:
            log.error("Retrieval gagal soal #%d: %s", no, e)
            contexts, context_str = [], ""

        payload = {
            "context"     : context_str,
            "chat_history": [],
            "question"    : question,
        }

        answer     = ""
        model_used = CONFIG["model_primary"]
        error_msg  = ""

        try:
            answer = call_llm_with_backoff(
                chain_primary, payload,
                max_retries=CONFIG["max_retries"],
                backoff_base=CONFIG["backoff_base"],
                llm_primary=llm_primary,
                api_key=api_key,
            )
            stats["success"] += 1
        except Exception as e_primary:
            try:
                answer = call_llm_with_backoff(
                    chain_fallback, payload,
                    max_retries=CONFIG["max_retries"],
                    backoff_base=CONFIG["backoff_base"],
                    llm_primary=llm_primary,
                    api_key=api_key,
                )
                model_used = CONFIG["model_fallback"]
                stats["fallback_used"] += 1
                stats["success"] += 1
            except Exception as e_fallback:
                error_msg  = str(e_fallback)
                answer     = f"[ERROR] {error_msg}"
                model_used = "error"
                stats["error"] += 1

        result_row = {
            "no"             : no,
            "kategori"       : kategori,
            "tipe"           : tipe,
            "referensi_pasal": referensi,
            "question"       : question,
            "answer"         : answer,
            "contexts"       : json.dumps(contexts, ensure_ascii=False),
            "ground_truth"   : gt,
            "context_raw"    : context_str,
            "model_used"     : model_used,
            "error"          : error_msg,
            "timestamp"      : datetime.now().isoformat(timespec="seconds"),
        }
        results.append(result_row)

        log_fh.write(json.dumps(result_row, ensure_ascii=False) + "\n")
        log_fh.flush()

        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(CONFIG["output_csv"], index=False, encoding="utf-8-sig")

        if stats["success"] + stats["error"] < len(df_questions):
            extra = 4.0 if CONFIG["use_query_expansion"] else 0.0
            time.sleep(CONFIG["delay_between"] + extra)

    log_fh.close()
    pbar.close()

    df_result = pd.DataFrame(results)
    df_result.to_csv(CONFIG["output_csv"], index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("  SELESAI")
    print("=" * 65)
    print(f"  Total soal diproses : {len(df_questions)}")
    print(f"  Berhasil            : {stats['success']}")
    print(f"  Pakai fallback      : {stats['fallback_used']}")
    print(f"  Di-skip (resume)    : {stats['skipped']}")
    print(f"  Error               : {stats['error']}")
    print("=" * 65)

if __name__ == "__main__":
    main()