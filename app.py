import streamlit as st
import os
import re
import pandas as pd
from datetime import datetime

# --- LANGCHAIN IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# --- 1. OPTIMASI DB DENGAN CONTEXTUAL BAB INJECTION (PERBAIKAN REGEX) ---
@st.cache_resource
def get_resources():
    """Memuat model embedding dan menyisipkan keterangan BAB ke setiap pasal yang ada di bawahnya"""
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    file_path = "perdes_sampah.txt"
    if not os.path.exists("faiss_index"):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Strategi: Pecah teks menggunakan regex berbasis kemunculan kata "Pasal <angka>"
            # Positif lookahead (?=...) digunakan agar kata "Pasal" tidak ikut terhapus saat di-split
            pasal_splits = re.split(r'\n(?=Pasal\s+\d+)', raw_text, flags=re.IGNORECASE)
            
            chunks = []
            current_bab = "BAB I KETENTUAN UMUM" # Fallback awal dokumen [cite: 1]
            
            for part in pasal_splits:
                part_cleaned = part.strip()
                if not part_cleaned:
                    continue
                
                # Lacak status BAB saat ini. Jika di dalam potongan teks ini terdapat deklarasi BAB baru,
                # kita perbarui variabel current_bab untuk pasal-pasal berikutnya.
                bab_match = re.search(r'(BAB\s+(?:[I|V|X|L|C]+|\d+)[^\n]*)', part_cleaned, re.IGNORECASE)
                if bab_match:
                    current_bab = bab_match.group(1).strip()
                
                # Masukkan text ke dalam chunk. Kita injeksikan info BAB di bagian atas
                # agar sewaktu similarity search, LLM selalu tahu pasal ini merujuk ke BAB mana.
                text_wrapper = f"[{current_bab}]\n{part_cleaned}"
                chunks.append(text_wrapper)
            
            # Filter chunk yang terlalu pendek atau kosong
            chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
            
            # Simpan ke database vektor FAISS lokal
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
        else:
            return None, None
    
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return embeddings, vector_db

# --- 2. TEMPLATE PROMPT DENGAN MESSAGES PLACEHOLDER (MEMORI) ---
def get_chat_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", """Anda adalah asisten virtual resmi Desa Tieng. Jawablah pertanyaan warga mengenai regulasi sampah dengan sopan, ringkas, dan ramah.
        
        Gunakan KONTEKS berikut sebagai acuan utama Anda untuk menjawab. Jangan bertele-tele agar hemat token.
        
        KONTEKS:
        {context}
        
        ATURAN:
        1. Gunakan bahasa yang merakyat namun tetap sopan.
        2. Sebutkan nomor Pasal atau Bab jika informasinya tertera pada konteks.
        3. Jika tidak ada secara spesifik (misal: sungai), gunakan logika 'tempat terlarang' dari Pasal 12 atau 38 untuk menghimbau warga.
        4. Nama pejabat ada di bagian akhir (pejabat desa).
        5. Jika jawaban tidak ada di dalam konteks, katakan dengan jujur dan sopan bahwa informasi tersebut belum diatur di Perdes atau tidak ditemukan."""),
        
        # Riwayat chat akan disisipkan secara otomatis di sini oleh LangChain
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", "{question}")
    ])

# --- 3. UI CUSTOMIZATION (CSS) ---
def local_css():
    st.markdown("""
        <style>
            [data-testid="stHeader"] { background-color: rgba(0,0,0,0); color: rgba(0,0,0,0); }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stFooter"] {display: none !important;}
            .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; max-width: 100% !important; }
            .stChatMessage { border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="Chatbot Desa Tieng", page_icon="🤖", initial_sidebar_state="expanded")
    local_css()
    
    st.header("🤖 Chatbot Peraturan Desa Tieng")
    st.subheader("Informasi Pengelolaan Sampah & Bank Sampah")

    api_key = os.getenv("GOOGLE_API_KEY") or (st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else None)
    if not api_key:
        st.error("API Key Gemini (GOOGLE_API_KEY) belum dikonfigurasi di file .env atau Secrets!")
        return

    embeddings, vector_db = get_resources()
    if vector_db is None:
        st.error("File 'perdes_sampah.txt' tidak ditemukan. Sediakan file tersebut di root direktori proyek Anda.")
        return

    # Inisialisasi riwayat pesan di session state Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Menampilkan percakapan yang sudah terjadi sebelumnya ke layar UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Logika Input Chat Pengguna
    if prompt := st.chat_input("Tanyakan aturan sampah, sanksi, atau bank sampah..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 1. Menyiapkan Konteks dan Riwayat (Sama seperti sebelumnya)
            docs = vector_db.similarity_search(prompt, k=4)
            context_string = "\n\n".join([doc.page_content for doc in docs])
            
            recent_messages = st.session_state.messages[:-1][-4:] 
            langchain_history = []
            for msg in recent_messages:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))

            prompt_template = get_chat_prompt_template()

            # --- MULAI PERCABANGAN / FALLBACK BERJENJANG ---
            try:
                # PILIHAN 1: Gemini 2.5 Flash Lite (Paling Hemat Token & Ringan)
                st.caption("⚡ Menggunakan Gemini 2.5 Flash Lite")
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    temperature=0.1, 
                    google_api_key=api_key
                )
                chain = prompt_template | model | StrOutputParser()
                
                for chunk in chain.stream({"context": context_string, "chat_history": langchain_history, "question": prompt}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            except Exception as e_lite:
                try:
                    # PILIHAN 2: Jika Pilihan 1 Gagal, Beralih ke Gemini 2.5 Flash
                    st.caption("🔄 Lite sibuk/habis kuota, beralih ke Gemini 2.5 Flash...")
                    model_25_flash = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash", 
                        temperature=0.1, 
                        google_api_key=api_key
                    )
                    chain = prompt_template | model_25_flash | StrOutputParser()
                    
                    # Reset respons jika sempat terisi setengah sebelum error
                    full_response = "" 
                    for chunk in chain.stream({"context": context_string, "chat_history": langchain_history, "question": prompt}):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        
                except Exception as e_25:
                    # PILIHAN 3: Jika Pilihan 2 Gagal, Gunakan Gemini 1.5 Flash (Latest) sebagai Penyelamat
                    st.caption("🚨 Menggunakan Cadangan Terakhir: Gemini 1.5 Flash")
                    model_15_flash = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        temperature=0.1, 
                        google_api_key=api_key
                    )
                    chain = prompt_template | model_15_flash | StrOutputParser()
                    
                    full_response = chain.invoke({
                        "context": context_string, 
                        "chat_history": langchain_history, 
                        "question": prompt
                    })

            # --- AKHIR PERCABANGAN ---
            
            # Tampilkan hasil akhir secara utuh dan simpan ke memori
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "context_retrieved": context_string 
            })

    # Bagian Sidebar Kontrol & Evaluasi
    with st.sidebar:
        st.title("Panel Kontrol")
        if st.button("Hapus Riwayat Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Fitur opsional: Mengunduh data percakapan saat ini untuk bahan evaluasi Tugas Akhir
        if st.session_state.messages:
            df = pd.DataFrame([{
                "Role": msg["role"], 
                "Content": msg["content"], 
                "Context_Retrieved": msg.get("context_retrieved", "")
            } for msg in st.session_state.messages])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Unduh Riwayat Obrolan (CSV)",
                data=csv,
                file_name=f"evaluasi_rag_tieng_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()