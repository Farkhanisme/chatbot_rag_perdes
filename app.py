import streamlit as st
import os
import pandas as pd
import base64
from datetime import datetime
import asyncio

# --- LANGCHAIN MODERN IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# --- 1. OPTIMASI CACHE & RESOURCE ---
@st.cache_resource
def get_resources():
    """Memuat model embedding dan database sekali saja ke memori"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    file_path = "perdes_sampah.txt"
    if not os.path.exists("faiss_index"):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = text_splitter.split_text(raw_text)
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
        else:
            return None, None
    
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return embeddings, vector_db

# --- 2. KONFIGURASI PROMPT ---
def get_prompt_template():
    template = """Anda adalah asisten virtual resmi Desa Tieng. Jawablah pertanyaan warga dengan sopan dan mudah dimengerti HANYA berdasarkan dokumen di bawah ini.
    
    KONTEKS:
    {context}
    
    ATURAN:
    1. Gunakan bahasa yang merakyat namun tetap sopan.
    2. Jika ada di dokumen, sebutkan Pasal/Bab-nya.
    3. Jika tidak ada secara spesifik (misal: sungai), gunakan logika 'tempat terlarang' dari Pasal 12 atau 38 untuk menghimbau warga.
    4. Nama pejabat ada di bagian akhir (Berita Acara).
    5. Jika benar-benar tidak ada, katakan maaf dengan jujur.

    Pertanyaan: {question}
    Jawaban:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- 3. UI RENDERER (CSS SEDERHANA) ---
def local_css():
    st.markdown("""
        <style>
            .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
            .main { background-color: #f5f7f9; }
            .stButton>button { width: 100%; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

# --- 4. MAIN APP ---
def main():
    st.set_page_config(page_title="Chatbot Desa Tieng", page_icon="🤖")
    local_css()
    
    st.header("🤖 Chatbot Peraturan Desa Tieng")
    st.subheader("Informasi Pengelolaan Sampah & Bank Sampah")

    # Cara yang lebih aman untuk cek API Key di lokal dan cloud
    api_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("API Key belum disetel di .env")
            return

    # Inisialisasi Resource (Caching Aktif)
    embeddings, vector_db = get_resources()
    
    if vector_db is None:
        st.error("File perdes_sampah.txt tidak ditemukan untuk inisialisasi.")
        return

    # Inisialisasi Riwayat Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan Riwayat Chat (Gaya Streamlit Modern)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Tanyakan sesuatu tentang sampah desa..."):
        # Tambahkan pertanyaan user ke UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses Jawaban (Streaming)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 1. Retrieval
                docs = vector_db.similarity_search(prompt, k=7)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Gunakan penamaan model yang lebih spesifik untuk jalur v1
                # Jika gemini-1.5-flash tetap 404, gunakan gemini-1.5-flash-001
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash", 
                    temperature=0.1, 
                    google_api_key=api_key
                )
                
                chain = get_prompt_template() | model | StrOutputParser()
                
                # Proses Streaming
                with st.spinner("Menghubungi server..."):
                    for chunk in chain.stream({"context": context, "question": prompt}):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        
            except Exception as e:
                # Jika masih error 404, gunakan fallback ke gemini-pro yang hampir pasti tersedia di semua versi
                st.warning("Menggunakan model cadangan karena kendala koneksi API.")
                model_fallback = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1, google_api_key=api_key)
                chain = get_prompt_template() | model_fallback | StrOutputParser()
                
                # Jalankan ulang invoke (atau stream) untuk fallback
                full_response = chain.invoke({"context": context, "question": prompt})
                message_placeholder.markdown(full_response)

    # Sidebar Tools
    with st.sidebar:
        st.title("Opsi")
        if st.button("Hapus Riwayat Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.session_state.messages:
            df = pd.DataFrame(st.session_state.messages)
            csv = df.to_csv(index=False)
            st.download_button("Download Chat (CSV)", csv, "history_chat.csv", "text/csv")

if __name__ == "__main__":
    main()