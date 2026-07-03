import time
import pandas as pd
import logging
import os
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision, AnswerRelevancy
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# --- KONFIGURASI LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("evaluasi.log"), logging.StreamHandler()])

# --- KONFIGURASI ---
API_KEYS = ["AIzaSyDczMb4Ld-JKMmMdQ6FMKrYxxnvpxpZz6U", "AIzaSyA9slMfWREpVniaMxS3Sz2FS_jK74hpRUk", "AIzaSyClgUrry4MFQncQaU-QUAeVPBoYcbrhCRI"]
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
INPUT_FILE = "evaluasi_ragas.csv"
OUTPUT_FILE = "hasil_evaluasi_final.csv"

def run_evaluation():
    df = pd.read_csv(INPUT_FILE)
    
    # Checkpointing: Memuat hasil yang sudah ada agar tidak mengulang dari nol
    if os.path.exists(OUTPUT_FILE):
        results_df = pd.read_csv(OUTPUT_FILE)
        processed_indices = set(results_df.index) # Asumsi index sama
        logging.info(f"Checkpoint ditemukan. Melanjutkan dari baris ke-{len(results_df)}")
    else:
        results_df = pd.DataFrame()
        processed_indices = set()

    run_config = RunConfig(max_workers=1)
    
    for i, row in df.iterrows():
        if i in processed_indices: continue
        
        success = False
        row_score = {}
        
        for key in API_KEYS:
            if success: break
            
            for model in MODELS:
                try:
                    llm = ChatOpenAI(model=model, api_key=key, base_url=BASE_URL)
                    emb = OpenAIEmbeddings(model="text-embedding-004", api_key=key, base_url=BASE_URL)
                    dataset = Dataset.from_dict({
                        "question": [row["Question"]], "answer": [row["Answer"]],
                        "contexts": [[row["Context_Retrieved"]]], "ground_truth": [row["Ground_Truth"]]
                    })
                    
                    # Evaluasi Per Metrik dengan jeda
                    metrics = [Faithfulness(llm=llm), ContextPrecision(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb)]
                    for m in metrics:
                        m_score = evaluate(dataset=dataset, metrics=[m], run_config=run_config)
                        row_score.update(m_score.to_pandas().iloc[0].to_dict())
                        time.sleep(15) # Jeda antar metrik
                    
                    # Simpan hasil sementara per baris
                    row_score_df = pd.DataFrame([row_score])
                    row_score_df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
                    
                    logging.info(f"Berhasil: Baris {i+1} selesai menggunakan {model}")
                    success = True
                    time.sleep(20) # Jeda antar baris
                    break
                except Exception as e:
                    logging.warning(f"Gagal {model}: {str(e)[:30]}...")
                    time.sleep(10)
        
        if not success:
            logging.error(f"Baris {i+1} gagal total.")

run_evaluation()