import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API Key dari .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. Konfigurasi API & Model
# Catatan: Gunakan gemini-1.5-flash untuk evaluasi karena lebih stabil untuk Ragas
os.environ["GOOGLE_API_KEY"] = api_key
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 2. Dataset Evaluasi
# Pastikan setiap list (question, answer, contexts, ground_truth) memiliki panjang yang sama (26)
data_sampel = {
    "question": [
        "Apa yang dimaksud dengan sampah rumah tangga?",
        "Apa itu Bank Sampah menurut Perdes ini?",
        "Apa saja yang termasuk sampah spesifik?",
        "Apa tujuan utama pengelolaan sampah di Desa Tieng?",
        "Siapa yang membentuk Pemuda Peduli Lingkungan?",
        "Apa tugas Unit Usaha BUMDesa dalam hal sampah?",
        "Bagaimana cara melakukan pengurangan sampah?",
        "Apa saja tahapan dalam menangani sampah?",
        "Fasilitas tempat sampah apa yang harus disediakan di rumah?",
        "Siapa yang bertanggung jawab mengangkut sampah dari rumah ke bank sampah?",
        "Siapa yang mengangkut sampah dari RPS ke TPA?",
        "Siapa yang menetapkan besaran retribusi untuk anggota?",
        "Apakah BUMDesa boleh memungut biaya layanan sampah?",
        "Perilaku apa yang bisa mendapatkan penghargaan/apresiasi?",
        "Apa bentuk depresiasi (sanksi) bagi pelanggar?",
        "Dengan siapa Pemerintah Desa boleh bekerja sama?",
        "Kapan kompensasi diberikan kepada warga?",
        "Apa saja bentuk kompensasi yang bisa diterima warga?",
        "Apa kewajiban masyarakat dalam pengelolaan sampah?",
        "Bolehkah Pengurus RW membuat peraturan khusus sampah?",
        "Apa syarat papan larangan buang sampah?",
        "Siapa yang mendirikan dan mengelola Bank Sampah?",
        "Bagaimana mekanisme tabungan di Bank Sampah?",
        "Apa saja jenis tabungan di Bank Sampah?",
        "Jenis sampah kertas apa saja yang diterima?",
        "Bagaimana sistem bagi hasil bank sampah ditentukan?"
    ],
    "answer": [
        # Paste 26 jawaban dari kolom 'Content' di CSV hasil download Streamlit Anda
    ],
    "contexts": [
        # Paste 26 data dari kolom 'Context_FAISS' di CSV hasil download Streamlit Anda
        # Formatnya harus: ["isi konteks 1"], ["isi konteks 2"], dst.
    ],
    "ground_truth": [
        "Sampah dari kegiatan sehari-hari rumah tangga, organik, tidak termasuk tinja.",
        "Tempat pemilahan dan pengumpulan sampah bernilai ekonomi.",
        "Sampah B3, medis, puing, dan limbah pertanian.",
        "Mewujudkan lingkungan bersih, sehat, dan mengubah perilaku warga.",
        "Kelompok masyarakat yang dibentuk oleh Pemerintah Desa.",
        "Melaksanakan kebijakan dan rencana Pemerintah Desa serta kerjasama kelompok.",
        "Pembatasan timbulan, pendauran ulang, dan pemanfaatan kembali.",
        "Pemilahan, pengumpulan, pengangkutan, pengolahan, dan pemrosesan akhir.",
        "Tempat sampah organik, anorganik, dan residu.",
        "Pengelola sampah yang dibentuk oleh Kepala Desa.",
        "Kelompok pengelola dan atau pemerintah desa.",
        "Hasil musyawarah kelompok yang disetujui Pemerintah Desa.",
        "Boleh, sesuai tarif yang ditetapkan Keputusan Kepala Desa.",
        "Inovasi, laporan pelanggaran, dan tertib penanganan sampah.",
        "Kritikan, komentar, ulasan, atau denda barang/jasa.",
        "Pemerintah di atasnya, masyarakat, atau pihak swasta.",
        "Akibat dampak negatif penanganan sampah di TPA.",
        "Pemulihan lingkungan, biaya kesehatan, dan ganti rugi.",
        "Wajib aktif dalam proses pengelolaan dan pemilahan sampah.",
        "Boleh, dikoordinir oleh Pengurus RW setempat.",
        "Jelas, mudah dibaca, singkat, dan atas nama Pemerintah Desa.",
        "Masyarakat atau kelompok masyarakat secara mandiri.",
        "Pemilahan, penyerahan, penimbangan, pencatatan, dan bagi hasil.",
        "Individu (biasa, pendidikan, lebaran, sosial) dan kolektif.",
        "Koran, majalah, kardus, dan dupleks.",
        "Hasil rapat pengurus bank sampah yang disosialisasikan."
    ]
}

# 3. Validasi Panjang Data (Penting untuk menghindari error Ragas)
lengths = [len(v) for v in data_sampel.values()]
if len(set(lengths)) > 1:
    print(f"Error: Panjang data tidak sama! {data_sampel.keys()} -> {lengths}")
else:
    # 4. Menjalankan Evaluasi
    dataset = Dataset.from_dict(data_sampel)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevance, context_precision],
        llm=evaluator_llm
    )

    # 5. Eksport Hasil
    df_result = result.to_pandas()
    df_result.to_csv("hasil_evaluasi_ragas_tieng.csv", index=False)
    print("Evaluasi selesai! File 'hasil_evaluasi_ragas_tieng.csv' telah dibuat.")