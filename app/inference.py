import joblib
import pandas as pd
import numpy as np
import re
import ast
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class CareerAI:
    def __init__(self, model_dir="models", data_dir="data/processed/Clean Dataset"):
        """
        Inisialisasi CareerAI dengan memuat model dan data yang diperlukan.
        Pastikan struktur folder sesuai dengan path default atau sesuaikan saat instansiasi.
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        # Path file artifact
        self.paths = {
            "model": os.path.join(model_dir, "best_model.joblib"),
            "tfidf": os.path.join(model_dir, "tfidf_vectorizer.joblib"),
            "encoder": os.path.join(model_dir, "label_encoder.joblib"),
            "skill_vec": os.path.join(model_dir, "skill_vectorizer.joblib"),
            "dataset": os.path.join(data_dir, "cleaned_data.csv")
        }

        # Load resources
        print("Loading models and data...")
        try:
            self.model = joblib.load(self.paths["model"])
            self.tfidf = joblib.load(self.paths["tfidf"])
            self.le = joblib.load(self.paths["encoder"])
            self.skill_vect = joblib.load(self.paths["skill_vec"])
            self.df = pd.read_csv(self.paths["dataset"])
            
            # Persiapan Matrix Skill untuk fitur Rekomendasi
            self.skill_matrix = self._prepare_skill_matrix()
            print("All resources loaded successfully.")
            
        except FileNotFoundError as e:
            print(f"Error: File tidak ditemukan. Pastikan path benar.\nDetail: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Membersihkan teks input (lowercase, regex, strip)."""
        txt = str(text).lower()
        txt = re.sub(r"http\S+|www\S+", "", txt) 
        txt = re.sub(r"[^a-zA-Z0-9 ]", " ", txt) 
        txt = re.sub(r"\s+", " ", txt).strip() 
        return txt

    def _prepare_skill_matrix(self):
        """Membuat matrix skill dari dataset untuk pencarian kemiripan."""
        # Fungsi parsing string list "['a', 'b']" menjadi list python
        def parse_tokens(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except (ValueError, SyntaxError):
                return []

        # Parse dan gabungkan skill dengan underscore (misal: "data science" -> "data_science")
        # Ini penting agar cocok dengan vocabulary vectorizer
        skills_text = self.df["skills_token"].apply(parse_tokens).apply(
            lambda x: " ".join([str(s).replace(" ", "_") for s in x])
        )
        
        # Transform ke TF-IDF Matrix
        return self.skill_vect.transform(skills_text)

    def predict_job_role(self, user_text, top_k=3):
        """
        Memprediksi pekerjaan berdasarkan deskripsi input user.
        """
        clean_input = self._clean_text(user_text)
        vec = self.tfidf.transform([clean_input])

        # Cek apakah model support predict_proba (MLP) atau decision_function (SVM/Linear)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            top_indices = np.argsort(probs)[::-1][:top_k]
            scores = probs[top_indices]
        else:
            decision = self.model.decision_function(vec)
            if decision.ndim == 1:
                decision = [decision]
            top_indices = np.argsort(decision[0])[::-1][:top_k]
            # Softmax sederhana untuk mengubah decision value menjadi pseudo-probability
            exp_scores = np.exp(decision[0][top_indices])
            scores = exp_scores / np.sum(exp_scores)

        labels = self.le.inverse_transform(top_indices)

        results = []
        for label, score in zip(labels, scores):
            results.append({
                "role": label,
                "confidence_score": float(score),
                "confidence_percent": f"{score*100:.2f}%"
            })
            
        return results

    def recommend_skills(self, user_skills_str, top_jobs=50, top_n_recommendations=10):
        """
        Merekomendasikan skill berdasarkan input skill user menggunakan Cosine Similarity.
        """
        # Preprocessing input skill user
        clean_input = self._clean_text(user_skills_str)
        clean_input = clean_input.replace(" ", "_") # Samakan format dengan training data
        
        user_vec = self.skill_vect.transform([clean_input])
        
        # Hitung kemiripan dengan database
        similarity = cosine_similarity(user_vec, self.skill_matrix).ravel()
        
        # Ambil indeks job yang paling mirip skill-set nya
        top_indices = np.argsort(similarity)[::-1][:top_jobs]
        
        # Ambil skill dari job-job mirip tersebut
        candidate_skills = self.df.iloc[top_indices]["skills_token"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        ).tolist()
        
        # Hitung frekuensi kemunculan skill
        skill_counter = Counter()
        for skill_list in candidate_skills:
            # Kembalikan underscore ke spasi untuk output yang mudah dibaca
            readable_skills = [s.replace("_", " ") for s in skill_list]
            skill_counter.update(readable_skills)
            
        # Filter: Jangan rekomendasikan skill yang sudah dimiliki user
        user_tokens = set(user_skills_str.lower().split(","))
        user_tokens = {t.strip() for t in user_tokens} # Clean whitespace
        
        final_recommendations = []
        for skill, count in skill_counter.most_common():
            if skill not in user_tokens:
                final_recommendations.append({"skill": skill, "relevance_count": count})
                if len(final_recommendations) >= top_n_recommendations:
                    break
                    
        return final_recommendations

# =============================================================================
# MAIN EXECUTION (CONTOH PENGGUNAAN)
# =============================================================================
if __name__ == "__main__":
    # Konfigurasi path (Sesuaikan dengan lokasi file Anda)
    # Asumsi: script ini dijalankan sejajar dengan folder 'models' dan 'data'
    MODEL_DIR = "models" 
    DATA_DIR = "data/processed/Clean Dataset"

    # Pastikan file ada sebelum menjalankan (untuk testing dummy)
    if os.path.exists(MODEL_DIR) and os.path.exists(DATA_DIR):
        
        # Inisialisasi System
        ai_system = CareerAI(model_dir=MODEL_DIR, data_dir=DATA_DIR)

        print("\n" + "="*50)
        print("TEST 1: PREDIKSI KARIR")
        print("="*50)
        
        sample_profile = """
        I have 3 years of experience in data analysis using Python, Pandas, and SQL. 
        I also create dashboards using Tableau and have knowledge of Machine Learning algorithms.
        """
        print(f"Input Profile: {sample_profile.strip()}\n")
        
        career_preds = ai_system.predict_job_role(sample_profile)
        for i, pred in enumerate(career_preds, 1):
            print(f"{i}. {pred['role'].upper()} (Confidence: {pred['confidence_percent']})")

        print("\n" + "="*50)
        print("TEST 2: REKOMENDASI SKILL")
        print("="*50)
        
        sample_skills = "python, sql, data analysis"
        print(f"Current Skills: {sample_skills}\n")
        
        skill_recs = ai_system.recommend_skills(sample_skills)
        print("Recommended Skills to Learn:")
        for i, rec in enumerate(skill_recs, 1):
            print(f"{i}. {rec['skill']} (Found in {rec['relevance_count']} similar profiles)")
            
    else:
        print("⚠️ Directory 'models' atau 'data' tidak ditemukan.")
        print("Pastikan struktur folder sudah benar sebelum menjalankan script ini.")