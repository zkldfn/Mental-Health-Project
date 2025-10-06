from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Insight dan rekomendasi
recommendation_map = {
    "Anxiety": {
        "sentiment": "Negatif",
        "insight": "Tanda-tanda kecemasan seperti kekhawatiran berlebihan.",
        "recommendation": "Coba teknik relaksasi seperti pernapasan dalam, journaling, atau konsultasi."
    },
    "Bipolar": {
        "sentiment": "Negatif",
        "insight": "Mood yang sangat fluktuatif antara fase tinggi dan rendah.",
        "recommendation": "Pantau perubahan mood dan pertimbangkan konsultasi psikiater."
    },
    "Depression": {
        "sentiment": "Negatif",
        "insight": "Indikasi perasaan sedih, hampa, dan kehilangan minat.",
        "recommendation": "Luangkan waktu untuk istirahat, dan pertimbangkan berbicara dengan profesional."
    },
    "Suicidal": {
        "sentiment": "Negatif",
        "insight": "Mengandung sinyal berbahaya yang memerlukan perhatian serius.",
        "recommendation": "Segera hubungi layanan bantuan profesional atau darurat."
    },
    "Stress": {
        "sentiment": "Negatif",
        "insight": "Menunjukkan tekanan mental yang tinggi atau beban emosional.",
        "recommendation": "Coba olahraga ringan, manajemen waktu, atau aktivitas yang menenangkan."
    },
    "Personality disorder": {
        "sentiment": "Netral",
        "insight": "Muncul pola perilaku tetap yang memengaruhi relasi dan keseharian.",
        "recommendation": "Diperlukan evaluasi psikologis lanjutan untuk diagnosis akurat."
    },
    "Normal": {
        "sentiment": "Positif",
        "insight": "Tidak ditemukan gejala gangguan mental yang mencolok.",
        "recommendation": "Tetap jaga pola hidup sehat dan hubungan sosial yang baik."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['input_text']
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]

        info = recommendation_map.get(prediction, {
            "sentiment": "Tidak diketahui",
            "insight": "Data tidak tersedia.",
            "recommendation": "Silakan coba masukan yang lain."
        })

        return render_template('index.html',
                               input_text=user_input,
                               prediction_text=f'Hasil Deteksi: {prediction}',
                               sentiment_text=info["sentiment"],
                               insight_text=info["insight"],
                               recommendation_text=info["recommendation"])

if __name__ == '__main__':
    app.run(debug=True)
