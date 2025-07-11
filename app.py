from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import os
import spacy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

skill_keywords = {
    "python", "java", "c++", "machine learning", "data analysis", "pandas",
    "numpy", "scikit-learn", "api", "aws", "cloud", "sql", "django"
}

def extract_text_from_pdf(file_storage):
    with pdfplumber.open(file_storage) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)

def extract_skills(text):
    text = text.lower()
    return set(skill for skill in skill_keywords if skill in text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_and_rank():
    job_file = request.files.get('job_description')
    resume_files = request.files.getlist('resumes')

    if not job_file or not resume_files:
        return jsonify({"error": "Missing job or resume file"}), 400

    job_text = extract_text_from_pdf(job_file)
    job_embedding = model.encode([job_text])[0]
    job_skills = extract_skills(job_text)

    results = []

    for file in resume_files:
        resume_text = extract_text_from_pdf(file)
        resume_embedding = model.encode([resume_text])[0]
        similarity = cosine_similarity([job_embedding], [resume_embedding])[0][0]
        resume_skills = extract_skills(resume_text)
        matched = resume_skills & job_skills
        missing = job_skills - resume_skills

        results.append({
            "filename": file.filename,
            "score": round(float(similarity), 4),
            "matched_skills": ", ".join(matched) if matched else "None",
            "missing_skills": ", ".join(missing) if missing else "None"
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
