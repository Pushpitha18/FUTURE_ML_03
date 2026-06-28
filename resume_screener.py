import re
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

job_description = """
We are looking for a Data Scientist with strong skills in Python, Machine Learning, 
SQL, and Data Visualization. Experience with Deep Learning and Git is a plus.
"""

resumes_data = {
    "Candidate_Name": ["Alice Sharma", "Bob Kumar", "Charlie Singh"],
    "Resume_Text": [
        "Data Scientist with 2 years experience. Expert in Python, Machine Learning, SQL, and data visualization using Tableau.",
        "Software Engineer skilled in Java, C++, Python, Web Development, and Git. Looking for developer roles.",
        "Experienced analyst. Proficient in Excel, SQL, Tableau, PowerBI, and basic Python data analysis."
    ]
}

df = pd.DataFrame(resumes_data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

cleaned_job_desc = clean_text(job_description)
df['Cleaned_Resume'] = df['Resume_Text'].apply(clean_text)

SKILL_BANK = ["python", "machine learning", "sql", "data visualization", "deep learning", "git", "java", "c++", "tableau", "excel"]

def extract_skills(text):
    extracted = []
    for skill in SKILL_BANK:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            extracted.append(skill)
    return extracted

job_skills = extract_skills(job_description)

all_texts = [cleaned_job_desc] + df['Cleaned_Resume'].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

df['Match_Score'] = np.round(similarity_scores * 100, 2)
df['Extracted_Skills'] = df['Resume_Text'].apply(extract_skills)

def find_missing_skills(candidate_skills):
    return [skill for skill in job_skills if skill not in candidate_skills]

df['Missing_Skills'] = df['Extracted_Skills'].apply(find_missing_skills)
ranked_df = df.sort_values(by='Match_Score', ascending=False)

print("\n==== JOB SCREENING REPORT ====")
print(f"Target Job Skills: {', '.join(job_skills).upper()}\n")

for index, row in ranked_df.iterrows():
    print(f"Candidate: {row['Candidate_Name']}")
    print(f" -> Match Score: {row['Match_Score']}%")
    print(f" -> Found Skills: {', '.join(row['Extracted_Skills'])}")
    if row['Missing_Skills']:
        print(f" ⚠️ Missing Skills: {', '.join(row['Missing_Skills'])}")
    else:
        print(" -> 🎉 Perfect Skill Match!")
    print("-" * 40)