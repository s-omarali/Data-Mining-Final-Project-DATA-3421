import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("training/xgb_fake_job_model.pkl")

# Example new job posting(s)
data = {
    "job_id": [0],
    "title": ["Software Engineer"],
    "company_profile": ["Leading tech company specializing in AI."],
    "description": ["Looking for an experienced engineer to join our team."],
    "requirements": ["Python, ML, problem-solving skills"],
    "benefits": ["Health insurance, 401k"],
    "department": [""],
    "salary_range": [""],
    "location": [""],
    "telecommuting": [0],
    "has_company_logo": [0],
    "has_questions": [0],
    "employment_type": [""],
    "required_experience": [""],
    "required_education": [""],
    "industry": [""],
    "function": [""]
}

df_new = pd.DataFrame(data)

# Recreate the combined_text column
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df_new["combined_text"] = df_new[text_cols].fillna("").agg(" ".join, axis=1)

# Predict
pred = model.predict(df_new)
pred_proba = model.predict_proba(df_new)[:, 1]

for i, p in enumerate(pred):
    label = "Fake" if p == 1 else "Real"
    print(f"Job {i+1}: {label} (probability of fake: {pred_proba[i]:.2f})")
