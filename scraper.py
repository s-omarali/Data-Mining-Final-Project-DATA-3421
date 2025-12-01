from playwright.sync_api import sync_playwright
import pandas as pd
import joblib

# Load your trained XGBoost model
model = joblib.load("training/xgb_fake_job_model.pkl")

# Columns needed for your pipeline
required_columns = [
    "title", "company_profile", "description", "requirements", "benefits",
    "department", "salary_range", "location", "telecommuting",
    "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function"
]

# Scraping function
def scrape_jobs(url, max_jobs=10):
    jobs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        # Adjust selector based on the website
        job_cards = page.query_selector_all(".job-card-container")
        
        for i, card in enumerate(job_cards):
            if i >= max_jobs:
                break
            title = card.query_selector(".job-card-list__title").inner_text().strip()
            company = card.query_selector(".job-card-container__company-name").inner_text().strip()
            description = card.query_selector(".job-card-container__description").inner_text().strip()
            
            jobs.append({
                "title": title,
                "company_profile": company,
                "description": description,
                "requirements": "",
                "benefits": "",
                "department": "",
                "salary_range": "",
                "location": "",
                "telecommuting": 0,
                "has_company_logo": 0,
                "has_questions": 0,
                "employment_type": "",
                "required_experience": "",
                "required_education": "",
                "industry": "",
                "function": ""
            })
        
        browser.close()
    return pd.DataFrame(jobs)

# Example usage
url = "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=United%20States"
df_jobs = scrape_jobs(url, max_jobs=5)
print(df_jobs.head())
print(df_jobs.columns)


# Combine text for your pipeline
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
print([col for col in text_cols if col not in df_jobs.columns])
df_jobs[text_cols] = df_jobs[text_cols].fillna("")
df_jobs["combined_text"] = df_jobs[text_cols].fillna("").agg(" ".join, axis=1)

# Make predictions
pred = model.predict(df_jobs)
pred_proba = model.predict_proba(df_jobs)[:, 1]

# Print results
for i, p in enumerate(pred):
    label = "Fake" if p == 1 else "Real"
    print(f"{df_jobs['title'][i]}: {label} (probability of fake: {pred_proba[i]:.2f})")
