import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 1. Project Overview and Context
    *Authorship Note: Lead Author: [Your Name] | Group Members: [Teammate Names]*

    ## Group Members & Project Roles
    * **[Your Name]:** Lead Author / Data Pipeline & Governance
    * **[Teammate 1]:** Group Member / Exploratory Data Analysis & Visualisation
    * **[Teammate 2]:** Group Member / Feature Engineering & Literature Audit

    ## Problem Statement & Business Context
    Commercial health and fitness clubs experience revenue unpredictability due to subscriber tier churn, unpredicted cancellations, and the underutilization of premium amenities. This project investigates member attendance patterns, facility usage habits, and membership tier allocations (`abonoment_type`: Standard vs Premium). The analytical objective is to provide actionable decision support for gym management to identify key engagement signals, optimize class schedules, and target high-risk members with proactive retention offers.

    ## Initial Research Questions
    1. Which behavioral usage attributes (such as visit frequency, session duration, and sauna usage) provide the strongest predictive signal for premium subscription tiers?
    2. How does participation in group fitness lessons correlate with overall weekly check-in frequency and member retention?
    3. Can gym members be segmented into distinct behavioral personas using unsupervised clustering without relying on sensitive demographic attributes?

    ## Initial Literature Search Keywords
    * `"gym membership churn prediction machine learning"`
    * `"customer segmentation health clubs business intelligence"`
    * `"algorithmic fairness subscriber classification"`

    ## Candidate Datasets Scouted (Week 1 First Pass)
    * **Option 1 (Primary Candidate):** `gym_membership.csv` (1,000 records, 17 attributes covering member check-ins, lesson attendance, and membership tiers; CC0 Public Domain via Kaggle).
    * **Option 2 (Alternative):** E-commerce Customer Churn Dataset (Kaggle).
    * **Option 3 (Alternative):** Telecom Subscription Retention Log (UCI Repository).
    """)
    return


if __name__ == "__main__":
    app.run()
