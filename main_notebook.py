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
    *Authorship Note: Lead Author: [Abdirahman Mohamed] | Group Members: [Justin Casey, Richie Adu, Gabirel Aina]*

    ## Group Members & Project Roles
    * **[Abdirahman Mohamed]:** Lead Author / Data Pipeline & Governance
    * **[Justin Casey]:** Group Member / Exploratory Data Analysis & Visualisation
    * **[Gabriel Aina]:** Group Member / Feature Engineering & Literature Audit
    * **[Richie Adu]:** Group Member / Feature Engineering & Literature Audit

    ## Problem Statement & Business Context (Week 1)
    Commercial health and fitness clubs experience revenue unpredictability due to subscriber tier churn, unpredicted cancellations, and the underutilization of premium amenities. This project investigates member attendance patterns, facility usage habits, and membership tier allocations (`abonoment_type`: Standard vs Premium). The analytical objective is to provide actionable decision support for gym management to identify key engagement signals, optimize class schedules, and target high-risk members with proactive retention offers.

    ## Initial Research Questions (Week 1)
    1. Which behavioral usage attributes (such as visit frequency, session duration, and sauna usage) provide the strongest predictive signal for premium subscription tiers?
    2. How does participation in group fitness lessons correlate with overall weekly check-in frequency and member retention?
    3. Can gym members be segmented into distinct behavioral personas using unsupervised clustering without relying on sensitive demographic attributes?

    ## Machine Learning Framing & Baselines (Week 2)
    The primary business intelligence problem is framed as **Supervised Binary Classification**: given observable member usage logs (features $X$), we predict whether a member holds a `Premium` or `Standard` subscription tier (target outcome $y$). Supervised classification is the appropriate learning type because historical membership labels are observed, allowing predictive models to learn boundaries between tiers (Fayyad et al., 1996). Unsupervised $K$-Means clustering serves as a secondary approach to discover usage personas, whereas reinforcement learning is a worse fit due to the absence of a live trial-and-error reward environment.

    * **Working Input Features ($X$):** `Age`, `gender`, `visit_per_week`, `avg_time_in_gym`, `uses_sauna`, `drink_abo`, `personal_training` (provisional until Week 4 audit).
    * **Working Target Outcome ($y$):** `abonoment_type` (`Premium` vs `Standard`) or `attend_group_lesson` (`True` vs `False`).
    * **Simplest Non-ML Baseline:** Predicting the majority class (e.g., assuming all members hold `Standard` subscriptions) or a simple business heuristic rule (e.g., "if `visit_per_week` > 3 and `uses_sauna` == True, predict `Premium`").

    ## Literature Grounding & Search Terms (Weeks 1 & 2)
    Framing customer tier prediction as a supervised classification problem aligns with established business intelligence literature, where observable behavioral logs are leveraged to model subscriber decisions (Fayyad et al., 1996).

    * Keywords: `"gym membership retention classification machine learning"`, `"algorithmic fairness customer segmentation clustering"`.

    ## Candidate Datasets Scouted (Week 1 First Pass)
    * **Option 1 (Primary Candidate):** `gym_membership.csv` (1,000 records, 17 attributes covering member check-ins, lesson attendance, and membership tiers; CC0 Public Domain via Kaggle).
    * **Option 2 (Alternative):** E-commerce Customer Churn Dataset (Kaggle).
    * **Option 3 (Alternative):** Telecom Subscription Retention Log (UCI Repository).
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
 
    """)
    return


if __name__ == "__main__":
    app.run()
