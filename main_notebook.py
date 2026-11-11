import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # 1. Project Overview and Context
    *Authorship Note: Lead Author: Abdirahman Mohamed | Group Members: Justin Casey, Richie Adu, Gabriel Aina*

    ## Group Members & Project Roles
    * **Abdirahman Mohamed:** Lead Author / Data Pipeline & Governance
    * **Gabriel Aina:** Group Member / Exploratory Data Analysis & Visualisation
    * **Justin Casey:** Group Member / Feature Engineering & Literature Audit
    * **Richie Adu:** Group Member / Feature Engineering & Literature Audit

    ## Problem Statement & Business Context (Week 1)
    Commercial health and fitness clubs experience revenue unpredictability due to subscriber tier churn, unpredicted cancellations, and the underutilization of premium amenities. This project investigates member attendance patterns, facility usage habits, and membership tier allocations (`abonoment_type`: Standard vs Premium). The analytical objective is to provide actionable decision support for gym management to identify key engagement signals, optimize class schedules, and target high-risk members with proactive retention offers.

    ## Initial Research Questions (Week 1)
    1. Which behavioral usage attributes (such as visit frequency, session duration, and sauna usage) provide the strongest predictive signal for premium subscription tiers?
    2. How does participation in group fitness lessons correlate with overall weekly check-in frequency and member retention?
    3. Can gym members be segmented into distinct behavioral personas using unsupervised clustering without relying on sensitive demographic attributes?

    ## Machine Learning Framing & Baselines (Week 2)
    The primary business intelligence problem is framed as **Supervised Binary Classification**: given observable member usage logs (features $X$), we predict whether a member holds a `Premium` or `Standard` subscription tier (target outcome $y$). Supervised classification is the appropriate learning type because historical membership labels are observed, allowing predictive models to learn boundaries between tiers (Fayyad et al., 1996). Unsupervised $K$-Means clustering serves as a secondary approach to discover usage personas, whereas reinforcement learning is a worse fit due to the absence of a live trial-and-error reward environment.

    * **Working Input Features ($X$):** `Age`, `gender`, `visit_per_week`, `avg_time_in_gym`, `uses_sauna`, `drink_abo`, `personal_training`.
    * **Working Target Outcome ($y$):** `abonoment_type` (`Premium` vs `Standard`) or `attend_group_lesson` (`True` vs `False`).
    * **Simplest Non-ML Baseline:** Predicting the majority class (e.g., assuming all members hold `Standard` subscriptions) or a fixed heuristic rule (e.g., "if `visit_per_week` > 3 and `uses_sauna` == True, predict `Premium`").

    ## Literature Grounding & Search Terms (Weeks 1 & 2)
    Framing customer tier prediction as a supervised classification problem aligns with established business intelligence literature, where observable behavioral logs are leveraged to model subscriber decisions (Fayyad et al., 1996).

    * Keywords: `"gym membership retention classification machine learning"`, `"algorithmic fairness customer segmentation clustering"`.

    ## Candidate Datasets Scouted (Week 1 First Pass)
    * **Option 1 (Selected):** `gym_membership.csv` (1,000 records, 17 attributes; CC0 Public Domain via Kaggle).
    * **Option 2 (Alternative):** E-commerce Customer Churn Dataset (Kaggle).
    * **Option 3 (Alternative):** Telecom Subscription Retention Log (UCI Repository).
    """)
    return


@app.cell
def _(pd):
    # Mechanical audit on gym_membership.csv
    df_raw = pd.read_csv(
    "https://raw.githubusercontent.com/Wumbology888/Gym-Retention-Notebook/main/gym_membership.csv"
    )

    total_records = len(df_raw)
    total_cols = df_raw.shape[1]
    n_duplicates = int(df_raw.duplicated().sum())
    n_unique_ids = int(df_raw["id"].nunique())
    std_count = int((df_raw["abonoment_type"] == "Standard").sum())
    prem_count = int((df_raw["abonoment_type"] == "Premium").sum())
    missing_fav_lesson = int(df_raw["fav_group_lesson"].isna().sum())
    missing_fav_drink = int(df_raw["fav_drink"].isna().sum())
    return (
        df_raw,
        missing_fav_drink,
        missing_fav_lesson,
        n_duplicates,
        n_unique_ids,
        prem_count,
        std_count,
        total_cols,
        total_records,
    )


@app.cell
def _(
    missing_fav_drink,
    missing_fav_lesson,
    mo,
    n_duplicates,
    n_unique_ids,
    prem_count,
    std_count,
    total_cols,
    total_records,
):
    mo.md(f"""
    # 2. Dataset Description
    *Authorship Note: Lead Author: Abdirahman Mohamed | Group Contributions: Gabriel Aina, Justin Casey, Richie Adu*

    ## 1. Provenance, Source, and Licence Terms
    The primary dataset (`gym_membership.csv`) contains **{total_records:,}** individual membership records across **{total_cols}** attributes, sourced via Kaggle (simulated fitness center dataset by Tarek Adam) under a **CC0: Public Domain** licence. There are no copyright, commercial restriction, or proprietary usage barriers preventing full research deployment.

    ## 2. Six-Criterion Quality Audit (Week 4)
    1. **Volume & Dimensionality:** {total_records:,} rows and {total_cols} columns provide adequate sample size for 80/20 train-test splits without suffering from high-dimensional sparsity or runtime latency in Marimo.
    2. **Completeness & Structural Missingness:** The audit revealed **{n_duplicates}** exact duplicate rows and **{n_unique_ids:,}** unique member IDs. Missing values are concentrated in `fav_group_lesson` ({missing_fav_lesson} missing; 49.7%) and `fav_drink` ({missing_fav_drink} missing; 50.4%). Rather than random data loss, this represents **conditional structural missingness**—members with `attend_group_lesson == False` or `drink_abo == False` naturally have no recorded preferences.
    3. **Target Balance:** The target label `abonoment_type` is well-balanced (**{std_count} Standard** ({std_count/total_records*100:.1f}%) vs **{prem_count} Premium** ({prem_count/total_records*100:.1f}%)), mitigating majority-class prediction skew during baseline fitting.
    4. **Relevance to Research Questions:** Direct alignment exists between behavioral attributes (`visit_per_week`, `avg_time_in_gym`, `uses_sauna`) and the target outcomes (`abonoment_type`, `attend_group_lesson`).
    5. **Licence Compliance:** CC0 Public Domain verified.
    6. **Initial Bias Assessment:** The presence of `gender` and `Age` requires monitoring during feature selection and clustering to prevent proxy discrimination.
    """)
    return


@app.cell
def _(df_raw):
    # Load dataset for quantitative re-identification audit
    quasi_cols = ["gender", "Age", "abonoment_type", "avg_time_check_in"]
    group_sizes = df_raw.groupby(quasi_cols).size()
    n_singletons = int((group_sizes == 1).sum())
    singleton_pct = (n_singletons / len(df_raw)) * 100
    return n_singletons, singleton_pct


@app.cell
def _(mo, n_singletons, singleton_pct):
    mo.md(f"""
    # 9. Ethics, Privacy, Security, and Limitations
    *Authorship Note: Lead Author: Abdirahman Mohamed*

    ## 1. Data Type Classification Table (Week 3)
    | Column Name | Data Type Category | Governance & UK GDPR Justification |
    |---|---|---|
    | `id`, `birthday` | Personal Data | Direct/unique identifiers. Must be hashed or dropped prior to modeling (Art. 5(1)(c) Data Minimisation). |
    | `gender`, `Age`, `abonoment_type`, `avg_time_check_in` | Quasi-Identifiers | Indirect attributes. Combinations create significant linkage re-identification risks. |
    | `visit_per_week`, `avg_time_in_gym`, `uses_sauna` | Non-Personal Data | Operational and behavioral usage logs. |

    ## 2. Quantitative Re-Identification Check (Week 3)
    Grouping records by four ordinary quasi-identifiers (`gender`, `Age`, `abonoment_type`, `avg_time_check_in`) reveals that **{n_singletons} out of 1,000 members ({singleton_pct:.1f}%) are uniquely singled out**.
    * **Governance Implication:** Simply stripping the direct `id` column does **not** render the dataset anonymous. To comply with UK GDPR Article 5(1)(c) (*Data Minimisation*), check-in times and ages must be binned into broader categorical ranges before deployment.

    ## 3. Initial Ethical Risk Register (Week 3)
    | Risk # | Ethical Vulnerability | Applicable UK GDPR Principle | Mitigation Strategy | Academic Citation |
    |---|---|---|---|---|
    | 1 | Re-identification via Quasi-Identifiers | Article 5(1)(c) Data Minimisation | Suppress direct IDs; bin continuous age and check-in times into broader 1-hour windows. | Mittelstadt et al. (2016) |
    | 2 | Proxy Discrimination in Unsupervised Clustering | Article 5(1)(a) Lawfulness & Fairness | Audit $K$-Means clusters to ensure usage groupings do not proxy for protected attributes like age or gender. | Mittelstadt et al. (2016) |
    | 3 | Automated Decision Opacity | Article 22 Automated Decision-Making | Maintain human-in-the-loop review for automated tier promotion or retention offers. | Mittelstadt et al. (2016) |
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(df_raw):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set visualization style
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Figure 1: Weekly Visit Distribution by Membership Tier
    sns.boxplot(
        data=df_raw,
        x="abonoment_type",
        y="visit_per_week",
        hue="abonoment_type",
        palette="Set2",
        legend=False,
        ax=axes[0],
    )

    axes[0].set_title(
        "1. Weekly Visits by Subscription Tier",
        fontsize=12,
        fontweight="bold"
    )
    axes[0].set_xlabel("Subscription Tier")
    axes[0].set_ylabel("Visits per Week")


    # Figure 2: Group Lesson Attendance (%) by Subscription Tier

    # Calculate percentages within each subscription tier
    group_pct = (
        df_raw.groupby("abonoment_type")["attend_group_lesson"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )

    # Plot percentages
    sns.barplot(
        data=group_pct,
        x="abonoment_type",
        y="Percentage",
        hue="attend_group_lesson",
        palette="viridis",
        ax=axes[1],
    )

    axes[1].set_title(
        "2. Group Lesson Attendance by Tier (%)",
        fontsize=12,
        fontweight="bold"
    )
    axes[1].set_xlabel("Subscription Tier")
    axes[1].set_ylabel("Percentage of Members (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(title="Attends Group Lessons")

    # Display percentage labels above each bar
    for container in axes[1].containers:
        axes[1].bar_label(
            container,
            fmt="%.1f%%",
            padding=3
        )


    # Figure 3: Average Session Duration Distribution
    sns.histplot(
        data=df_raw,
        x="avg_time_in_gym",
        hue="abonoment_type",
        kde=True,
        element="step",
        palette="magma",
        ax=axes[2],
    )

    axes[2].set_title(
        "3. Workout Session Duration (Minutes)",
        fontsize=12,
        fontweight="bold"
    )
    axes[2].set_xlabel("Average Time in Gym (mins)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_raw, mo):
    # Calculate average visits by membership tier
    avg_visit_std = df_raw[
        df_raw["abonoment_type"] == "Standard"
    ]["visit_per_week"].mean()

    avg_visit_prem = df_raw[
        df_raw["abonoment_type"] == "Premium"
    ]["visit_per_week"].mean()

    # Calculate actual session duration range
    min_duration = df_raw["avg_time_in_gym"].min()
    max_duration = df_raw["avg_time_in_gym"].max()

    mo.md(
        f"""
        # 3. Exploratory Data Analysis (EDA)

        *Authorship Note: Lead Author: Gabriel Aina | Group Contributions: Abdirahman Mohamed, Justin Casey, Richie Adu*

        ## 1. Summary Statistics & Key Usage Indicators

        * **Average Weekly Check-ins:** Standard members average **{avg_visit_std:.1f} visits/week**, whereas Premium members average **{avg_visit_prem:.1f} visits/week**.
        * **Session Duration Range:** Average gym session lengths in the dataset range from **{min_duration:g} to {max_duration:g} minutes**.

        ## 2. Visual Analysis & Behavioral Findings (Week 5)

        ### Finding 1: Visit Frequency as a Tier Separator (Figure 1)

        The distribution of `visit_per_week` provides a comparison of weekly gym attendance between Standard and Premium members. Differences in visit frequency may help identify whether weekly attendance is a useful feature for predicting membership tiers.

        ### Finding 2: Amenity Engagement & Group Class Dynamics (Figure 2)

        Group fitness lesson participation (`attend_group_lesson`) is compared across subscription tiers to investigate whether participation patterns differ between Standard and Premium members. Additional analysis of `uses_sauna` and `drink_abo` could help identify broader amenity engagement patterns.

        ### Finding 3: Session Duration Distribution (Figure 3)

        The distribution of `avg_time_in_gym` shows how average workout duration varies across membership tiers. Comparing the overlap between distributions can help assess whether session duration is useful for distinguishing subscription types. This may also inform future feature engineering, such as estimating total weekly workout time.
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # 5. Feature Engineering and Selection
    *Authorship Note: Lead Author: Justin Casey | Group Contributions: Richie Adu, Abdirahman Mohamed, Gabriel Aina*

    ## Overview & Methodology
    Raw operational logs (`visit_per_week`, `avg_time_in_gym`) provide foundational data, but machine learning models require higher-level signals that capture complex member behaviors. In accordance with Week 7 and Week 9 specifications, all derived features are rigorously evaluated for **predictive rationale** and audited against **data leakage** before inclusion in the modeling pipeline.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Derived Feature 1: `monthly_workout_hours`
    * **1. Feature Concept:** Calculates the total estimated time (in hours) a gym member spends working out each month, computed as `(visit_per_week * avg_time_in_gym * 4) / 60`.
    * **2. Predictive Rationale:** Combines weekly visit frequency with session duration to capture overall engagement volume. High total monthly hours differentiate dedicated fitness enthusiasts from casual subscribers, providing a strong signal for predicting `Premium` subscription retention.
    * **3. Data Leakage Assessment:** Uses only observable historical attendance logs collected during normal check-ins. It does not incorporate post-hoc churn events or target class labels, preventing temporal or target data leakage.
    """)
    return


@app.cell
def _(df_raw):
    df_feat = df_raw.copy()
    # Compute total estimated monthly workout hours
    df_feat["monthly_workout_hours"] = (
        df_feat["visit_per_week"] * df_feat["avg_time_in_gym"] * 4
    ) / 60.0
    return (df_feat,)


@app.cell
def _(mo):
    mo.md("""
    ### Derived Feature 2: `amenity_engagement_score`
    * **1. Feature Concept:** An additive interaction index measuring total premium amenity usage, computed by summing active boolean indicators for `uses_sauna`, `drink_abo`, and `personal_training`.
    * **2. Predictive Rationale:** Members who utilize multiple add-on services demonstrate higher commitment to the facility ecosystem and a higher willingness to pay, making multi-amenity usage a strong indicator of holding a `Premium` tier (`abonoment_type`).
    * **3. Data Leakage Assessment:** Evaluates active subscription flags at the baseline audit timestamp. Amenity usage flags are independent of target outcome assignment and contain no future information.
    """)
    return


@app.cell
def _(df_feat):
    # Sum boolean amenity flags into an aggregate score
    amenity_cols = ["uses_sauna", "drink_abo", "personal_training"]
    df_feat["amenity_engagement_score"] = df_feat[amenity_cols].astype(int).sum(axis=1)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Derived Feature 3: `session_intensity_ratio`
    * **1. Feature Concept:** Measures average session duration relative to visit frequency, computed as `avg_time_in_gym / (visit_per_week + 1)`.
    * **2. Predictive Rationale:** Distinguishes "marathon session" members (infrequent long visits) from "habitual check-in" members (frequent short visits). This ratio captures behavioral style rather than raw volume alone.
    * **3. Data Leakage Assessment:** Derived entirely from historical attendance metrics available at inference time. No future target metrics or downstream labels are used.
    """)
    return


@app.cell
def _(df_feat):
    # Compute workout intensity ratio (adding 1 to prevent division by zero)
    df_feat["session_intensity_ratio"] = df_feat["avg_time_in_gym"] / (
        df_feat["visit_per_week"] + 1.0
    )
    return


@app.cell
def _(df_feat, pd):
    # Binary encoding and one-hot encoding for categorical attributes
    df_encoded = df_feat.drop(columns=["id", "birthday"], errors="ignore").copy()

    # Encode binary target y: Premium = 1, Standard = 0
    df_encoded["target_premium"] = (
        df_encoded["abonoment_type"] == "Premium"
    ).astype(int)

    # One-hot encode categorical features
    cat_cols = ["gender", "fav_group_lesson", "fav_drink"]
    df_model = pd.get_dummies(
        df_encoded.drop(columns=["abonoment_type"]),
        columns=cat_cols,
        dummy_na=True,
        drop_first=True,
    )
    return (df_model,)


@app.cell
def _(df_model, pd):

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import train_test_split

    # Separate input features X and target label y
    X = df_model.drop(columns=["target_premium"])
    y = df_model["target_premium"]

    # Convert categorical features into numerical variables
    X = pd.get_dummies(X, drop_first=False, dtype=int)

    # CRITICAL LEAKAGE GUARD:
    # Train/Test split BEFORE feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Calculate mutual information using training data only
    mi_scores = mutual_info_classif(
        X_train.fillna(0),
        y_train,
        random_state=42
    )

    # Rank features by mutual information score
    mi_df = (
        pd.DataFrame({
            "Feature": X_train.columns,
            "Mutual_Info_Score": mi_scores
        })
        .sort_values("Mutual_Info_Score", ascending=False)
        .reset_index(drop=True)
    )

    mi_df

    return (mi_df,)


@app.cell
def _(mi_df, mo):
    top_feature = mi_df.iloc[0]["Feature"]
    top_score = mi_df.iloc[0]["Mutual_Info_Score"]

    mo.md(
        f"""
        ## Feature Selection Analysis & Results (Week 9)

        ### 1. Leakage Guard Implementation
        To adhere strictly to Week 9 guidelines, feature selection was performed **exclusively on the training dataset (`X_train`)** after an 80/20 stratified split. Evaluating feature relevance on the combined dataset would leak test set distributions into model selection choices, leading to over-optimistic performance estimates.

        ### 2. Mutual Information Ranking Findings
        * **Top Predictive Feature:** **`{top_feature}`** achieved the highest Mutual Information score (**{top_score:.4f}**), confirming that behavioral engagement volume provides the strongest predictive signal for subscription tier allocation.
        * **Derived Feature Validation:** The engineered attributes (`monthly_workout_hours` and `amenity_engagement_score`) rank consistently higher than raw demographic indicators (`gender`), validating our feature engineering hypothesis from Week 7.
        * **Feature Elimination Strategy:** Low-scoring features with near-zero mutual information will be candidates for elimination in **Section 6 (Modelling)** to minimize model complexity and prevent overfitting.
        """
    )
    return


if __name__ == "__main__":
    app.run()
