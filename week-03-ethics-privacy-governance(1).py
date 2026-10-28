# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Week 3 companion notebook: ethics, privacy, and data governance

    This notebook applies this week's techniques to the running example,
    **Streamline Media**, a fictional UK streaming service whose BI team wants
    to predict subscriber churn (subscribers cancelling their subscription).
    It carries out three tasks that correspond to the work you will do in
    **section 9** of your group's main notebook:

    1. It classifies every column by GDPR data type.
    2. It measures how far a few ordinary attributes can identify an individual
       subscriber (re-identification risk).
    3. It drafts an initial ethical risk register.

    Run this notebook and read the explanations. Then do the same work on
    **your own dataset** in section 9 of the main notebook. This companion is a
    working notebook. The polished, cited ethics record belongs in the main
    notebook.
    """)
    return


@app.cell(hide_code=True)
def _(np, pd):
    def generate_streamline_data(n=2000, seed=42):
        """Synthetic Streamline Media subscriber dataset (reproducible)."""
        rng = np.random.default_rng(seed)
        tenure_months = rng.integers(1, 49, n)
        plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
        fee_map = {"Basic": 6.99, "Standard": 9.99, "Premium": 14.99}
        monthly_fee = np.array([fee_map[p] for p in plan])
        watch_hours = np.clip(
            rng.gamma(2.0, 6.0, n) - tenure_months * 0.05, 0, None
        )
        support_tickets = rng.poisson(0.6, n)
        satisfaction = np.clip(
            rng.normal(7, 2, n) - support_tickets * 0.4, 1, 10
        )
        logit = (
            0.5 + 0.45 * support_tickets + 0.12 * monthly_fee
            - 0.10 * watch_hours - 0.35 * satisfaction
        )
        churn_prob = 1 / (1 + np.exp(-logit))
        churned = rng.binomial(1, churn_prob)
        return pd.DataFrame(
            {
                "customer_id": [f"C{100000 + i}" for i in range(n)],
                "tenure_months": tenure_months,
                "plan": plan,
                "monthly_fee": monthly_fee,
                "watch_hours_month": np.round(watch_hours, 1),
                "support_tickets_90d": support_tickets,
                "satisfaction_score": np.round(satisfaction, 1),
                "region": rng.choice(
                    ["South West", "London", "Midlands", "Scotland"], n
                ),
                "churned": churned,
            }
        )

    return (generate_streamline_data,)


@app.cell(hide_code=True)
def _(generate_streamline_data):
    df = generate_streamline_data()
    return (df,)


@app.cell(hide_code=True)
def _(df, mo):
    mo.md(f"""
    ### The data are synthetic

    The **{len(df):,}** rows in this dataset describe **no real people**. The
    generator creates the subscribers using a fixed random seed. The data
    therefore contain no personal data, so no lawful basis under GDPR Art. 6
    is needed (a lawful basis is the legal justification for processing
    personal data), and there is no licensing issue. The module uses
    synthetic data for this reason.

    Your own project dataset is not synthetic. When you work with data about
    real individuals, GDPR applies in full. You need a lawful basis, and you
    must follow the Art. 5 principles. The re-identification risks below
    then become real obligations. In this notebook, the Streamline columns
    are classified as if they described real subscribers, so that you can
    apply the same method to your own data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 1: classify every column by GDPR data type

    There are four categories:

    - **Personal data** (Art. 4(1) GDPR): any information relating to an
      identified or identifiable person.
    - **Special category data** (Art. 9 GDPR): data about race, ethnicity,
      health, religion, biometrics, sexual orientation and similar topics. It
      requires an additional lawful basis and stronger safeguards.
    - **Quasi-identifier**: an attribute that does not identify a person alone,
      but can do so when combined with other attributes.
    - **Non-personal data**: data that cannot be linked to a person.

    | Column | Classification | Why |
    |---|---|---|
    | `customer_id` | Identifier (personal data) | Identifies one subscriber. Treat it as a pseudonymous key, not as anonymous |
    | `region` | Quasi-identifier | Broad on its own; combined with tenure and plan it narrows to a few people |
    | `tenure_months` | Quasi-identifier | Account attribute that narrows groups further |
    | `plan` | Quasi-identifier | Many subscribers share a plan, but it can identify in combination |
    | `monthly_fee` | Personal data | Account attribute linked to the subscriber |
    | `watch_hours_month` | Personal data (behavioural) | Shows how the subscriber uses the service |
    | `support_tickets_90d` | Personal data (behavioural) | Shows how often the subscriber contacts support |
    | `satisfaction_score` | Personal data (opinion) | The subscriber's own rating of the service |
    | `churned` | Personal data (outcome) | A business outcome linked to a person |

    No column in this dataset is **special category data**. However, the table
    does not cover the model's predicted churn output. If that output is combined
    with `region`, it could act as a proxy for protected characteristics, because
    region is related to demographic and socioeconomic characteristics. A proxy
    is a variable that indirectly reveals a characteristic it is not meant to
    record. This risk is recorded in the register below.
    """)
    return


@app.cell(hide_code=True)
def _(df, pd):
    link_key = df.groupby(["region", "plan", "tenure_months"]).size().rename("group_size")
    link_summary = pd.DataFrame(
        {
            "distinct_combinations": [len(link_key)],
            "singleton_combinations": [int((link_key == 1).sum())],
            "min_group_size": [int(link_key.min())],
            "max_group_size": [int(link_key.max())],
        }
    )
    link_summary
    return (link_key,)


@app.cell(hide_code=True)
def _(df, link_key, mo):
    singletons = int((link_key == 1).sum())
    link_share = singletons / len(df)
    mo.md(
        f"""
        ### Step 2: re-identification using three ordinary attributes

        Re-identification means linking a record in a dataset to a named
        individual. This step groups subscribers by `region`, `plan`, and
        `tenure_months`. In **{link_share:.1%}** of cases ({singletons} of
        {len(df):,} subscribers), a subscriber is the only member of their
        combination. These subscribers can be singled out using three attributes
        that contain no name and no obvious identifier.

        This check uses **k-anonymity**. A dataset is k-anonymous for a set of
        columns if every combination of values appears at least k times. Here the
        smallest group has size 1, so the dataset is only 1-anonymous for these
        three columns. Subscribers who are alone in their group are the main
        concern. They could be re-identified if the data were leaked or linked
        to another source.
        """
    )
    return


@app.cell(hide_code=True)
def _(df, mo):
    fine_key = df.groupby(["region", "plan", "tenure_months", "watch_hours_month"]).size()
    fine_singletons = int((fine_key == 1).sum())
    fine_share = fine_singletons / len(df)
    mo.md(
        f"""
        ### Adding a fourth column increases re-identification risk

        When `watch_hours_month`, an ordinary behavioural measure, is added to the
        grouping, **{fine_share:.0%}** of subscribers are alone in their
        combination. The conclusion is not that this column should be removed.
        The conclusion is that quasi-identifiers add to each other's identifying
        power. The absence of names in a file does not make it anonymous.
        Anonymity must be measured, and it decreases as more detailed attributes
        are kept. This is the issue addressed by the data minimisation principle
        in Art. 5(1)(c) GDPR, which requires that personal data be limited to
        what is necessary.

        For your own dataset, run the same check on the columns most likely to
        identify individuals. Report the share of unique combinations in
        section 9.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 3: the initial ethical risk register

    A risk register is a table that lists each risk. It needs at least three
    columns: the risk, the GDPR principles it relates to, and a source that
    supports it. This draft uses this week's seeding reference, Mittelstadt et
    al. (2016), "The Ethics of Algorithms: Mapping the Debate", *Big Data and
    Society*, and the GDPR articles. In your main notebook, replace the source
    column with the papers that you verified this week, one for each risk.

    | # | Risk | GDPR principle(s) | Source |
    |---|---|---|---|
    | 1 | Subscriber-level churn predictions feeding automated retention offers or pricing (automated decision-making / profiling) | Art. 22 (automated decision-making incl. profiling); Art. 5(1)(a) (lawfulness, fairness, transparency) | Mittelstadt et al. (2016): algorithms make consequential decisions that are hard to scrutinise |
    | 2 | Re-identification of subscribers from quasi-identifiers (region, plan and tenure identify a measurable share) | Art. 5(1)(c) (data minimisation); Recital 26 (singling out) | Mittelstadt et al. (2016): concerns about identifiability and opacity; see the check above |
    | 3 | Unfair outcomes for a region or plan group, or proxy discrimination against protected characteristics | Art. 5(1)(a) (fairness); Art. 9 if combined with special category proxies | Mittelstadt et al. (2016): unfair outcomes as a type of ethical concern |

    Each row must answer two questions before it is complete. How would we know
    that this risk is real in our data? What would we measure to detect it? If
    you cannot answer both, the risk is still an assumption and not yet a
    register entry.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Your task: section 9 of the main notebook

    Repeat these three steps on **your own dataset**. Write the polished
    narrative into section 9 of your group's main notebook:

    - Classify every column by GDPR data type, with a one-line reason for each.
    - Run a re-identification check and report the share of unique combinations.
    - Record the top three risks, the GDPR principles they relate to, and one
      verified academic source for each risk, cited inline.

    Section 9 is developed over several weeks. It opens this week and is
    extended each week until week 19, when you complete the privacy, security,
    and limitations analysis.

    This companion notebook is a working notebook. It is not submitted.
    """)
    return


if __name__ == "__main__":
    app.run()
