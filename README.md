
# Algorithmic Audit Framework for Black-Box Ranking Systems

This repository contains a modular analytics engineering and diagnostic machine learning pipeline designed to audit black-box marketplace sorting algorithms. E-commerce platforms frequently deploy undocumented, non-linear optimization constraints ("shadow penalties") that suppress merchant visibility despite strong performance metrics. This framework identifies, localizes, and validates these hidden systemic constraints using a combination of relational database engineering, machine learning surrogate modeling, game-theoretic feature attribution, and automated change-point inference.

---

## 🏗️ Repository Architecture

The project strictly follows production-grade modular design patterns, separating data ingestion, engineering, estimation, and discovery into dedicated layers:

```text
shadow_ranking_project/
│
├── data/                               # data persistence & visualization layer
│   ├── marketplace_data.db             # local embedded duckdb storage instance
│   ├── shadow_threshold_audit.png      # SHAP localized dependence plot
│   ├── distributional_rank_audit.png    # baseline empirical density profile
│   └── discovered_change_point_audit.png # data-driven threshold discovery visualization
│
├── src/                                # reusable core logic package modules
│   ├── __init__.py
│   ├── database.py                     # schema design & statistical parameter mirroring
│   ├── feature_engineering.py          # preprocessing & stratified validation splits
│   ├── audit_model.py                  # tree ensemble estimation & global evaluation
│   ├── explainability.py               # local game-theoretic feature attribution
│   └── statistical_audit.py            # inferential statistics & population segmentation
│
├── main.py                             # centralized operational pipeline controller
├── explore.py                          # data distribution profiling sandbox
└── discover_threshold.py               # automated change-point discovery engine

```

---

## 📊 Analytics Engineering Strategy

To enforce computational efficiency and eliminate network I/O constraints within restricted runtime environments, this pipeline implements **Statistical Parameter Mirroring** instead of directly loading external multi-gigabyte flat files. The database generator maps the empirical characteristics of the **Olist Brazilian E-Commerce Dataset** directly into a localized relational database:

* **Lognormal Pricing Distribution:** Captures high-variance, fat-tailed marketplace pricing structures using an anchored lognormal distribution ($\text{clip}[5.0, 800.0]$).
* **Categorical Transaction Weights:** Models realistic volume imbalances across premium product segments (`health_beauty`, `computers_accessories`, `watches_gifts`, `sports_leisure`, `auto`).
* **Logistics Fulfillment Realism:** Simulates messy, non-uniform fulfillment delays using an exponential scale parameter ($\beta = 30$) to accurately reflect real-world shipping distribution shapes.

---

## 🔬 Multi-Stage Audit Methodology

The framework evaluates the black-box ecosystem through three decoupled confirmation layers to ensure robust detection metrics.

### 1. Machine Learning Surrogate Auditing

A Random Forest Classifier acts as an adversarial surrogate model trained to duplicate the platform's top-tier visibility allocations (the upper 15% visibility window). The model achieves a classification accuracy of **96.17%** and a top-tier $F_1$-score of **0.8670**, proving that platform sorting behavior is highly deterministic and structurally predictable.

Global feature importances reveal a significant analytics paradox: while `historical_sales` serves as the dominant global split driver (65.85%), `processing_time_hours` emerges as the second most critical variable (12.09%), completely outranking localized engagement variables like conversion and click-through rates.

### 2. Localized Game-Theoretic Attribution

To expose why a logistics variable outranks core conversion drivers, the framework uses SHAP (SHapley Additive exPlanations) to isolate point-by-point feature attributions.

> **[INSERT FILE HERE: `data/shadow_threshold_audit.png`]**
> *Locate this plot immediately under this section. This visualization graphs the SHAP scatter plot where fulfillment latency shifts from a stable neutral state to a sharp vertical drop, exposing the precise localized breaking point in visibility.*

Between 0 and 48 hours, logistics delays maintain a stable, near-zero or positive contribution to visibility logs. The exact moment fulfillment latencies exceed the 48-hour threshold, the Shapley values experience a severe, discontinuous vertical drop down to deep negative territory (ranging from -0.1 to -0.4), capturing a non-linear drop-dead policy filter.

### 3. Automated Change-Point Discovery & Statistical Inference

To ensure scientific validity and remove experimental bias, a standalone execution sandbox (`discover_threshold.py`) acts as a blind black-box discovery engine. Pretending to have zero prior knowledge of the threshold, it sweeps through candidate tracking intervals (from 12 to 84 hours in 2-hour increments) and calculates population variance optimization.

The algorithm automatically isolates the exact point of maximum systemic rupture at **48 hours**, where the distance between population means reaches its zenith.

```text
--- algorithmic discovery report ---
identified system fracture point: 48 hours latency
maximal practical effect size (cohen's d): 1.68
welch's t-statistic at discovery point: -159.5563
empirical p-value profile: 0.0000e+00

```

* **Practical Significance:** A calculated Cohen's $d$ of **1.68** demonstrates a monumental practical effect size, proving that the two population distributions are fundamentally separated and do not merely differ due to large sample scaling.
* **Inferential Significance:** A Welch's $t$-statistic of **$-159.56$** ($p = 0.0000e+00$) firmly rejects the null hypothesis, demonstrating a statistically and practically significant regime change across the 48-hour boundary.

---

## 📈 Population Distribution Analysis

The global population displacement is illustrated through the system density profiles:

> **[INSERT FILE HERE: `data/discovered_change_point_audit.png`]**
> *Locate this distribution histogram at the bottom of the empirical section. This visualization explicitly illustrates the systemic population separation between compliant and penalized ranking regimes.*

* **Compliant Allocation Regime ($\le 48\text{h}$, Blue):** Concentrates uniformly inside premium, low-index ranking real estate, dominating organic visibility brackets.
* **Penalized Allocation Regime ($> 48\text{h}$, Red):** Suffers a wholesale structural displacement, compressing marginalized merchants entirely out of search visibility and piling them heavily into the lowest exposure brackets ($6,000$ to $12,000$).

---

## ⚠️ Limitations & Real-World Extrapolations

While this framework establishes a highly robust methodology for reverse-engineering algorithmic constraints, translating the audit from a mirrored environment to an external production system introduces key technical constraints:

* **The Synthetic Ground-Truth Boundary:** The primary limitation is that the data generation script relies on a hard mathematical drop-dead threshold. While the discovery engine independently recovered the 48-hour breaking point with zero internal knowledge, production algorithms typically rely on smooth, dynamic decay functions or deep neural embeddings rather than static step-penalties.
* **Data Ingestion Constraints:** Real-world algorithmic auditing must rely on observable outcomes extracted externally. In a production audit (e.g., assessing Amazon search or Uber allocations), features like historical sales or internal scores must be proxy-estimated via high-frequency programmatic web scraping, consumer panel monitoring, and public merchant API endpoints.
* **Omitted Variable Bias (OVB):** Real sorting engines ingest thousands of dense multi-modal signals simultaneously, including localized user search intent, real-time localized inventory positions, and historical behavioral sequence vectors. Excluding these hidden vectors can inflate the apparent effect size of observable features.
* **Adversarial Personalization Confounders:** Modern sorting frameworks personalize search results at an individual user level. An external audit must isolate population-level systemic biases from localized user-level personalization noise by standardizing geographic locations, using unauthenticated tracking sessions, and aggregating large-scale cross-sectional sampling runs.

---

## 🚀 Replicating the Framework Locally

### Prerequisites

Install core data engineering and statistical dependencies:

```bash
pip install duckdb pandas scikit-learn shap matplotlib scipy statsmodels

```

### End-to-End Orchestration Run

Execute the master orchestration switchboard to construct the DuckDB relational layers, train the surrogate forest, and export localized explainability assets in a continuous loop:

```bash
python shadow_ranking_project/main.py

```

### Standalone Discovery Execution

Run the automated grid-search change-point model to independently sweep populations, calculate Cohen's $d$ effect metrics, and generate the independent validation charts:

```bash
python shadow_ranking_project/discover_threshold.py

```
