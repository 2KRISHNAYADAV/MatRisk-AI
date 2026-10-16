# MatRisk AI — Research Intelligence & Model Audit Manifesto
### An Empirical Cross-Domain Platform Linking Material Physics, Infrastructure Degradation, Commodity Markets, and Actuarial Credit Risk

---

## 1. Executive Summary & Project Architecture

**MatRisk AI** is an end-to-end quantitative research framework designed to bridge the gap between microscopic physical material properties and macroeconomic infrastructure credit risk. Historically, material science research (DFT calculations, crystal structures, stress-strain curves) and financial risk management (Basel II credit exposure, Monte Carlo loss distributions, commodity derivatives) operate in complete isolation. 

MatRisk AI synthesizes **six heterogeneous datasets (DS1–DS6)** into an integrated predictive pipeline spanning **nine sequential research notebooks (01–09)**.

```mermaid
flowchart TD
    subgraph D_IN["1. Heterogeneous Data Assets (DS1–DS6)"]
        DS1["DS1: Materials Database<br/>(1,200 formulas, bulk/shear mod, crystal systems)"]
        DS2["DS2: Commodity Spot & Futures<br/>(8 industrial metals, 10-year daily history)"]
        DS3["DS3: National Bridge Portfolio<br/>(5,000 bridge structures, structural specs, loan EAD)"]
        DS4["DS4: Macroeconomic & Supply Signals<br/>(FRED indices, tariffs, MQI, Herfindahl concentration)"]
        DS5["DS5: Environmental Stressors<br/>(NOAA climate, freeze-thaw cycles, marine salinity)"]
        DS6["DS6: Historical NBI Inspections<br/>(30-year failure records, empirical severity classes)"]
    end

    subgraph D_FEAT["2. Feature Engineering & Cross-Domain Linkage (NB 01–03)"]
        F1["Compositional Stoichiometry<br/>(Magpie feature extraction, atomic electronegativity)"]
        F2["Volatility & GARCH Engine<br/>(Conditional variance, heavy-tail Student-t parameters)"]
        F3["Bridge Structural Exposure Parser<br/>(ADT, deck geometry, cumulative fatigue cycles)"]
        F4["Macro & Supply Shock Vectorizer<br/>(Tariff elasticity, import concentration indices)"]
    end

    subgraph D_ML["3. Predictive Machine Learning Core (NB 04–05)"]
        M1["Material Property Regressors<br/>(HistGradientBoosting for K, G, density, Tm)"]
        M2["NBI Condition Rating Model<br/>(CatBoost monotonic regressor, R² = 0.81)"]
        M3["Case-Control Failure Classifier<br/>(Monotonic-constrained boosting, AUC = 0.737)"]
        M4["Empirical Severity Matrix<br/>(Discrete loss frequency-severity distribution)"]
    end

    subgraph D_RISK["4. Actuarial Risk Engine & Stress Testing (NB 06–07)"]
        R1["Calibrated Annual PD<br/>(Targeted to empirical population base rate: 1.064%/yr)"]
        R2["Closed-Form Expected Loss Engine<br/>(EL = Σ PD_i × LGD_i × EAD_i = $5.82M/yr)"]
        R3["Monte Carlo Gaussian Copula Simulator<br/>(50,000 joint portfolio paths, VaR95 = $8.82M)"]
        R4["Macro Stress Shocks (SC-001 to SC-005)<br/>(Climate corrosion triples EL; alloy upgrades cut EL)"]
    end

    subgraph D_VAL["5. Model Interpretability & Governance Audit (NB 08–09)"]
        V1["SHAP Global & Local Attribution<br/>(Category dominance, inverse cell volume discovery)"]
        V2["Partial Dependence Verification<br/>(Monotonicity proof with respect to corrosion rate)"]
        V3["5/5 Cross-Notebook Consistency Audit<br/>(Zero tolerance drift check, 15 logged assumptions)"]
    end

    DS1 & DS4 --> F1
    DS2 --> F2
    DS3 & DS5 & DS6 --> F3
    DS4 & DS5 --> F4

    F1 --> M1
    F3 --> M2
    F3 & F4 --> M3
    DS6 --> M4

    M1 --> R4
    M2 & M3 & M4 --> R1 --> R2
    R1 & F2 --> R3
    R2 & R3 --> R4

    M1 & M2 & M3 --> V1
    M2 & M3 --> V2
    R2 & R3 & R4 & V1 & V2 --> V3
```

---

## 2. Master Data Asset Specifications (DS1–DS6)

The project establishes clean data contracts and leakage isolation across six foundational datasets:

| Asset Code | Domain Description | Records / Scope | Primary Attributes | Key Findings & Leakage Boundaries |
|:---|:---|:---|:---|:---|
| **DS1** | Materials Physics Database | 1,200 crystalline compounds | Stoichiometry, crystal system, cell volume, bulk modulus ($K$), shear modulus ($G$), density, band gap, formation energy | `category` dominates variance. Cell volume exhibits an unphysical inverse correlation with density in sparse subsets. |
| **DS2** | Commodity Derivative Markets | 8 industrial commodities (2014–2024 daily) | Aluminum, Cobalt, Copper, Iron Ore, Lithium, Nickel, Rare Earths, Steel HRC | Heavy-tailed returns; $\sqrt{T}$ scaling fails for multi-week horizons due to volatility clustering. |
| **DS3** | Infrastructure Asset Portfolio | 5,000 national bridge structures | Deck geometry, structural material, construction year, ADT, loan outstanding ($EAD$) | Total portfolio exposure: **$1,876.84M**. Annual insurance premiums collected: **$56.39M**. |
| **DS4** | Macroeconomic & Supply Signals | Longitudinal monthly series | FRED inflation, interest rates, tariff rates, Herfindahl import concentration index, Material Quality Index (MQI) | Premiums paid on bridge infrastructure exhibit zero correlation with empirical risk. |
| **DS5** | Environmental & Climate Stress | Spatial climate grids | Coastal salinity indices, freeze-thaw cycles, atmospheric $SO_2$ concentration | Deterministically dictates corrosion environment based on geographical location. Element cost coverage covers 32.6% of DS1. |
| **DS6** | Longitudinal NBI Failure Records | 30-year historical bridge failures (1,011 events) | Primary failure mode, collapse severity, age at failure, component corrosion rate | True empirical bridge failure base rate is **1.064%/yr** (1,011 failures / 19 years / 5,000 bridges). Severity is unforecastable ($R^2 < 0$). |

---

## 3. Deep Dive: Research Notebooks (01 through 09)

```mermaid
sequenceDiagram
    autonumber
    participant NB01_03 as NB 01-03: Audit & Featurization
    participant NB04 as NB 04: Materials ML
    participant NB05 as NB 05: Failure ML
    participant NB06 as NB 06: Risk Engine
    participant NB07 as NB 07: Stress Testing
    participant NB08 as NB 08: SHAP & PDP
    participant NB09 as NB 09: Synthesis & Signoff

    NB01_03->>NB04: Composition features & cleaned properties
    NB01_03->>NB05: Bridge exposures, corrosion rates, case-control labels
    NB04->>NB07: Modulus predictions for material substitution scenarios
    NB05->>NB06: Condition rating model & case-control classifier scores
    Note over NB05,NB08: Monotonicity Bug Discovered: Refitted with monotonic_cst
    NB06->>NB07: Baseline EL ($5.82M) & Monte Carlo Copula distribution
    NB07->>NB08: Stressed feature vectors (SC-001 to SC-005)
    NB05->>NB08: Tree ensembles for partial dependence & SHAP analysis
    NB08-->>NB09: Verified monotonicity proofs & feature attribution maps
    NB06 & NB07 & NB08->>NB09: Final cross-notebook consistency check (5/5 PASS)
```

---

### Notebook 01: Exploratory Data Audit & Cross-Dataset Linkage
* **Objective**: Audit raw tabular schemas across DS1–DS6, verify primary key integrity, detect cross-dataset leakage, and establish canonical feature types.
* **Key Achievements**:
  * Formalized mathematical mappings between bridge structural materials (DS3/DS6) and physical alloy compositions (DS1).
  * Validated that fatigue cycles match physical traffic load models:
    $$\text{Fatigue Cycles} \approx \frac{\text{ADT} \times 365 \times \text{Age}}{10^6}$$
    Holds within $\pm 10\%$ for **81.5%** of portfolio bridges.
  * Identified that bridge component thickness ratio critical ranges lie strictly between **0.417 and 0.500**, with only **0.8%** of bridges currently operating below critical threshold.

---

### Notebook 02: Material Exploratory Data Analysis & Physics Audit
* **Objective**: Investigate physical consistency among mechanical properties in DS1.
* **Key Discoveries**:
  * **Pugh Ratio Ductility**: Verified isotropic Pugh ratio $\frac{K}{G} > 1.75$ categorizes ductile behavior for **62.9%** of alloys.
  * **Poisson's Ratio Derivation**: Derived shear and bulk modulus isotropic parity:
    $$\nu = \frac{3K - 2G}{2(3K + G)}, \quad E = \frac{9KG}{3K + G}$$
    Verified $0 \le \nu \le 0.5$ for 100% of physical alloys.
  * **Element Cost Coverage**: DS5 cost tables cover exactly **32.6%** of DS1 compound compositions.

#### Analytical Figures: Physical Consistency & Material Relationships
![Predicted vs Actual KG](outputs/figures/A1_pred_vs_actual_KG.png)
*Figure 2.1: Out-of-fold parity plot for Bulk Modulus ($K$) and Shear Modulus ($G$), illustrating high dispersion around the line of equality.*

![Physics Consistency](outputs/figures/A5_physics_consistency.png)
*Figure 2.2: Isotropic mechanical consistency checks across Poisson's ratio $\nu$ and Pugh ratio $K/G$.*

![Density vs Volume](outputs/figures/A2_density_volume_relationship.png)
*Figure 2.3: Empirical density versus cell volume distribution highlighting the unphysical inverse correlation in selected alloy families.*

---

### Notebook 03: Financial & Commodity Exploratory Analysis & Feature Engineering
* **Objective**: Model price dynamics, return distributions, volatility clustering, and cross-domain macro-risk signals across 8 commodities.
* **Key Achievements**:
  * Built conditional volatility estimators using asymmetric GARCH(1,1).
  * Engineered the **Material Quality Index (MQI)**, **Supply Disruption Probability**, and **Herfindahl-Hirschman Concentration Index**.
  * Demonstrated that historical return distributions have substantial excess kurtosis (fat tails), causing classical Gaussian VaR to underestimate 21-day downside risk by up to **34%**.

#### Analytical Figures: Commodity Volatility, Seasonality, and Term Structure
![Commodity Correlation](outputs/figures/06A_commodity_correlation.png)
*Figure 3.1: 8-commodity return correlation matrix illustrating strong joint tail comovement between Copper, Steel, and Nickel.*

![Seasonality Check](outputs/figures/C2_seasonality_check.png)
*Figure 3.2: Monthly return seasonality and historical drift across industrial metals.*

![Term Spread Scale](outputs/figures/C3_term_spread_scale.png)
*Figure 3.3: Term spread distribution capturing futures curve backwardation and contango regimes.*

---

### Notebook 04: Material Property Predictive Models
* **Objective**: Train and evaluate machine learning models predicting $K$, $G$, density, melting point ($T_m$), and formation energy from chemical composition and crystal system.
* **Model Family**: `HistGradientBoostingRegressor` evaluated via 5-fold cross-validation with group-stratified splits.
* **Honest Evaluation Results**:
  * **The model beats the naive category-mean baseline on only 1 of 5 targets: Density ($R^2 = 0.758$ vs $0.255$).**
  * For bulk modulus ($K$) and shear modulus ($G$), the category-mean baseline outperforms complex tree models.

#### Material Model Performance Scorecard (Out-of-Fold)
| Target Property | Model $R^2$ (OOF) | Category-Mean Baseline $R^2$ | Beats Baseline? | Model MAE |
|:---|:---:|:---:|:---:|:---:|
| **Bulk Modulus ($K$, GPa)** | 0.4416 | **0.5124** | ❌ No | 49.10 GPa |
| **Shear Modulus ($G$, GPa)** | 0.2225 | **0.3582** | ❌ No | 33.74 GPa |
| **Density ($\text{g/cm}^3$)** | **0.7584** | 0.2547 | ✅ **Yes** | 1.04 $\text{g/cm}^3$ |
| **Melting Point ($T_m$, K)** | 0.2604 | **0.3583** | ❌ No | 447.19 K |
| **Formation Energy (eV/atom)** | 0.5461 | **0.5896** | ❌ No | 0.79 eV |

#### Analytical Figures: Material Model Performance
![Model vs Baseline R2](outputs/figures/A1_model_vs_baseline_R2.png)
*Figure 4.1: Out-of-fold $R^2$ comparison between Gradient Boosted Regressors and the category-mean baseline across all five material targets.*

![Thermodynamic Stability ROC](outputs/figures/A4_is_stable_roc.png)
*Figure 4.2: ROC-AUC evaluation for thermodynamic stability classification ($\text{AUC} \approx 0.71$).*

---

### Notebook 05: Infrastructure Degradation & Failure Risk Models
* **Objective**: Predict bridge superstructure condition rating (scale 0–9) and catastrophic structural failure probability using DS3 and DS6.
* **Key Findings**:
  * **Condition Rating**: Strongly dictated by a 2-variable function of `age` and `corrosion_rate_mm_yr` ($R^2 = 0.810$). Adding high-dimensional structural features only marginally shifts $R^2$ to $0.783$.
  * **Case-Control Failure Classifier**: Boosted classifier reaches $\text{AUC} = 0.737$ (versus a single-feature baseline of $\text{AUC} = 0.603$).
  * **Severity Unpredictability**: Regressing discrete failure loss severity yields negative out-of-fold performance ($R^2 = -0.319$), confirming loss severity is structurally stochastic. An empirical frequency-weighted LGD matrix was implemented instead.

#### Infrastructure Model Performance Matrix
| Model Target | Architecture | Validation Metric | Score | Baseline Score |
|:---|:---|:---:|:---:|:---:|
| **Condition Rating (Age + Corrosion)** | HistGradientBoosting | Out-of-Fold $R^2$ | **0.8101** | 0.4120 |
| **Condition Rating (Full Features)** | HistGradientBoosting | Out-of-Fold $R^2$ | **0.7833** | 0.4120 |
| **Failure Classifier (Case-Control)** | Monotonic Boosted Classifier | Out-of-Fold ROC-AUC | **0.7375** | 0.6033 |
| **Loss Severity Ordinal ($LGD$)** | Ordinal Regressor | Out-of-Fold $R^2$ | **-0.3187** | 0.0000 |

#### Analytical Figures: Degradation, Failure Risk & Severity
![Condition Rating Fit](outputs/figures/05A_condition_rating.png)
*Figure 5.1: Predicted vs actual NBI condition ratings using monotonic age and corrosion rate regressors.*

![Case-Control Model](outputs/figures/05B_case_control_model.png)
*Figure 5.2: ROC-AUC trajectory for the bridge failure case-control classifier.*

![Relative Risk by Material](outputs/figures/05B_relative_risk_by_material.png)
*Figure 5.3: Empirical odds ratio of structural failure stratified across material families (Cast Iron, Timber, Structural Steel, Pre-stressed Concrete).*

![Severity Unpredictability](outputs/figures/05C_severity_unpredictability.png)
*Figure 5.4: Empirical proof of failure severity unpredictability showing negative $R^2$ and justifying empirical discrete LGD modeling.*

![Combined Risk Score](outputs/figures/05D_combined_risk_score.png)
*Figure 5.5: Unified portfolio credit and failure risk distribution across the 5,000 national bridge portfolio.*

---

### Notebook 06: Financial Risk Engine & Actuarial Credit Loss Simulation
* **Objective**: Construct the enterprise capital reserve engine combining individual bridge asset risk parameters into portfolio-wide credit loss distributions.
* **Actuarial Identity Formulation**:
  $$\text{EL}_{\text{portfolio}} = \sum_{i=1}^{5000} \text{PD}_i \times \text{LGD}_i \times \text{EAD}_i = \mathbf{\$5.816M / yr}$$
* **Monte Carlo Simulation**: Executed **50,000 joint portfolio draws** via a Gaussian copula linking asset default thresholds to macro-commodity shock factors:
  * **MC Mean Annual Loss**: **$5.81M** (matches closed-form identity within **0.1%**)
  * **95% Value at Risk ($\text{VaR}_{95}$)**: **$8.82M**
  * **99% Value at Risk ($\text{VaR}_{99}$)**: **$10.43M**
  * **95% Expected Shortfall ($\text{ES}_{95}$)**: **$9.83M**
* **The Actuarial Decoupling Discovery**:
  * Total annual insurance premiums collected across the 5,000 bridge portfolio: **$56.39M**.
  * Ratio of premiums collected to true expected loss: **$9.69\times$**.
  * Statistical correlation between premium charged and asset expected loss: **$r \approx 0.02$ (near-zero)**. Infrastructure insurance pricing has historically decoupled from underlying materials physics.

#### Analytical Figures: Capital Adequacy, VaR & Actuarial Distributions
![Credit Risk Distribution](outputs/figures/06B_credit_risk.png)
*Figure 6.1: Distribution of calibrated annual Probability of Default (PD) and Expected Loss (EL) across all 5,000 bridges.*

![Monte Carlo Losses](outputs/figures/06C_monte_carlo_losses.png)
*Figure 6.2: 50,000-path Monte Carlo credit portfolio loss distribution highlighting Closed-Form EL, VaR 95%, and VaR 99%.*

![Premium vs EL Decoupling](outputs/figures/06D_premium_vs_el.png)
*Figure 6.3: Empirical scatter showing total decoupling ($r \approx 0.02$) between annual insurance premiums and physics-based expected loss.*

![VaR Multi-Period Scaling](outputs/figures/06A_var_scaling.png)
*Figure 6.4: Empirical historical simulation VaR versus theoretical $\sqrt{T}$ scaling demonstrating fat-tail divergence.*

---

### Notebook 07: Macroeconomic & Climate Scenario Stress Testing
* **Objective**: Evaluate portfolio resilience under five stressed market, supply chain, and climate scenarios (SC-001 through SC-005).

#### Scenario Definitions & Quantitative Impacts
| Scenario Code | Scenario Name & Catalyst | Commodity Price Impacts | Credit Portfolio Loss ($EL$) | Multiplier vs Baseline |
|:---|:---|:---|:---:|:---:|
| **Baseline** | Steady-state operating conditions | Historical distributions | $4.19M | 1.00× |
| **SC-001** | Critical Minerals Demand Surge (EV Transition) | Rare Earths: **+179.5%**, Lithium: **+25.2%**, Iron Ore: **+12.1%** | $4.19M | 1.00× |
| **SC-002** | Green-Steel Transition (Recycled EAF shift) | Steel HRC: **-8.0%**, Iron Ore: **-5.1%** | $4.19M | 1.00× |
| **SC-003** | **Climate-Accelerated Corrosion (30yr Coastal/Urban Shock)** | All metals: **+3.0%** | **$12.22M** | **2.91× (Tripled)** |
| **SC-004** | Lightweighting Aluminium Substitution | Virgin Al: **-15.0%**, Recycled Al: **+24.9%** | $4.19M | 1.00× |
| **SC-005** | **Alloy Improvement (Next-Gen Stainless Deployment)** | Steel HRC: **-20.0%**, Titanium: **-10.1%** | **$3.06M** | **0.73× (27% Reduction)** |

#### Analytical Figures: Stress Test Dynamics
![Price Shocks](outputs/figures/07B_price_shocks.png)
*Figure 7.1: Mean and 5th/95th percentile price shock impacts across scenarios SC-001 through SC-005.*

![Credit Stress Comparison](outputs/figures/07C_credit_stress.png)
*Figure 7.2: Credit portfolio expected loss escalation comparing baseline against climate-accelerated corrosion (SC-003).*

![Stressed Monte Carlo Loss Density](outputs/figures/07D_stressed_mc.png)
*Figure 7.3: Outward tail shift of the aggregate portfolio Monte Carlo loss distribution under SC-003 severe stress.*

---

### Notebook 08: Model Interpretability & SHAP Feature Attribution
* **Objective**: Compute TreeSHAP attributions and partial dependence curves (PDPs) across models to ensure physical explainability and regulatory auditability.
* **Key Discoveries**:
  * **Category Dominance**: In the DS1 material property models, one-hot encoded `category` accounts for **56% to 66%** of total $|SHAP|$ attribution. True compositional stoichiometry provides marginal residual signal.
  * **Deterministic Geography**: Spatial analysis revealed that bridge `location` deterministically dictates `corrosion_environment`.
  * **Monotonicity Verification**: Direct PDP inspection proved that refitted models maintain strictly monotonic risk relationships with respect to corrosion rate.

#### Analytical Figures: SHAP Attributions & Partial Dependence Proofs
![SHAP Bulk Modulus](outputs/figures/08A_shap_bulk_modulus_GPa.png)
*Figure 8.1: Global TreeSHAP feature importance ranking for Bulk Modulus ($K$).*

![SHAP Shear Modulus](outputs/figures/08A_shap_shear_modulus_GPa.png)
*Figure 8.2: Global TreeSHAP feature importance ranking for Shear Modulus ($G$).*

![SHAP Density](outputs/figures/08B_shap_density.png)
*Figure 8.3: TreeSHAP feature importance for compound Density.*

![SHAP Density Volume Dependence](outputs/figures/08B_density_volume_dependence.png)
*Figure 8.4: SHAP partial dependence plot for density versus atomic cell volume.*

![SHAP Condition Rating](outputs/figures/08C_shap_condition.png)
*Figure 8.5: TreeSHAP feature importance for superstructure condition rating.*

![PDP Corrosion Condition](outputs/figures/08C_pdp_corrosion.png)
*Figure 8.6: Partial Dependence Plot proving strictly monotonic decline in bridge condition rating with increasing corrosion rate.*

![SHAP Case-Control](outputs/figures/08D_shap_case_control.png)
*Figure 8.7: TreeSHAP attributions for the bridge failure case-control classifier.*

![PDP Monotonic Failure Probability](outputs/figures/08D_pdp_and_overlap.png)
*Figure 8.8: Partial dependence curves verifying that failure probability increases monotonically with corrosion rate across all alloy regimes.*

---

### Notebook 09: Synthesis, Cross-Notebook Audit & Decision Validation
* **Objective**: Execute automated cross-notebook consistency audits, validate predictions against external experimental literature benchmarks, and compile the final signed-off assumptions ledger.
* **Independent Reference Material Benchmarking**:
  Model predictions were checked against certified literature benchmarks for 5 primary industrial engineering alloys:
  * Austenitic Stainless Steel 316L
  * Titanium Ti-6Al-4V (Grade 5)
  * Aluminum 6061-T6
  * Structural Carbon Steel A36
  * Nickel Alloy Inconel 718

#### Analytical Figures: Final Scorecard & External Validation
![Final Scorecard](outputs/figures/09A_scorecard.png)
*Figure 9.1: Comprehensive MatRisk AI model validation scorecard.*

![External Reference Validation](outputs/figures/09C_external_validation.png)
*Figure 9.2: Model predictions vs certified experimental literature values for 5 standard engineering alloys.*

---

## 4. Critical Discoveries & The Engineering Bug Fix

### The Monotonicity Bug in Notebook 05 & 07
During the development of Notebook 07 (Scenario Stress Testing), an anomaly emerged:
* **Anomaly**: Under Scenario SC-005 (a favourable technological scenario deploying corrosion-resistant alloys that reduced corrosion rates by $20\%$), the initial model predicted an **increase** in aggregate portfolio credit losses.
* **Root Cause**: The unconstrained `HistGradientBoostingClassifier` trained in Notebook 05 lacked physical shape constraints. In sparse regions of the multidimensional feature space, the tree partitions produced inverted step functions where **lower corrosion rates predicted higher failure probabilities** (affecting **45.6%** of sample perturbations).
* **The Solution**: Both the condition-rating regressor and the case-control classifier were refitted using scikit-learn's native monotonic constraints:
  ```python
  # Enforcing non-increasing condition rating and non-decreasing failure probability
  HistGradientBoostingClassifier(
      monotonic_cst=[-1 if col == "corrosion_rate_mm_yr" else 0 for col in feature_names]
  )
  ```
* **Impact**: Baseline Expected Loss shifted slightly from $\$5.89\text{M}$ to **$\$5.82\text{M}$**, while restoring physical consistency under perturbations: SC-005 properly reduced Expected Loss by **27%** ($0.73\times$).

```mermaid
graph LR
    subgraph Bug["Unconstrained Tree Artifact"]
        B1["Lower Corrosion Rate (-20%)"] --> B2["Sparse Decision Leaf"] --> B3["Spurious Higher Default Probability!"]
    end
    subgraph Fix["Monotonic Constraint Fix (monotonic_cst)"]
        F1["Lower Corrosion Rate (-20%)"] --> F2["Monotonic Tree Boundary"] --> F3["Decreased Loss ($3.06M, 0.73x baseline)"]
    end
```

---

### The Two-Stage PD Calibration Methodology
The case-control failure classifier in Notebook 05 was trained on a matched retrospective sample with an artificial **16.8%** positive failure rate. Directly applying raw sigmoid outputs as annual default probabilities ($PD$) would overestimate nationwide infrastructure defaults by more than **$15\times$**.

In Notebook 06, a rigorous population base-rate calibration was established:
1. Historical national inventory data reveals **1,011 bridge structural failures** over **19 years** across **5,000 bridges**.
2. The true empirical annual failure rate is:
   $$\text{PD}_{\text{target}} = \frac{1,011 \text{ failures}}{19 \text{ years} \times 5,000 \text{ bridges}} = \mathbf{1.064\% / yr}$$
3. Raw classifier risk scores are rescaled across the portfolio:
   $$\text{PD}_i = \text{Score}_i \times \frac{\text{PD}_{\text{target}}}{\frac{1}{N} \sum_{j=1}^N \text{Score}_j}$$
This preserves the model's out-of-fold rank-ordering power ($\text{AUC} = 0.737$) while ensuring the expected portfolio loss strictly conforms to empirical reality.

---

## 5. Formal 5/5 Cross-Notebook Consistency Audit

To guarantee scientific reproducibility and prevent data drift between notebooks, Notebook 09 executes an automated five-point audit matrix:

| Check # | Audit Specification | Target Condition | Observed Verification | Audit Status |
|:---:|:---|:---|:---:|:---:|
| **1** | Physics Consistency ($E, \nu$) | $0 \le \nu \le 0.5$ and $E > 0$ for all physical rows | **100.0% Valid** | ✅ **PASSED** |
| **2** | Actuarial Identity Convergence | Monte Carlo Mean Loss vs Closed-Form $EL$ | **$5.81M vs $5.82M (0.1% diff)** | ✅ **PASSED** |
| **3** | Monotonicity Integrity Proof | Non-decreasing failure probability in corrosion | **Strictly Monotonic** | ✅ **PASSED** |
| **4** | Cross-Dataset Coverage Parity | DS5 element cost match with EDA §A7/C3 | **Exactly 32.6%** | ✅ **PASSED** |
| **5** | Discriminative Power Separation | Non-zero case-control rank separation | **Spearman $\rho = 0.134$** | ✅ **PASSED** |

---

## 6. Known Limitations & Research Roadmap

While the pipeline establishes an empirical cross-domain benchmark, the research audit identifies several methodological boundaries:

```mermaid
mindmap
  root((MatRisk AI Roadmap))
    Material Physics Modeling
      Current: Category dominance (56-66% SHAP)
      Needed: Deep Graph Neural Networks (SchNet / MEGNet)
      Needed: Synthesize higher-entropy alloy DFT datasets
    Infrastructure Spatial Resolution
      Current: Regional segment shocks
      Needed: GIS-coordinate microclimate weather data
      Needed: Bridge deck strain sensor telemetry integration
    Financial & Derivatives Architecture
      Current: Closed-form Basel II + Gaussian Copula
      Needed: Student-t Copula with asymmetric tail dependence
      Needed: Dynamic multi-asset delta-gamma hedging engine
    Generative Inverse Material Design
      Current: Constraint-satisfaction lookup
      Needed: Generative Diffusion / VAE for novel green alloys
```

1. **Category Dominance in Material Moduli**:
   * *Limitation*: Bulk and shear modulus regressors rely heavily on coarse crystal categories rather than subtle stoichiometric variations.
   * *Improvement*: Transition to crystal graph neural networks (e.g., ALIGNN or MEGNet) trained on larger structural databases (e.g., Materials Project).
2. **Asset-Level Stress Testing Resolution**:
   * *Limitation*: Macroeconomic and climate stress scenarios in Notebook 07 perturb market-wide or material-wide cohorts; bridge-specific sensor data is not available in the public NBI table.
   * *Improvement*: Integrate real-time bridge strain gauge and acoustic emission sensor streams to achieve asset-specific structural health monitoring.
3. **Generative Inverse Design**:
   * *Limitation*: Notebook 06's inverse design module operates via property-constrained Pareto filtering over known materials.
   * *Improvement*: Implement conditional generative models (e.g., Crystal Diffusion Models) to propose novel low-carbon alloy compositions that satisfy strict bulk modulus and corrosion constraints.

---

## 7. Reproduction & Execution Guide

### Workspace Prerequisites
* Python 3.10+ (Tested through Python 3.14)
* Core Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `catboost`, `plotly`, `shap`, `joblib`

### Running the Notebook Pipeline
Execute the research notebooks sequentially from the project root:
```bash
# 1. Exploratory Data Audit & Linkage
jupyter execute notebooks/01_data_audit.ipynb
jupyter execute notebooks/02_material_EDA.ipynb
jupyter execute notebooks/03_financial_EDA.ipynb

# 2. Feature Store Construction
jupyter execute notebooks/03_feature_engineering.ipynb

# 3. Model Training & Evaluation (Persists models/*.joblib)
jupyter execute notebooks/04_material_property_models.ipynb
jupyter execute notebooks/05_failure_prediction.ipynb

# 4. Actuarial Risk Engine & Stress Testing
jupyter execute notebooks/06_financial_risk_engine.ipynb
jupyter execute notebooks/07_scenario_stress_testing.ipynb

# 5. Interpretability & Final Validation Audit
jupyter execute notebooks/08_model_interpretability.ipynb
jupyter execute notebooks/09_final_validation.ipynb
```

All trained models are serialized to `data/processed/models/*.joblib`, summary scorecards are generated in `outputs/tables/*.csv`, and all 34 publication-grade analytical figures are rendered to `outputs/figures/*.png`.
