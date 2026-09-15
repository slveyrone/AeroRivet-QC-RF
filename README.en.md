# AeroRivet-QC-RF

*[Türkçe](README.md) | English*

An end-to-end **Machine Learning-based Quality Control** project for aircraft structural assembly (rivet/hole) processes. Separate Random Forest models for two rivet types (**MS20470** and **NAS1097**) are trained on a realistic production-line process model; decisions are made through a **three-tier response architecture** rather than a binary accept/reject, and every check is written to a traceability record.

## Project Summary

In aircraft fuselage assembly, rivet holes are subject to tight tolerances on diameter, countersink depth, and squeeze force. But not every rivet type goes through the same checks: **MS20470** (Universal/Protruding Head, dome-shaped) rivets are never countersunk, while for **NAS1097** (100° Flush/Countersunk Head) rivets, countersink depth is also a critical parameter. This project:

1. Builds a synthetic dataset with an **FAA-H-8083-31A** referenced, Cpk ≈ 1.33 capable process and a realistic (~2%) nonconformance rate,
2. Trains a class-weighted **Random Forest** classifier per rivet type,
3. Derives two thresholds from the data instead of a fixed one: a **capacity-constrained safety floor** and a **high-precision** threshold,
4. Presents results in a **Streamlit dashboard** that tiers decisions by rivet type and part criticality, with a calibration gate and traceability logging.

## Tiered Response Architecture

Instead of binary "locked/unlocked" logic, threshold strictness is matched to the cost of the action it triggers:

| Tier | Threshold logic | Required action | Line load |
|---|---|---|---|
| **Green** | Below the yellow threshold | Standard SPC monitoring, no extra step | ~85% |
| **Yellow** | Recall-maximizing threshold within the capacity ceiling | Documented secondary check — work does not stop | ~15% |
| **Red** | Threshold achieving Precision ≥ 90% | Work stops, routed to MRB, NDT triggered on Class A | ~2% |

**The cheap action is triggered by high recall; the expensive action by high precision.** In addition, a yellow tier on a **Class A** (primary structural) part is automatically escalated to red — this is the code equivalent of the operation × part-criticality matrix.

## Realistic Process Model

Earlier versions had a ~40% nonconformance rate, which did not represent a real production line. The current data generation has four layers:

1. **Capable baseline process:** All characteristics centered within the tolerance band, Cpk ≈ 1.33–1.39. With common-cause variation alone, the nonconformance rate stays below 0.1%.
2. **Special-cause excursions:** The real source of defects — drill wear (oversize holes), regulator drift (force deviation), and countersink cutter wear (depth drift). About 4% of parts are produced under an excursion condition, and most nonconformances originate there. This is precisely why SPC exists.
3. **Measurement/sensor noise:** The decision is made on the true physical value; only the measured (noisy) value appears in the dataset. Gauge variation is kept at ~30% of process variation.
4. **Asymmetric human inspection error:** 3% of genuine nonconformances are missed (`escape`), while 0.1% of sound parts are falsely condemned (`KONTROL_YANLIS_RED`). This asymmetry reflects real shop-floor behavior.

Result: a recorded nonconformance rate of 1.90% for MS20470 and 2.32% for NAS1097.

## Safety Floor: Why 99% Recall Cannot Be Imposed Directly

The most important technical finding from this build: **a 99% recall target cannot be chased against noisy labels.** The `KONTROL_YANLIS_RED` rows are physically sound parts that an inspector falsely condemned; the model correctly assigns them ~0 probability. When the recall target is imposed against those labels, the threshold collapses to zero and **100% of the line falls into the alert tier** — the system becomes unusable.

The fix has two parts:

- **The safety floor is measured against MRB-confirmed physical nonconformances**, not raw inspector labels. The real-world equivalent: ground truth is the MRB disposition, not the initial inspection tag.
- **The threshold is chosen under an operational capacity ceiling** (default: at most 15% of parts may go to a secondary check). If the target cannot be met within that capacity, the system does not hide it — it computes and reports the capacity that would be required.

## Model FAI Protocol

The characteristic-based verification logic of AS9102 Rev C Form 3 is carried over to model versions: **a new model version is a new "first article."** Each training run produces `model_fai_kaydi_<type>.json` containing the model fingerprint (shortened SHA-256), the decision threshold, and a separate recall verification per nonconformance type; the quality engineer approval field is left blank.

Acceptance criteria: recall ≥ 95% per nonconformance type, overall recall ≥ 99% on physical nonconformances, Brier score ≤ 0.05.

**Current status — the protocol is working:** the NAS1097 model returns **NONCONFORMING** from FAI verification. Within the 15% capacity ceiling it misses one countersink upper-limit nonconformance (recall 97.67%). The script reports that reaching the target would require raising secondary-check capacity to 16%. The model was deliberately not tuned until it passed; the result is left as-is to demonstrate that the gate actually works. The dashboard flags this model with a warning banner on every load.

## Current Results (reproducible with seed=42)

| Metric | MS20470 | NAS1097 |
|---|---|---|
| Recorded nonconformance rate | 1.90% | 2.32% |
| Yellow threshold | 0.0073 | 0.0049 |
| Physical recall (at yellow threshold) | 100.00% (37/37) | 97.67% (42/43) |
| Yellow line load | 15.0% | 15.0% |
| Red threshold | 0.7609 | 0.7105 |
| Red precision / recall | 91.67% / 86.84% | 90.48% / 82.61% |
| Red line load | 1.8% | 2.1% |
| Brier score (calibration) | 0.0035 | 0.0047 |
| Model FAI result | CONFORMING | **NONCONFORMING** |

## Project Structure

```
AeroRivet-QC-RF/
├── 01_veri_uretimi.py                      # Synthetic data from capable process + special-cause model
├── 02_model_egitimi.py                     # Class-weighted RF training, dual threshold derivation, Model FAI
├── app.py                                  # Tiered response dashboard (3 tabs)
├── requirements.txt                        # Python dependencies
├── havacilik_montaj_verisi_<type>.csv      # Generated synthetic data (created when the script runs)
├── montaj_modeli_<type>.pkl                # Trained models (created when the script runs)
├── metrikler_<type>.pkl                    # Test data, thresholds, FAI record (created when the script runs)
├── model_fai_kaydi_<type>.json             # Model FAI verification record (created when the script runs)
└── izlenebilirlik_kaydi.csv                # Digital birth certificate records (accumulates as the dashboard is used)
```

## Dataset Variables

Measurement columns hold **measured** (noisy) values; the model only ever sees these. The `Hata_Nedeni` (defect cause) column is not given to the model as a feature — it is used solely for per-cause recall verification in the Model FAI step.

| Variable | Description | Tolerance / Distribution | MS20470 | NAS1097 |
|---|---|---|:---:|:---:|
| `Delik_Capi_mm` | Measured hole diameter | Nominal 4.76 · limits 4.74–4.84 · σ=0.005 | Yes | Yes |
| `Baski_Kuvveti_PSI` | Measured squeeze force | Nominal 3000 · limits 2500–3500 · σ=120 | Yes | Yes |
| `Havsa_Derinligi_mm` | Measured countersink depth | Nominal 1.20 · limits 1.15–1.25 · σ=0.012 | No | Yes |
| `Operator_Tecrube_Yil` | Operator experience in years | Integer 1–15; process spread ×1.4 when inexperienced | Yes | Yes |
| `Rework_Gerekli` | Target variable (0=Sound, 1=Rework) | Rule-based + asymmetric inspection error | Yes | Yes |
| `Hata_Nedeni` | Nonconformance type (for FAI verification) | Not used as a feature | Yes | Yes |

Lower limits were added deliberately: insufficient squeeze force produces an under-driven shop head, a shallow countersink leaves the head proud of the surface, and an over-deep countersink creates a knife-edge condition — all three are genuine reject conditions.

## Setup and Running

```bash
pip install -r requirements.txt

python 01_veri_uretimi.py     # Synthetic data for both rivet types
python 02_model_egitimi.py    # Training, threshold derivation, Model FAI verification
streamlit run app.py          # Dashboard
```

Dashboard tabs:
- **Field Quality Control:** Select rivet type and part criticality class, verify the measuring tool's calibration validity (an expired tool blocks the check), enter measurements, and get a tiered decision. Every check is written to the traceability log.
- **Model Performance and FAI Record:** Safety floor status, tier thresholds, Brier score, the characteristic-level FAI table, and where the thresholds sit on the PR curve.
- **Traceability Log:** Digital birth certificate records with CSV export.

## Out of Scope

The following are defined in the architecture report but are not implemented in this codebase; they require enterprise system integration: MES/PLM integration, smart-tool physical inhibition, MRB workflow and signature management, the Operational Competence Profile (OYP) and OJT records, NDT equipment integration, and automated drift monitoring.

## Technology Stack

- **scikit-learn** — class-weighted Random Forest, PR curve, Brier score
- **pandas / numpy** — process simulation and data processing
- **streamlit** — tiered response dashboard
- **matplotlib / seaborn** — PR curve visualization
- **joblib** — model and metrics serialization
