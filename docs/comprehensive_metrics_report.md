# Geo-AI Project: Comprehensive Metrics & Data Report

**Generated Date:** 2025-12-19
**Model:** DeepLabV3+ (ResNet50)
**Dataset:** Server Multiclass (Theme 1)

---

## 1. Training Configuration & Dataset Stats
| Metric | Value |
| :--- | :--- |
| **Total Epochs Run** | 118 (Interrupted) |
| **Best Epoch** | 114 |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 (ReduceLROnPlateau) |
| **Loss Function** | Focal Loss (gamma=2.0) + Dice Loss |
| **Training Set Size** | 6,043 Chips (Oversampled) |
| **Validation Set Size** | 672 Chips |

---

## 2. Global Model Performance (Best Epoch 114)
| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Micro-Averaged Mean IoU** | **0.9193** (91.93%) | High overall pixel accuracy |
| **Mean Per-Class IoU** | **0.6789** (67.89%) | Average performance across 7 classes |
| **Final Training Loss** | 0.6509 | (Epoch 114) |
| **Final Validation Loss** | 0.4458 | (Epoch 114) |

---

## 3. Detailed Per-Class Performance
*Evaluated on Validation Set (672 Chips)*

| Class ID | Class Name | **IoU** (Intersection/Union) | **Accuracy** (Recall) | **Precision** |
| :--- | :--- | :--- | :--- | :--- |
| 0 | **Background** | **0.9586** | 0.9750 | 0.9828 |
| 1 | **Building** | **0.6425** | 0.8121 | 0.7547 |
| 2 | **Road** | **0.7527** | 0.8723 | 0.8459 |
| 3 | **Water** | **0.9346** | 0.9776 | 0.9550 |
| 4 | **Infra** | **0.0000** | 0.0000 | 0.0000 |
| 5 | **RCC** | **0.8325** | 0.9170 | 0.9003 |
| 6 | **Tiled** | **0.6318** | 0.7466 | 0.8041 |

**Analysis:**
*   **Top Performer:** Background (95.8% IoU) and Water (93.4% IoU).
*   **Strong Performer:** RCC (83.2% IoU) - The model excels at detecting concrete roofs.
*   **Average Performer:** Road (75.2%) and Building (64.2%).
*   **Critical Outlier:** Infra (0.00%).

---

## 4. "Infra" Class Probability Statistics
*Quantifying the "Blindness" to Class 4*

| Metric | Value |
| :--- | :--- |
| **Total Infra Pixels Evaluated** | 17,804 |
| **Mean Confidence** | 0.12% (0.0012) |
| **Max Confidence Observed** | 1.18% (0.0118) |
| **Pixels with >10% Confidence** | 0.00% |
| **Pixels with >50% Confidence** | 0.00% |

*vs. RCC (Control Class)*
| Metric | Value |
| :--- | :--- |
| **RCC Mean Confidence** | 93.62% |
| **RCC > 50% Confidence** | 96.10% |

---

## 5. Confusion Matrix Summary
*(Inferred from IoU scores)*
*   **False Negatives (Infra):** 100% of Infra pixels were predicted as another class (likely Background or Building).
*   **False Positives (Infra):** 0 pixels were incorrectly predicted as Infra. The model is extremely conservative (zero predictions).

---
**End of Metrics Report**
