# Evaluation Report: Server 2 Run

## Summary
The training run was interrupted at Epoch 118. We evaluated the `best_model_multiclass.pth` (saved at Epoch 114) against the validation set.

**Status:** ⚠️ **Partial Success**
- **Global Performance:** Strong (Micro Mean IoU ~91.9%)
- **Target Improvement:** **Failed** for "Infra".
- **Secondary Improvement:** Marginal for "Tiled".

## Per-Class Results

| Class | IoU | Accuracy | Precision | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Background** | 0.9586 | 0.9750 | 0.9828 | Dominant class |
| **Building** | 0.6425 | 0.8121 | 0.7547 | Decent |
| **Road** | 0.7527 | 0.8723 | 0.8459 | Good |
| **Water** | 0.9346 | 0.9776 | 0.9550 | Excellent |
| **Infra** | **0.0000** | **0.0000** | **0.0000** | ❌ **CRITICAL FAILURE** |
| **RCC** | 0.8325 | 0.9170 | 0.9003 | Strong |
| **Tiled** | 0.6318 | 0.7466 | 0.8041 | ⚠️ Slight improvement |

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Analysis
1.  **Infra Blindness Persists:** Despite 20x oversampling and Focal Loss, the model has not learned to classify "Infra" pixels correctly. It likely confusing them with **Background** or **Building**.
2.  **Possible Reasons:**
    *   **Imbalance Limit:** 20x weight might still be insufficient if the class is 1:10,000.
    *   **Training Time:** 18 epochs of fine-tuning (from epoch 100) might not be enough to traverse the loss landscape out of the local minimum (where it ignores Infra).
    *   **Thresholding:** The model might be predicting low probabilities for Infra (e.g., 20%) which lose to Background (80%) in `argmax`.

## Recommendations
1.  **Probability Check:** We should check the *raw probabilities* for Infra pixels. If the model predicts ~10-30% for Infra, we can lower the decision threshold for that class specifically during inference (e.g., `if infra_prob > 0.1: class = Infra`).
2.  **Aggressive Oversampling:** Increase weight to 50x or 100x.
3.  **Specialized Tuning:** Freeze the encoder and trained layers, and only train the specific head or use a binary classifier for Infra.
