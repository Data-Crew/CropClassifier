import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ExtendedMetricsCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for logging extended classification metrics after each epoch.

    This callback computes and logs:
    - Confusion matrix (saved as .png)
    - Classification report (saved as .txt)
    - Balanced accuracy and macro F1-score (added to logs)
    - Per-class recall (added to logs)

    Parameters:
    - val_ds (tf.data.Dataset): Validation dataset with batched inputs and one-hot encoded labels.
    - label_legend (list[str]): List of class labels (used for metrics and plots).
    - output_dir (str): Directory to save classification reports and confusion matrix images. 
                        It is recommended to match `log_dir` from the training run.
    """

    def __init__(self, val_ds, label_legend, output_dir):
        super().__init__()
        self.val_ds = val_ds
        self.label_legend = label_legend
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for x_batch, y_batch in self.val_ds:
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, target_names=self.label_legend, digits=3, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        recall_per_class = recall_score(y_true, y_pred, average=None)

        # Add metrics to logs for Keras tracking
        if logs is not None:
            logs["val_balanced_accuracy"] = balanced_acc
            logs["val_macro_f1"] = macro_f1
            logs["val_micro_f1"] = micro_f1
            for idx, label in enumerate(self.label_legend):
                logs[f"val_recall_{label}"] = recall_per_class[idx]

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.label_legend, yticklabels=self.label_legend)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, f"confusion_matrix_epoch_{epoch + 1}.png")
        plt.savefig(fig_path)
        plt.close()

        # Save classification report
        report_path = os.path.join(self.output_dir, f"classification_report_epoch_{epoch + 1}.txt")
        with open(report_path, "w") as f:
            f.write(cr)

        # Console output
        print(f"\n[Confusion Matrix] Saved at {fig_path}")
        print(f"[Classification Report] Saved at {report_path}")
        print(f"Balanced Accuracy: {balanced_acc:.3f}, Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}\n")

class AnnealedAlphaCallback(tf.keras.callbacks.Callback):
    def __init__(self, alpha_vector, alpha_var,
                 class_index, max_alpha=1.3, total_epochs=15):
        super().__init__()
        self.alpha_vector = alpha_vector          
        self.alpha_var    = alpha_var             
        self.cid          = class_index
        self.base_val     = alpha_vector[class_index]
        self.max_alpha    = max_alpha
        self.T            = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        frac = epoch / max(1, self.T - 1)
        new_val = self.max_alpha - 0.5 * (self.max_alpha - self.base_val) \
                  * (1 + np.cos(np.pi * frac))
        self.alpha_vector[self.cid] = new_val     
        alpha_np = self.alpha_var.numpy()
        alpha_np[self.cid] = new_val
        self.alpha_var.assign(alpha_np)           

        print(f"[AnnealedAlpha] Epoch {epoch+1}: "
              f"alpha[Cultivated]={new_val:.3f}")