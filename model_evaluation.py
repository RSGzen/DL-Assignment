import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from itertools import cycle

#function evaluate_model arguments:
#model: The Trained ResNet CNN Model
#history: Keras History object returned by model.fit() (containtraining logs accuracy/loss per epoch)
#X_test: Test features (numpy array)
#y_test: One-hot encoded true labels (numpy array)
#class_names: List of Dataset Names (Happy, Disgust, Angry..., etc.)
#output_dir: Outputs save in output folder

def evaluate_model(model, history, X_test, y_test, class_names, output_dir='outputs'):
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Predict class probabilities and labels
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy Score: {acc:.4f}")

    # 2. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # 3. Confusion Matrix
    # Raw Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
    plt.show()

    # 4. Accuracy & Loss Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.show()

    # 5. AUC Score (Macro)
    #Convert y_true values to binary one-hot encoded format
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    # Calculate the MACRO average of AUC using One-vs-Rest (OVR)
    auc_macro = roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')

    print(f"\nMacro AUC Score (OvR): {auc_macro:.4f}")

    # 6. ROC Curve Plot
    # Empty Dictionaries:
    # fpr = False Positive Rete
    # tpr = True Positive Rate
    # roc_auc = AUC Score for each class
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(class_names)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), colors):
        # Calculate fpr & tpr for each class
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])

        # Calculate AUC for each class
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], lw=2, color=color,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    #Labelling
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess', alpha=0.7)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve (OvR)\nMacro AUC = {auc_macro:.4f}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.show()

    # To write all Accuracy Sore, Classification Report & Macro AUC Score in outputs.txt
    with open(os.path.join(output_dir, 'outputs.txt'), 'w') as f:
        f.write(f"Accuracy Score: {acc:.4f}")
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro AUC Score (OvR): {auc_macro:.4f}")
