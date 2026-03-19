from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Detailed classification metrics.
    """
    y_pred = (y_prob > threshold).astype(int)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    return {
        'report': report,
        'cm': cm,
        'pr_auc': pr_auc,
        'f1': f1_score(y_true, y_pred)
    }

def plot_confusion_matrix(cm, classes=['Stayed', 'Resigned']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_pr_curve(y_true, y_prob, label='Model'):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, label=f'{label} (AUC = {auc(recall, precision):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
