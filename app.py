from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import seaborn as sns

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', measures=[])

@app.route('/calculate', methods=['POST'])
def calculate():
    tp = int(request.form['tp'])
    tn = int(request.form['tn'])
    fp = int(request.form['fp'])
    fn = int(request.form['fn'])
    
    # Calculate confusion matrix
    cm = np.array([[tp, fp], [fn, tn]])
    
    # Generate confusion matrix graph
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    img_cm = io.BytesIO()
    plt.savefig(img_cm, format='png')
    img_cm.seek(0)
    confusion_matrix_url = base64.b64encode(img_cm.getvalue()).decode()
    plt.close()
    
    # Generate heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    img_heatmap = io.BytesIO()
    plt.savefig(img_heatmap, format='png')
    img_heatmap.seek(0)
    heatmap_url = base64.b64encode(img_heatmap.getvalue()).decode()
    plt.close()
    
    # Generate bar graphs
    labels = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
    values = [tp, fp, fn, tn]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'orange', 'red', 'green'])
    plt.title('Classification Results')
    plt.ylabel('Count')
    
    img_bar = io.BytesIO()
    plt.savefig(img_bar, format='png')
    img_bar.seek(0)
    bar_url = base64.b64encode(img_bar.getvalue()).decode()
    plt.close()
    
    # Calculate measures
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0
    
    measures = [
        {"name": "Sensitivity", "value": safe_divide(tp, tp + fn), "formula": "TPR = TP / (TP + FN)"},
        {"name": "Specificity", "value": safe_divide(tn, fp + tn), "formula": "SPC = TN / (FP + TN)"},
        {"name": "Precision", "value": safe_divide(tp, tp + fp), "formula": "PPV = TP / (TP + FP)"},
        {"name": "Negative Predictive Value", "value": safe_divide(tn, tn + fn), "formula": "NPV = TN / (TN + FN)"},
        {"name": "False Positive Rate", "value": safe_divide(fp, fp + tn), "formula": "FPR = FP / (FP + TN)"},
        {"name": "False Discovery Rate", "value": safe_divide(fp, fp + tp), "formula": "FDR = FP / (FP + TP)"},
        {"name": "False Negative Rate", "value": safe_divide(fn, fn + tp), "formula": "FNR = FN / (FN + TP)"},
        {"name": "Accuracy", "value": safe_divide(tp + tn, tp + fp + fn + tn), "formula": "ACC = (TP + TN) / (P + N)"},
        {"name": "F1 Score", "value": safe_divide(2 * tp, 2 * tp + fp + fn), "formula": "F1 = 2TP / (2TP + FP + FN)"},
        {"name": "Matthews Correlation Coefficient", "value": safe_divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), "formula": "TP*TN - FP*FN / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))"}
    ]
    
    return render_template('index.html', confusion_matrix_url=confusion_matrix_url, heatmap_url=heatmap_url, bar_url=bar_url, measures=measures)

if __name__ == '__main__':
    app.run(debug=True)
