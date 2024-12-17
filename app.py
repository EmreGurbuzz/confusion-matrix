from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', task_count=0, task_labels=[], complexities=[], impacts=[])

@app.route('/calculate', methods=['POST'])
def calculate():
    task_labels = []
    complexities = []
    impacts = []
    i = 0
    while f'taskLabel{i}' in request.form:
        task_labels.append(request.form[f'taskLabel{i}'])
        complexities.append(int(request.form[f'complexity{i}']))
        impacts.append(int(request.form[f'impact{i}']))
        i += 1
    
    # Create complexity and impact matrices
    complexity_matrix = np.array(complexities).reshape((len(task_labels), 1))
    impact_matrix = np.array(impacts).reshape((len(task_labels), 1))
    
    # Calculate measures (example calculations, adjust as needed)
    TP = complexity_matrix
    FP = impact_matrix
    FN = complexity_matrix
    TN = impact_matrix
    
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0
    
    measures = [
        {"name": "Hassasiyet", "value": np.mean([safe_divide(tp, tp + fn) for tp, fn in zip(TP, FN)]), "formula": "TPR = TP / (TP + FN)"},
        {"name": "Özgüllük", "value": np.mean([safe_divide(tn, fp + tn) for tn, fp in zip(TN, FP)]), "formula": "SPC = TN / (FP + TN)"},
        {"name": "Kesinlik", "value": np.mean([safe_divide(tp, tp + fp) for tp, fp in zip(TP, FP)]), "formula": "PPV = TP / (TP + FP)"},
        {"name": "Negatif Tahmini Değer", "value": np.mean([safe_divide(tn, tn + fn) for tn, fn in zip(TN, FN)]), "formula": "NBD = TN / (TN + FN)"},
        {"name": "Yanlış Pozitif Oranı", "value": np.mean([safe_divide(fp, fp + tn) for fp, tn in zip(FP, TN)]), "formula": "FPR = FP / (FP + TN)"},
        {"name": "Yanlış Keşif Oranı", "value": np.mean([safe_divide(fp, fp + tp) for fp, tp in zip(FP, TP)]), "formula": "FDR = FP / (FP + TP)"},
        {"name": "Yanlış Negatif Oranı", "value": np.mean([safe_divide(fn, fn + tp) for fn, tp in zip(FN, TP)]), "formula": "FNR = FN / (FN + TP)"},
        {"name": "Kesinlik", "value": np.mean([safe_divide(tp + tn, tp + fp + fn + tn) for tp, fp, fn, tn in zip(TP, FP, FN, TN)]), "formula": "ACC = (TP + TN) / (P + N)"},
        {"name": "F1 Puanı", "value": np.mean([safe_divide(2 * tp, 2 * tp + fp + fn) for tp, fp, fn in zip(TP, FP, FN)]), "formula": "F1 = 2TP / (2TP + FP + FN)"},
        {"name": "Matthews Korelasyon Katsayısı", "value": np.mean([safe_divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) for tp, fp, fn, tn in zip(TP, FP, FN, TN)]), "formula": "TP*TN - FP*FN / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))"}
    ]
    
    # Generate combined graph
    plt.figure(figsize=(10, 6))
    
    # Bar chart for complexities and impacts
    bar_width = 0.35
    index = np.arange(len(complexities))
    plt.bar(index, complexities, bar_width, label='Karmaşıklık')
    plt.bar(index + bar_width, impacts, bar_width, label='Etki')
    
    # Heatmap for complexities and impacts
    heatmap_data = np.array([complexities, impacts])
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', alpha=0.5, extent=[-0.5, len(complexities)-0.5, -0.5, 1.5])
    
    plt.xlabel('Değerler')
    plt.ylabel('Değerler (1-5)')
    plt.title('Karmaşıklık ve Etki Matrisi')
    plt.xticks(index + bar_width / 2, [f'{i+1}' for i in index], rotation=45)
    plt.yticks([0, 1], ['Karmaşıklık', 'Etki'])
    plt.colorbar(label='Değer')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template('index.html', task_count=len(task_labels), task_labels=task_labels, complexities=complexities, impacts=impacts, graph_url=graph_url, measures=measures)

if __name__ == '__main__':
    app.run(debug=True)
