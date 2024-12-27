import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import seaborn as sns
from PIL import Image, ImageTk

def calculate():
    try:
        tp = int(entry_tp.get())
        tn = int(entry_tn.get())
        fp = int(entry_fp.get())
        fn = int(entry_fn.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid integers for TP, TN, FP, and FN.")
        return
    
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
    confusion_matrix_img = Image.open(img_cm)
    confusion_matrix_img = ImageTk.PhotoImage(confusion_matrix_img)
    label_cm.config(image=confusion_matrix_img)
    label_cm.image = confusion_matrix_img
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
    heatmap_img = Image.open(img_heatmap)
    heatmap_img = ImageTk.PhotoImage(heatmap_img)
    label_heatmap.config(image=heatmap_img)
    label_heatmap.image = heatmap_img
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
    bar_img = Image.open(img_bar)
    bar_img = ImageTk.PhotoImage(bar_img)
    label_bar.config(image=bar_img)
    label_bar.image = bar_img
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
    
    for measure in measures:
        tree.insert("", "end", values=(measure["name"], measure["value"], measure["formula"]))

root = tk.Tk()
root.title("Confusion Matrix Hesaplayıcı")
root.state('zoomed')  # Make the window fullscreen

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)

ttk.Label(frame, text="True Positive (TP):").grid(row=0, column=0, sticky=tk.W)
entry_tp = ttk.Entry(frame)
entry_tp.grid(row=0, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="True Negative (TN):").grid(row=1, column=0, sticky=tk.W)
entry_tn = ttk.Entry(frame)
entry_tn.grid(row=1, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="False Positive (FP):").grid(row=2, column=0, sticky=tk.W)
entry_fp = ttk.Entry(frame)
entry_fp.grid(row=2, column=1, sticky=(tk.W, tk.E))

ttk.Label(frame, text="False Negative (FN):").grid(row=3, column=0, sticky=tk.W)
entry_fn = ttk.Entry(frame)
entry_fn.grid(row=3, column=1, sticky=(tk.W, tk.E))

ttk.Button(frame, text="Hesapla", command=calculate).grid(row=4, column=0, columnspan=2)

label_cm = ttk.Label(frame)
label_cm.grid(row=5, column=0, columnspan=2)

label_heatmap = ttk.Label(frame)
label_heatmap.grid(row=6, column=0, columnspan=2)

label_bar = ttk.Label(frame)
label_bar.grid(row=7, column=0, columnspan=2)

tree = ttk.Treeview(frame, columns=("Ölçüm", "Değer", "Türevler"), show="headings")
tree.heading("Ölçüm", text="Ölçüm")
tree.heading("Değer", text="Değer")
tree.heading("Türevler", text="Türevler")
tree.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()
