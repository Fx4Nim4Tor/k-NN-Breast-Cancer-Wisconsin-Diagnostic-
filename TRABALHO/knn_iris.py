import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox


cancer = load_breast_cancer()
X = cancer.data[:, :10]  
y = cancer.target  

scaler = StandardScaler()
X = scaler.fit_transform(X)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo k-NN: {accuracy * 100:.2f}%')

cv_scores = cross_val_score(knn, X, y, cv=5)
print(f'Acurácia média com validação cruzada: {cv_scores.mean() * 100:.2f}%')
print(f'Desvio padrão das acurácias: {cv_scores.std() * 100:.2f}%')

print('\nRelatório de Classificação:\n', classification_report(y_test, y_pred, target_names=cancer.target_names))
print('\nMatriz de Confusão:\n', confusion_matrix(y_test, y_pred))

def predict():
    try:
        input_data = [float(entry.get()) for entry in entries]
        input_data = scaler.transform([input_data])

        prediction = knn.predict(input_data)
        result = "Benigno" if prediction[0] == 1 else "Maligno"
        messagebox.showinfo("Resultado", f"O tumor é classificado como: {result}")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao fazer previsão: {e}")

root = tk.Tk()
root.title("Classificador de Tumores de Mama")

labels = [
    "Raio (média):",
    "Textura (média):",
    "Perímetro (média):",
    "Área (média):",
    "Suavidade (média):",
    "Compacidade (média):",
    "Concavidade (média):",
    "Pontos Côncavos (média):",
    "Simetria (média):",
    "Dimensão Fractal (média):"
]

entries = []
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entries.append(entry)

tk.Button(root, text="Classificar", command=predict).grid(row=len(labels), column=0, columnspan=2)

root.mainloop()