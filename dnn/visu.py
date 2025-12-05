import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("dataset.csv")
sns.set_style("whitegrid")

x_axis = "valor"
range1 = df[(df["area"] > 0) & (df["area"] < 5000)]
range2 = df[(df["area"] >= 5000)]

print(range1.describe())
print(range2.describe())

print(df)

# 3. Criação da Visualização da Distribuição
plt.figure(figsize=(10, 6))

# Usando seaborn.histplot para criar o histograma e o KDE
# 'kde=True' adiciona a curva de densidade estimada
# 'bins' controla o número de barras (ajuste conforme necessário)
sns.histplot(data=df, x=x_axis, kde=True, bins=30, color="skyblue", edgecolor="black")

# 4. Adicionar Títulos e Rótulos
plt.title("Distribuição de Frequência dos Dados", fontsize=16)
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Frequência", fontsize=12)

# 5. Exibir o Gráfico
plt.show()

print("\n--- Estatísticas Descritivas ---")
print(df[x_axis].describe())

val_unit = df["valor"] / df["area"]


plt.scatter(
    df["area"],
    val_unit,
    marker="o",  # Adiciona marcadores em cada ponto de dados
    color="blue",  # Define a cor da linha
)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
