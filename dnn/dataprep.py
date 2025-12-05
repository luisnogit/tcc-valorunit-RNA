from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
import sklearn.preprocessing


def extract_setor(x: str):
    """
    docstring
    """
    num_str = x.split(".")
    return int(num_str[0])


df = pd.read_csv("dataset.csv", index_col=0).drop(labels=["ano"], axis=1)
df['cadastro'] = df['cadastro'].apply(lambda x: extract_setor(x))
cadastros_val = pd.read_csv("cadastro.csv", header=None)

df = df.join(cadastros_val, on="cadastro", how="right").drop(columns=[0, 'cadastro'])
df = df[df["area"] > 100]
df = df[df["area"] < 15000]


valor = df["valor"] / df["area"]
df = df.drop(columns=["valor", "frente"])
df["valor"] = valor
df = df[df["valor"] > 10]
df = df[df["valor"] < 30000]

df = df[(np.abs(stats.zscore(df["area"])) < 2)]

plt.scatter(df['area'], df['valor'], alpha=0.3)
plt.show()
print(df.describe())
# df = df.map(lambda x: np.log(x))
scaler = sklearn.preprocessing.MinMaxScaler()
df.columns = df.columns.astype(str)
df[['area', '1','valor']] = scaler.fit_transform(df)
print(df.isna().sum())
print(df.describe())
print(df)
plt.scatter(df['area'], df['valor'], alpha=0.3)
plt.show()

print(f"datamin:{scaler.data_min_}, datamax{scaler.data_max_}, range{scaler.data_range_}, min_{scaler.min_}")


df.to_csv("./dnn/dataset.csv", header=False, index=False)
