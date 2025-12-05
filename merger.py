import numpy as np
import pandas as pd

dfparquet = pd.read_csv("./datasetparquet.csv",header=None, index_col=None)
df2025 = pd.read_csv("./2025.csv", header=None,index_col=None)
print(dfparquet)
print(df2025)

df = pd.concat([df2025, dfparquet], ignore_index=True, axis=0)
print(df)
df.columns = ["cadastro", "valor", "area", "frente", "ano"]
print(df.count())
print(df.describe())
print(df)
df.drop(index=df[df["area"] < 50].index, inplace=True)

print(df.count())
print(df.describe())

df.to_csv("dataset.csv")
