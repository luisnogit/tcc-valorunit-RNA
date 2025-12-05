import pandas as pd


def numtocad(number: str):
    """
    docstring
    """
    if number.__len__() == 10:
        return f"0{number[0:2]}.{number[2:5]}.{number[5:9]}-{number[9:10]}"
    else:
        return f"{number[0:3]}.{number[3:6]}.{number[6:10]}-{number[10:11]}"


sheets = pd.read_excel(
    "./GUIAS DE ITBI PAGAS XLS 28102025.xlsx",
    dtype={"N° do Cadastro (SQL)": str},
    sheet_name=None,
)
li = []
for _, df in sheets.items():
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True).drop(
    labels=[
        "Nome do Logradouro",
        "Número",
        "Complemento",
        "OBSERVAÇÕES",
        "USO",
        "DESCRIÇÃO",
        "PADRÃO",
        "CONCEITO",
    ],
    axis="columns",
)
df = df[
    [
        "N° do Cadastro (SQL)",
        "Natureza de Transação",
        "Valor de Transação (declarado pelo contribuinte)",
        "Proporção Transmitida (%)",
        "Área do Terreno (m2)",
        "Padrão (IPTU)",
        "Testada (m)",
    ]
]
df = df[df["Natureza de Transação"] == "1.Compra e venda"]
df = df[df["Proporção Transmitida (%)"] == 100]
df = df[df["Padrão (IPTU)"] == 0]
# df["N° do Cadastro (SQL)"] = df["N° do Cadastro (SQL)"].astype(str)
df = df.drop(
    labels=["Padrão (IPTU)", "Proporção Transmitida (%)", "Natureza de Transação"],
    axis="columns",
)

df["Ano"] = "2025"
print(df)
df["N° do Cadastro (SQL)"] = df["N° do Cadastro (SQL)"].map(lambda x: numtocad(x))
print(df["N° do Cadastro (SQL)"])
df.to_csv("./2025.csv", index=None, header=False)
print(df)
print(df.describe())

