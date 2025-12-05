import pandas as pd

# df = pd.read_csv("planilhageral.csv", index_col=0, dtype=str)
# print(df.map(lambda x: x.rjust(3, "0")))
sheet = pd.read_excel("./planilhageral2.xlsx", sheet_name="PlanilhaGeral", index_col=None, header=None)

# li = []
# for _, df in sheets.items():
#     li.append(df)
# df = pd.concat(li, axis=0, ignore_index=True)
sheet = sheet[[0,4]]
print(sheet)
print(sheet.describe())
list = sheet.groupby([0]).mean()

list.to_csv("./cadastro.csv", header=None)

# df = pd.concat(li, axis=0, ignore_index=True)
