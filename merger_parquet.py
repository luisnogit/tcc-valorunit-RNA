import glob
import pandas as pd


# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)
year = 0
filestring = (f"/home/luis/Documents/patinotcc/staging/*-{year}.parquet",)

rawdata = glob.glob("/home/luis/Documents/patinotcc/staging/*.parquet")

li = []

for file in rawdata:
    df = pd.read_parquet(file, engine='fastparquet')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True).drop(
    labels=[
        "street_name",
        "property_number",
        "property_complement",
        "transaction_date",
        "construction_completion_year_iptu",
        "block_number",
        "month",
        "reference_market_value",
        "sql_status",
        "proportional_reference_market_value",
        "usage_description_iptu",
        "fiscal_sector",

    ],
    axis="columns",
)
df = df[df["usage_code_iptu"] == "0"]
df = df[df["transaction_nature"] == "1.Compra e venda"]
df = df[df["transmitted_proportion_percent"] == 100]
df = df.drop(columns=["transmitted_proportion_percent", "construction_standard_description_iptu", "transaction_nature", "ideal_fraction", "built_area_sqm", "usage_code_iptu", "construction_standard_code_iptu"])
df = df.dropna()
df.to_csv("./datasetparquet.csv",header=False, index=False)
print(df)
print(df.describe())