import pandas as pd
import arff


# Load the CSV
df = pd.read_csv("hf://datasets/mihaicata/diabetes/all_data_processed.tsv", sep="\t")

inputs_expanded = (
    df["inputs"]
    .astype(str)
    .str.replace("'", "")
    .str.replace(" ", "")
    .str.replace("[", "")
    .str.replace("]", "")
    .str.split(",", expand=True)
)

inputs_expanded.columns = [f"input_{i}" for i in range(inputs_expanded.shape[1])]
inputs_expanded = inputs_expanded.dropna(axis=1).apply(pd.to_numeric)

df = pd.concat([inputs_expanded, df["label"].reset_index(drop=True)], axis=1)
print(df.head(10))

# Define the attributes
attributes = []
for col in df.columns:
    attributes.append((col, "NUMERIC"))

data = df.values.tolist()

# Écriture correcte avec liac-arff
arff.dump(
    "data/dataset_simpleclass.arff",
    data,
    relation="Diabete",
    names=[attr[0] for attr in attributes],
)
