import pandas as pd
import arff

# Load the CSV
df = pd.read_csv("data/diabetes.csv")

# Define the attributes
attributes = []
for col in df.columns:
    attributes.append((col, "NUMERIC"))

data = df.values.tolist()

# Écriture correcte avec liac-arff
arff.dump(
    "data/dataset_simpleclass.arff",
    data,
    relation="Diabetes",
    names=[attr[0] for attr in attributes],
)
