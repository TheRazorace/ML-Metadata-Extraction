import pandas as pd
import matplotlib.pyplot as plt

# Load the original CSV file into a pandas DataFrame
df = pd.read_csv("Dataset_examples/IndiaTourism.csv")

# Extract metadata information
metadata = {
    "number_of_rows": df.shape[0],
    "number_of_columns": df.shape[1],
    "column_names": list(df.columns),
    "column_data_types": list(df.dtypes),
    "memory_usage": df.memory_usage().sum() / 1024 ** 2,
    "dataframe_statistics": df.describe().transpose().to_dict(),
    "missing_values": df.isnull().sum().to_dict(),
    "unique_values": [df[col].nunique() for col in df.columns],
    "value_counts": [df[col].value_counts().to_dict() for col in df.columns],
}

plt.plot(df['Domestic-2019-20'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Column')
plt.show()

# Save the metadata to a new CSV file
metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
metadata_df.to_csv("Dataset_examples/IndiaTourismMetadata.csv", header=False)



