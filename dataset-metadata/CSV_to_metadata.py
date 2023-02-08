import pandas as pd
import matplotlib.pyplot as plt

# Load the original CSV file into a pandas DataFrame
df = pd.read_csv("Dataset_examples/IndiaTourism.csv")

# Extract metadata information
feature_df = pd.DataFrame()
feature_df['did'] = list(df.columns)
feature_df['feature_id'] = df.shape[1]
feature_df['name'] = list(df.columns)
feature_df['type'] = list(df.dtypes)
feature_df['distinct_values'] = [df[col].nunique() for col in df.columns]
feature_df['missing_values'] = df.isnull().sum().to_dict()


# metadata = {
#     "number_of_rows": df.shape[0],
#     "feature_count": df.shape[1],
#     "feature_names": list(df.columns),
#     "feature_type": list(df.dtypes),
#     "memory_usage": df.memory_usage().sum() / 1024 ** 2,
#     "dataframe_statistics": df.describe().transpose().to_dict(),
#     "missing_values": df.isnull().sum().to_dict(),
#     "feature_distinct": [df[col].nunique() for col in df.columns],
#     "value_counts": [df[col].value_counts().to_dict() for col in df.columns],
# }

plt.plot(df['Domestic-2019-20'])
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Column')
plt.show()

# Save the metadata to a new CSV file
metadata_df = pd.DataFrame.from_dict(feature_df, orient='index')
metadata_df.to_csv("Dataset_examples/IndiaTourismMetadata.csv", header=False)



