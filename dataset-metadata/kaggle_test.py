from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import zipfile
import pandas as pd
import os

api = KaggleApi()
api.authenticate()

#Documentation--> https://github.com/Kaggle/kaggle-api


#get dataset metadata
datasets = kaggle.api.datasets_list(search="countries")
dataset_df = pd.DataFrame(datasets)[:1]
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(dataset_df)

print(dataset_df['id'])

#####TO DO

#Download a dataset using ref column: 
# download specific dataset
dataset = api.dataset_download_cli("ahsan81/hotel-reservations-classification-dataset")
print(dataset)

# Get a list of all files in the current directory
file_list = os.listdir()
#print(file_list)
counter = 0

# Search for the zip file
zip_filename = ""
for filename in file_list:
    if filename.endswith(".zip"):
        zip_filename = filename
        with zipfile.ZipFile(filename) as z:
            z.extractall(".")
            file_list = z.namelist()

            # Search for the csv files
            for filename in file_list:
                if filename.endswith(".csv"):
                    csv_filename = filename
                    # convert csv to pd DataFrame
                    df = pd.read_csv(csv_filename)
                    counter += 1
                    # extract metadata
                    metadata = {
                        "number_of_rows": df.shape[0],
                        "feature_count": df.shape[1],
                        "feature_names": list(df.columns),
                        "feature_type": list(df.dtypes),
                        "memory_usage": df.memory_usage().sum() / 1024 ** 2,
                        "dataframe_statistics": df.describe().transpose().to_dict(),
                        "missing_values": df.isnull().sum().to_dict(),
                        "feature_distinct": [df[col].nunique() for col in df.columns],
                        "value_counts": [df[col].value_counts().to_dict() for col in df.columns],
                    }

                    # convert metadata to csv
                    metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
                    metadata_name = "Dataset_examples/metadata_" + csv_filename

                    metadata_df.to_csv(metadata_name, header=False)
                    # remove original csv file
                    os.remove(csv_filename)

os.remove(zip_filename)


#
# with zipfile.ZipFile(zip_filename) as z:
#      z.extractall(".")
#      file_list = z.namelist()
#
#      for filename in file_list:
#         if filename.endswith(".csv"):
#             csv_filename = filename
#             break


#Save the dataset as pandas dataframe and delete the original dataset from your computer (for storage management)
# os.remove("hotel-reservations-classification-dataset.zip")
# os.remove(csv_filename)
#
#
# #Find some extra metadata (e.g column names, datatypes, number of features...)

#
# metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
# metadata_name = "Dataset_examples/metadata_" + csv_filename
#
# metadata_df.to_csv(metadata_name, header=False)



##### 


#get scripts related to first dataset 
# kernels = kaggle.api.kernels_list_with_http_info(dataset=list(dataset_df['ref'])[0], page_size=1)[0]
# kernel_df = pd.DataFrame(kernels)
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(kernel_df)

