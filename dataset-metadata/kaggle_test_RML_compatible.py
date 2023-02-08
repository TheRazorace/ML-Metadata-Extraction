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

dataset_dict = {d['id']: d['ref'] for d in datasets}

print(dataset_dict)

#####TO DO

#Download a dataset using ref column: 
# # download specific dataset
# dataset = api.dataset_download_cli("ahsan81/hotel-reservations-classification-dataset")

for id, ref in dataset_dict.items():

    dataset = api.dataset_download_cli(ref)
    print(dataset)

    #dataset_metadata = api.dataset_metadata_cli(ref)
    #print(dataset_metadata)

    #dataset_name = "Dataset_examples/datasetMD_" + str(id)
    #dataset_df.to_csv(dataset_name, index=False, header=False)




    # Get a list of all files in the current directory
    file_list = os.listdir()
    #print(file_list)

    # Search for the zip file
    zip_filename = ""
    for filename in file_list:
        if filename.endswith(".zip"):
            zip_filename = filename
            with zipfile.ZipFile(filename) as z:
                z.extractall(".")
                file_list = z.namelist()
                feature_count = []
                did_features = []
                file_name = []
                feature_name = []
                feature_type = []
                feature_distinct = []
                feature_missing = []
                feature_count = []
                counter = 0

                # Search for the csv files
                for filename in file_list:

                    if filename.endswith(".csv"):
                        csv_filename = filename
                        # convert csv to pd DataFrame
                        df = pd.read_csv(csv_filename)

                        for feature in list(df.columns)[1:]:
                            feature_count.append(counter)
                            did_features.append(id)
                            file_name.append(csv_filename)
                            feature_name.append(feature)
                            feature_type.append(df[feature].dtype)
                            feature_distinct.append(len(df[feature].unique()))
                            feature_missing.append(df[feature].isnull().sum())
                            counter += 1
                        # remove original csv file
                        os.remove(csv_filename)




                    feature_df = pd.DataFrame({
                        "feature_id": feature_count,
                        "dataset_id": did_features,
                        "file_name": file_name,
                        "feature_name": feature_name,
                        "feature_type": feature_type,
                        "feature_distinct": feature_distinct,
                        "feature_missing": feature_missing
                    })

                # convert metadata to csv
                feature_name = "Dataset_examples/featureMD_" + str(id)
                header = ["feature_id", "dataset_id", "file_name", "feature_name","feature_type","feature_distinct","feature_missing"]
                feature_df.to_csv(feature_name, index =False,header=header)

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
# #Save metadata in a new dataframe




##### 


#get scripts related to first dataset 
# kernels = kaggle.api.kernels_list_with_http_info(dataset=list(dataset_df['ref'])[0], page_size=1)[0]
# kernel_df = pd.DataFrame(kernels)
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(kernel_df)

