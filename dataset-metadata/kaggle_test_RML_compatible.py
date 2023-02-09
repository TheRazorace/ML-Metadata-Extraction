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
print(len(datasets))
dataset_df = pd.DataFrame(datasets)[:2]
print(dataset_df)


for dataset in datasets:

    data = dict()

    for feature in dataset:

        #Hardcoded extraction of metadata
        #data ={"subtitleNullable": [dataset["subtitleNullable"]],"creatorNameNullable": dataset["creatorNameNullable"], "creatorUrlNullable" : dataset["creatorUrlNullable"], "creatorUrlNullable" : dataset["creatorUrlNullable"], "urlNullable" : dataset["urlNullable"], "licenseNameNullable" : dataset["licenseNameNullable"], "descriptionNullable" : dataset["descriptionNullable"], "ownerNameNullable" : dataset["ownerNameNullable"], "ownerRefNullable" : dataset["ownerRefNullable"], "titleNullable" : dataset["titleNullable"], "currentVersionNumberNullable" : dataset["currentVersionNumberNullable"], "usabilityRatingNullable" : dataset["usabilityRatingNullable"], "id" : dataset["id"], "ref" : dataset["ref"], "subtitle" : dataset["subtitle"], "hasSubtitle" : dataset["hasSubtitle"], "creatorName" : dataset["creatorName"], "hasCreatorName" : dataset["hasCreatorName"], "creatorUrl" : dataset["creatorUrl"], "hasCreatorUrl" : dataset["hasCreatorUrl"], "totalBytes" : dataset["totalBytes"], "hasTotalBytes" : dataset["hasTotalBytes"], "url" : dataset["url"], "hasUrl" : dataset["hasUrl"], "lastUpdated" : dataset["lastUpdated"], "downloadCount" : dataset["downloadCount"], "isPrivate" : dataset["isFeatured"], "licenseName" : dataset["licenseName"], "hasLicenseName" : dataset["hasLicenseName"], "description" : dataset["description"], "hasDescription" : dataset["hasDescription"], "ownerName" : dataset["ownerName"], "hasOwnerName" : dataset["hasOwnerName"], "ownerRef" : dataset["ownerRef"], "hasOwnerRef" : dataset["hasOwnerRef"], "kernelCount" : dataset["kernelCount"], "title" : dataset["title"], "hasTitle" : dataset["hasTitle"], "topicCount" : dataset["topicCount"], "viewCount" : dataset["viewCount"], "voteCount" : dataset["voteCount"], "currentVersionNumber" : dataset["currentVersionNumber"], "hasCurrentVersionNumber" : dataset["hasCurrentVersionNumber"], "usabilityRating" : dataset["usabilityRating"], "hasUsabilityRating" : dataset["hasUsabilityRating"]}
        if(feature != "tags" and feature != "versions" and feature != "files"):
            data[feature] = dataset[feature]


    dataset_df = pd.DataFrame(data, index = [0])
    dataset_name = "Dataset_examples/DatasetMD/DatasetMD_" + str(dataset["id"])
    dataset_df.to_csv(dataset_name, index=False, header=True)





# dataset_df = pd.DataFrame(datasets)[1:2]
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#   print(dataset_df)



dataset_dict = {d['id']: d['ref'] for d in datasets}
print("line 47")

#####TO DO

#Download a dataset using ref column: 
# # download specific dataset
# dataset = api.dataset_download_cli("ahsan81/hotel-reservations-classification-dataset")

for id, ref in dataset_dict.items():
    print("line 56")
    #Download dataset with reference ref
    dataset = api.dataset_download_cli(ref)
    file_list = os.listdir()

    # Search for the zip file
    zip_filename = ""
    for filename in file_list:
        if filename.endswith(".zip"):
            zip_filename = filename
            with zipfile.ZipFile(filename) as z:
                z.extractall(".")
                file_list = z.namelist()
                #metadata for fileMD
                files_count = []
                did_files = []
                file_name = []
                nr_of_rows = []
                nr_of_features = []
                file_counter = 0
                print("line 76")

                #metadata for featureMD
                feature_count = []
                did_features = []
                features_file_name = []
                feature_name = []
                feature_type = []
                feature_distinct = []
                feature_missing = []
                feature_counter = 0

                # Search for the csv files
                for filename in file_list:

                    if filename.endswith(".csv"):
                        csv_filename = filename
                        # convert csv to pd DataFrame
                        df = pd.read_csv(csv_filename)

                        print("line 96")

                        files_count.append(file_counter)
                        did_files.append(id)
                        file_name.append(csv_filename)
                        nr_of_rows.append(df.shape[0])
                        nr_of_features.append(df.shape[1] - 1)
                        file_counter += 1

                        for feature in list(df.columns)[1:]:
                            feature_count.append(feature_counter)
                            did_features.append(id)
                            features_file_name.append(csv_filename)
                            feature_name.append(feature)
                            feature_type.append(df[feature].dtype)
                            feature_distinct.append(len(df[feature].unique()))
                            feature_missing.append(df[feature].isnull().sum())
                            feature_counter += 1
                        # remove original csv file
                        os.remove(csv_filename)




                #Creating the file dataframe
                file_df = pd.DataFrame({
                    "file_id": files_count,
                    "dataset_id": did_files,
                    "file_name": file_name,
                    "nr_of_rows": nr_of_rows,
                    "nr_of_features": nr_of_features
                })

                #Creating the feature dataframe
                feature_df = pd.DataFrame({
                    "feature_id": feature_count,
                    "dataset_id": did_features,
                    "file_name": features_file_name,
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "feature_distinct": feature_distinct,
                    "feature_missing": feature_missing
                })

                # convert file metadata to csv
                file_name = "Dataset_examples/FileMD/fileMD_" + str(id)
                header = ["file_id", "dataset_id", "file_name", "nr_of_rows","nr_of_features"]
                file_df.to_csv(file_name, index =False,header=header)

                # convert feature metadata to csv
                feature_name = "Dataset_examples/FeatureMD/featureMD_" + str(id)
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

