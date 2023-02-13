from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import zipfile
import pandas as pd
import os


def datasetMD_extraction(data_df):
    dataset_df = data_df.drop(['versions', 'files','tags'], axis=1)
    dataset_name = "Dataset_examples/DatasetMD/Kaggle_DatasetMD"
    dataset_df.to_csv(dataset_name, index=False, header=True)

def tagsMD_extraction(data_df):
    tags_per_dataset = list(data_df['tags'])
    did_creators = []
    tags = []
    for i in range(len(idlist)):
        for j in range(len(tags_per_dataset[i])):
            did_creators.append(idlist[i])
            tags.append(tags_per_dataset[i][j])

    tag_df = pd.DataFrame()
    tag_df['did_per_tag'] = did_creators
    tag_df['tag_per_dataset'] = tags

    parsed_tags = []
    did = []
    column_name = []
    column_description = []
    col_name = tag_df.iloc[0]

    for i in range(len(tag_df)):
        row = tag_df.iloc[i]
        desired = row['tag_per_dataset']
        keys = desired.keys()

        for key in keys:
            did.append(row['did_per_tag'])
            column_name.append(key)
            column_description.append(desired[key])

        parsed_tags = pd.DataFrame({
            'did': did,
            'column_name': column_name,
            'column_description': column_description
        })

    parsed_tag_df = pd.DataFrame(parsed_tags)

    tags_name = "Dataset_examples/DatasetMD/Kaggle_DatasetTags"
    parsed_tag_df.to_csv(tags_name, index=False, header=True)


def features_files_extraction(data_dict, ):
    # metadata for fileMD
    files_count = []
    did_files = []
    file_name = []
    nr_of_rows = []
    nr_of_features = []
    file_counter = 0

    # metadata for featureMD
    feature_count = []
    did_features = []
    features_file_name = []
    feature_name = []
    feature_type = []
    feature_distinct = []
    feature_missing = []
    feature_counter = 0

    for id, ref in data_dict.items():
        # Download dataset with reference ref
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

                    # Search for the csv files
                    for filename in file_list:

                        if filename.endswith(".csv"):
                            csv_filename = filename
                            # convert csv to pd DataFrame
                            df = pd.read_csv(csv_filename)

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

        # Creating the file dataframe
        file_df = pd.DataFrame({
            "file_id": files_count,
            "dataset_id": did_files,
            "file_name": file_name,
            "nr_of_rows": nr_of_rows,
            "nr_of_features": nr_of_features
        })

        # Creating the feature dataframe
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
        fileNAME = "Dataset_examples/FileMD/Kaggle_FileMD"
        header = ["file_id", "dataset_id", "file_name", "nr_of_rows", "nr_of_features"]
        file_df.to_csv(fileNAME, index=False, header=header)

        # convert feature metadata to csv
        featureNAME = "Dataset_examples/FeatureMD/Kaggle_FeatureMD"
        header = ["feature_id", "dataset_id", "file_name", "feature_name", "feature_type", "feature_distinct",
                  "feature_missing"]
        feature_df.to_csv(featureNAME, index=False, header=header)

        os.remove(zip_filename)


if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()

    # Documentation--> https://github.com/Kaggle/kaggle-api

    # get dataset metadata
    datasets = kaggle.api.datasets_list(search="countries")
    print(len(datasets))
    dataset_df = pd.DataFrame(datasets)[:6]
    idlist = list(dataset_df['id'])
    datasetMD_extraction(dataset_df)
    ######### TODO
    # Add tags_df
    # tags_df = dataset_df.filter(['id', 'tags'])
    # pd.options.display.max_columns = None
    # print(tags_df)

    tagsMD_extraction(dataset_df)




    dataset_dict = {d['id']: d['ref'] for d in dataset_df.to_dict(orient='records')}

    features_files_extraction(dataset_dict)

#
# with zipfile.ZipFile(zip_filename) as z:
#      z.extractall(".")
#      file_list = z.namelist()
#
#      for filename in file_list:
#         if filename.endswith(".csv"):
#             csv_filename = filename
#             break


# Save the dataset as pandas dataframe and delete the original dataset from your computer (for storage management)
# os.remove("hotel-reservations-classification-dataset.zip")
# os.remove(csv_filename)
#
#
# #Find some extra metadata (e.g column names, datatypes, number of features...)
# #Save metadata in a new dataframe


#####


# get scripts related to first dataset
# kernels = kaggle.api.kernels_list_with_http_info(dataset=list(dataset_df['ref'])[0], page_size=1)[0]
# kernel_df = pd.DataFrame(kernels)
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(kernel_df)
