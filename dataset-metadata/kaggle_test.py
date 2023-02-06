from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import zipfile
import pandas as pd

api = KaggleApi()
api.authenticate()

#Documentation--> https://github.com/Kaggle/kaggle-api


#get dataset metadata
datasets = kaggle.api.datasets_list(search="countries")
dataset_df = pd.DataFrame(datasets)[:1]
with pd.option_context('display.max_columns', None, 'display.max_rows', None):
    print(dataset_df)

#####TO DO

#Download a dataset using ref column: 
# download specific dataset
# dataset = api.dataset_download_cli("ahsan81/hotel-reservations-classification-dataset")  
# with zipfile.ZipFile("hotel-reservations-classification-dataset.zip") as z:
#     z.extractall(".")

#Save the dataset as pandas dataframe and delete the original dataset from your computer (for storage management)


#Find some extra metadata (e.g column names, datatypes, number of features...)
#Save metadata in a new dataframe

##### 


#get scripts related to first dataset 
# kernels = kaggle.api.kernels_list_with_http_info(dataset=list(dataset_df['ref'])[0], page_size=1)[0]
# kernel_df = pd.DataFrame(kernels)
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(kernel_df)

