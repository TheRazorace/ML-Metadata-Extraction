import openml
import pandas as pd

#openML configuration
openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'

#Get 10 datasets metadata and save 1 as a dataframe
dlist = openml.datasets.list_datasets(size=10)
dataset_df = pd.DataFrame.from_dict(dlist, orient="index").reset_index()[:1]
dataset_df['index']=dataset_df.index

idlist = list(dataset_df['did'])
 
#Add more metadata using get_dataset() function
dataset_df['cache_format'] = [openml.datasets.get_dataset(did).cache_format for did in idlist]
dataset_df['description'] = [openml.datasets.get_dataset(did).description for did in idlist]
dataset_df['creator'] = [openml.datasets.get_dataset(did).creator for did in idlist]
dataset_df['contributor'] = [openml.datasets.get_dataset(did).contributor for did in idlist]
dataset_df['collection_date'] = [openml.datasets.get_dataset(did).collection_date for did in idlist]
dataset_df['upload_date'] = [openml.datasets.get_dataset(did).upload_date for did in idlist]
dataset_df['language'] = [openml.datasets.get_dataset(did).language for did in idlist]
dataset_df['licence'] = [openml.datasets.get_dataset(did).licence for did in idlist]
dataset_df['url'] = [openml.datasets.get_dataset(did).url for did in idlist]
dataset_df['default_target_attribute'] = [openml.datasets.get_dataset(did).default_target_attribute for did in idlist]
dataset_df['row_id_attribute'] = [openml.datasets.get_dataset(did).row_id_attribute for did in idlist]
dataset_df['ignore_attribute'] = [openml.datasets.get_dataset(did).ignore_attribute for did in idlist]
dataset_df['version_label'] = [openml.datasets.get_dataset(did).version_label for did in idlist]
dataset_df['citation'] = [openml.datasets.get_dataset(did).citation for did in idlist]
dataset_df['tag'] = [openml.datasets.get_dataset(did).tag for did in idlist]
dataset_df['visibility'] = [openml.datasets.get_dataset(did).visibility for did in idlist]
dataset_df['original_data_url'] = [openml.datasets.get_dataset(did).original_data_url for did in idlist]
dataset_df['paper_url'] = [openml.datasets.get_dataset(did).paper_url for did in idlist]
dataset_df['md5_checksum'] = [openml.datasets.get_dataset(did).md5_checksum for did in idlist]

#creation of separate creators df for dataset creators
creators_per_dataset = list(dataset_df['creator'])
did_creators = []
creators = []
for i in range(len(idlist)):
    for j in range(len(creators_per_dataset[i])):
        did_creators.append(idlist[i])
        creators.append(creators_per_dataset[i][j])
        
creator_df = pd.DataFrame()
creator_df['did_per_creator'] = did_creators
creator_df['creator_per_dataset'] = creators 

#creation of separate tags df for dataset tags
tags_per_dataset = list(dataset_df['tag'])
did_creators = []
tags = []
for i in range(len(idlist)):
    for j in range(len(tags_per_dataset[i])):
        did_creators.append(idlist[i])
        tags.append(tags_per_dataset[i][j])
        
tag_df = pd.DataFrame()
tag_df['did_per_tag'] = did_creators
tag_df['tag_per_dataset'] = tags 


#creation of separate features df for dataset features
did_features = []
feature_name = []
feature_type = []
feature_distinct = []
feature_missing = []
feature_count = []
for did in idlist:
    counter = 0
    dataset = openml.datasets.get_dataset(did)
    features = dataset.get_data()[3]
    for feature in features:
        feature_count.append(counter)
        did_features.append(did)
        feature_name.append(feature)
        feature_type.append(dataset.get_data(feature)[1].dtype)
        feature_distinct.append(len(dataset.get_data(feature)[1].unique()))
        feature_missing.append(dataset.get_data(feature)[1].isnull().sum())
        counter += 1
        
feature_df = pd.DataFrame()
feature_df['did'] = did_features
feature_df['feature_id'] = feature_count
feature_df['name'] = feature_name
feature_df['type'] = feature_type
feature_df['distinct_values'] = feature_distinct
feature_df['missing_values'] = feature_missing


#print and save dfs as csvs to use for mappings
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(dataset_df)

# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(feature_df)

# compression_opts = dict(method='zip', archive_name='datasets2.csv')  
# dataset_df.to_csv('datasets.zip', index=False, compression=compression_opts)  

# compression_opts = dict(method='zip', archive_name='creators2.csv')  
# creator_df.to_csv('creators.zip', index=False, compression=compression_opts) 

# compression_opts = dict(method='zip', archive_name='features2.csv')  
# feature_df.to_csv('features.zip', index=False, compression=compression_opts) 

# compression_opts = dict(method='zip', archive_name='tags2.csv')  
# tag_df.to_csv('tags.zip', index=False, compression=compression_opts) 

