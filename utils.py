import pandas as pd


def preprocess_dataset(dataset_path, output_path):
    dataset = pd.read_csv(dataset_path, sep=' ', header=None)
    dataset.drop(columns={138}, inplace=True)
    dataset.replace(to_replace=r'^.*:', value='', regex=True, inplace=True)

    dataset.rename(columns={0: 'relevance', 1: 'queryId'}, inplace=True)
    new_columns_name = {key: value for key, value in zip(range(2, 138), range(1, 137))}
    dataset.rename(columns=new_columns_name, inplace=True)

    dataset = dataset.astype({key: 'float32' for key in range(1, 137)})
    dataset = dataset.astype({'relevance': 'int8', 'queryId': 'int32'})

    store = pd.HDFStore(output_path+'/'+'processed_train.h5')
    store['processed_train'] = dataset
    store.close()


def load_dataframe(dataset_path):
    dataset_store = pd.HDFStore(dataset_path, 'r')
    dataset = dataset_store['processed_train']
    dataset_store.close()
    return dataset


