import pandas as pd
import zipfile
import os


# Downloading the Datasets from Kaggle.

def get_dataset(path, file):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(path, file_name=file, path='Data')

    except ConnectionError:
        print('An error has occurred, please check information provided and try again.')

    else:
        print("Connection Made, Your File is now ready")

    finally:
        print('Process Finished')


def un_zip_folders(file_path):
    folders = os.listdir(file_path)
    for folder in folders:
        if folder.endswith('.zip'):
            filePath = file_path + '/' + folder
            zip_file = zipfile.ZipFile(filePath)
            for names in zip_file.namelist():
                zip_file.extract(names, file_path)
            zip_file.close()


paths = ['cdc/behavioral-risk-factor-surveillance-system', 'kmader/skin-cancer-mnist-ham10000']

files = ['2015.csv', '']

for path, file in zip(paths, files):
    get_dataset(path, file)


un_zip_folders('Data')
