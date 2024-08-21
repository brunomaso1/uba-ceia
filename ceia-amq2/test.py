import kagglehub
import kagglehub.config
import pandas as pd

def get_data():
    # Download the latest version.
    kagglehub.config.DEFAULT_CACHE_FOLDER = 'data'
    # resp = kagglehub.dataset_download('jsphyg/weather-dataset-rattle-package/versions/2', path='weatherAUS.csv', force_download=True)
    df = pd.read_csv("data/datasets/jsphyg/weather-dataset-rattle-package/versions/2/weatherAUS.csv", compression='zip')
    print(df.info())

get_data()