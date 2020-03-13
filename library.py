import pandas as pd
from sklearn import preprocessing

def data_processing(location):
    column_names = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    raw_dataset = pd.read_csv\
        (location,
         names=column_names, na_values = "?", comment='\t',
         sep=",", skipinitialspace=True)
    wd = pd.DataFrame(raw_dataset[['wd']])
    x = raw_dataset[['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].values.astype(float)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled)
    raw_dataset = pd.concat([df_normalized, wd], axis=1)
    raw_dataset.columns = ['PM2.5', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd']
    return raw_dataset