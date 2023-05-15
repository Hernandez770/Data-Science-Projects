import pandas as pd
def load_data():
    df = pd.read_csv('/Users/adriangarcia/Desktop/Project_ML_Adri/data/processed/merged_20_years.csv')  
    df = df[['SP.DYN.LE00.IN','NY.GDP.MKTP.KD','SP.POP.TOTL','AG.SRF.TOTL.K2']]
    df = df.dropna()
    return df