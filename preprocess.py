'''
All indicators must already be saved as JSON objects.
Combines all data into 1 table
'''
import pandas as pd
import ast
import os

data_type = "JSON"
symbol = "TSLA"

indicators = ["RSI", "SMA", "WMA", "MOM", "STOCH", "MACD", "ADOSC", "CCI"]


def create_indicator_dataframe(indicator):
    indicator = indicator.upper()
    file = open(symbol+"_"+indicator+"."+data_type, 'r')
    contents = file.read()
    data = ast.literal_eval(contents)
    file.close()
    df = pd.DataFrame(data).transpose()
    original_cols = df.columns.tolist()
    df.reset_index(level=0, inplace=True)
    df.columns = ["Date"] + original_cols
    return df


if __name__ == "__main__":
    os.chdir(symbol)
    df = pd.read_csv(symbol+".csv")

    for i in indicators:
        df = df.merge(create_indicator_dataframe(i), on="Date")

    df.to_csv(symbol+"_final.csv")
