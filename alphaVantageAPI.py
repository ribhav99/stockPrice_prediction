'''
Use the alpha vantage API with alpha_vantage wrapper.
Methods to save technical indicator values as JSON objects.
'''
import json
import pandas
import os
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

api_key = "VQZIWQ4GHCOFOGJD"
data_type = "JSON"
ti = TechIndicators(key=api_key, output_format=data_type)
ts = TimeSeries(key=api_key, output_format=data_type)


def write_to_file(file, data_type, data, metadata):
    if data_type == "JSON":
        json.dump(data, file)
        # file.write("\n\n\n\n\n\n\n\n\n\n\n\n\n")
        # json.dump(meta_data, file)
        print(metadata)


def get_RSI(symbol="MFC", interval='daily', time_period=15, series_type='close'):
    data, meta_data = ti.get_rsi(
        symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    file = open(symbol+'_'+'RSI.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_SMA(symbol="MFC", interval='daily', time_period=15, series_type='close'):
    data, meta_data = ti.get_sma(
        symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    file = open(symbol+'_'+'SMA.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_WMA(symbol="MFC", interval='daily', time_period=15, series_type='close'):
    data, meta_data = ti.get_wma(
        symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    file = open(symbol+'_'+'WMA.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_MOM(symbol="MFC", interval='daily', time_period=15, series_type='close'):
    data, meta_data = ti.get_mom(
        symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    file = open(symbol+'_'+'MOM.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_STOCH(symbol="MFC", interval='daily', slowkmatype=1, slowdmatype=1):
    data, meta_data = ti.get_stoch(
        symbol=symbol, interval=interval, slowkmatype=slowkmatype, slowdmatype=slowdmatype)

    file = open(symbol+'_'+'STOCH.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_MACD(symbol="MFC", interval='daily', series_type='close'):
    data, meta_data = ti.get_macd(
        symbol=symbol, interval=interval, series_type=series_type)

    file = open(symbol+'_'+'MACD.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_WILLR(symbol="MFC", interval='daily', time_period=15):
    data, meta_data = ti.get_willr(
        symbol=symbol, interval=interval, time_period=time_period)

    file = open(symbol+'_'+'WILLR.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_ADOSC(symbol="MFC", interval='daily'):
    data, meta_data = ti.get_adosc(
        symbol=symbol, interval=interval)

    file = open(symbol+'_'+'ADOSC.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


def get_CCI(symbol="MFC", interval='daily', time_period=15):
    data, meta_data = ti.get_cci(
        symbol=symbol, interval=interval, time_period=time_period)

    file = open(symbol+'_'+'CCI.'+data_type, "w")
    write_to_file(file, data_type, data, meta_data)
    file.close()

    return True


if __name__ == '__main__':
    symbol = "TSLA"
    if not os.path.isdir(symbol):
        os.mkdir(symbol)

    os.chdir(symbol)

    # data, meta_data = ts.get_intraday(
    #     symbol="JPM", interval='1min')
    # print(data)
    # print(meta_data)

    # get_RSI(symbol=symbol)
    # get_SMA(symbol=symbol)
    # get_WMA(symbol=symbol)
    # get_MOM(symbol=symbol)
    # get_STOCH(symbol=symbol)
    # get_MACD(symbol=symbol)
    # get_WILLR(symbol=symbol)
    # get_ADOSC(symbol=symbol)
    # get_CCI(symbol=symbol)
