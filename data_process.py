import talib
import numpy as np
import math
import pandas as pd
import talib as ta

def get_df(filename):
    tech = pd.read_csv(filename,index_col=0)
    dclose = np.array(tech.close)
    volume = np.array(tech.volume)
    tech['RSI'] = ta.RSI(np.array(tech.close))
    tech['OBV'] = ta.OBV(np.array(tech.close),np.array(tech.volume))
    tech['NATR'] = ta.NATR(np.array(tech.high),np.array(tech.low),np.array(tech.close))
    tech['upper'],tech['middle'],tech['lower'] = ta.BBANDS(np.array(tech.close), timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
    tech['DEMA'] = ta.DEMA(dclose, timeperiod=30)
    tech['EMA'] = ta.EMA(dclose, timeperiod=30)
    tech['HT_TRENDLINE'] = ta.HT_TRENDLINE(dclose)
    tech['KAMA'] = ta.KAMA(dclose, timeperiod=30)
    tech['MA'] = ta.MA(dclose, timeperiod=30, matype=0)
#    tech['mama'], tech['fama'] = ta.MAMA(dclose, fastlimit=0, slowlimit=0)
    tech['MIDPOINT'] = ta.MIDPOINT(dclose, timeperiod=14)
    tech['SMA'] = ta.SMA(dclose, timeperiod=30)
    tech['T3'] = ta.T3(dclose, timeperiod=5, vfactor=0)
    tech['TEMA'] = ta.TEMA(dclose, timeperiod=30)
    tech['TRIMA'] = ta.TRIMA(dclose, timeperiod=30)
    tech['WMA'] = ta.WMA(dclose, timeperiod=30)
    tech['APO'] = ta.APO(dclose, fastperiod=12, slowperiod=26, matype=0)
    tech['CMO'] = ta.CMO(dclose, timeperiod=14)
    tech['macd'], tech['macdsignal'], tech['macdhist'] = ta.MACD(dclose, fastperiod=12, slowperiod=26, signalperiod=9)
    tech['MOM'] = ta.MOM(dclose, timeperiod=10)
    tech['PPO'] = ta.PPO(dclose, fastperiod=12, slowperiod=26, matype=0)
    tech['ROC'] = ta.ROC(dclose, timeperiod=10)
    tech['ROCR'] = ta.ROCR(dclose, timeperiod=10)
    tech['ROCP'] = ta.ROCP(dclose, timeperiod=10)
    tech['ROCR100'] = ta.ROCR100(dclose, timeperiod=10)
    tech['RSI'] = ta.RSI(dclose, timeperiod=14)
    tech['fastk'], tech['fastd'] = ta.STOCHRSI(dclose, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    tech['TRIX'] = ta.TRIX(dclose, timeperiod=30)
    tech['OBV'] = ta.OBV(dclose,volume)
    tech['HT_DCPHASE'] = ta.HT_DCPHASE(dclose)
    tech['inphase'], tech['quadrature'] = ta.HT_PHASOR(dclose)
    tech['sine'], tech['leadsine'] = ta.HT_SINE(dclose)
    tech['HT_TRENDMODE'] = ta.HT_TRENDMODE(dclose)
    df = tech.fillna(method='bfill')
    return df

def MA(data,window):
    MA = [0]*(window-1)
    for i in range(len(data)-window+1):
        MA.append(np.mean(data[i:i+window]))
    return MA
def vov(data,window):
    VO = [0]*(window-1)
    for i in range(len(data)-window+1):
        VO.append(np.std(data[i:i+window]))
    return VO
def vod(data):
    k = (data[-1]-data[0])/(len(data)-1)
    d = []
    for i in range(len(data)):
        d.append(data[i]-data[0]-k*i)
    d = np.array(d)
    mn = np.mean(np.absolute(d))
    st = np.sqrt(np.sum(d**2))
    return mn,st
def vod_father(data,window):
    MEAN = [0]*(window-1)
    STD = [0]*(window-1)
    for i in range(len(data)-window+1):
        mn,st = vod(data[i:i+window])
        MEAN.append(mn)
        STD.append(st)
    return MEAN,STD

def ind(filename):
    filename = filename
    data = get_df(filename)

    close = data.close

    data['MA5'] = MA(close,5)
    data['MA10'] = MA(close,10)
    data['MA20'] = MA(close,20)
    data['volitility'] = (data['high']-data['low'])/data['open']

    data['VO5'] = vov(close,5)
    data['VO10'] = vov(close,10)
    data['VO20'] = vov(close,20)

    data['close_off_high'] = (data.high - data.close) / (data.high - data.low) - 1
    data['pct_change'] = (data.close.shift(-1) -data.close)/data.close
    data['pct_change'] = data['pct_change'].shift(1)

    data["Oma5"],data["Osd5"] =  vod_father(close,5)
    data["Oma10"],data["Osd10"] =  vod_father(close,10)
    data["Oma15"],data["Osd15"] =  vod_father(close,15)

    data = data.drop(['open', 'macdsignal','leadsine','high','ROCR', 'ROCR100','Oma5','Osd5','Oma15','Osd15','MA5','MA20','VO5','VO20','ROCP', 'low','upper','middle', 'lower', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA','MIDPOINT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA','close_off_high'],axis=1)
    return data[60:]

if __name__=="__main__":
    print(ind())
