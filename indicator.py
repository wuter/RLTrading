#coding=utf-8
import talib
import numpy as np
import math
import pandas as pd
np.set_printoptions(threshold=np.inf)

global num
num = 0

#第一个字典是标号和指标的对应，第二个字典是指标与数值的对应
dict1 = {}
dict2 = {}

def Add(indicator,values):
    global num
    dict1[num] = indicator
    dict2[indicator] = values
    num = num + 1

#将所有的指标添加进两个字典中
#布林线
def indicator(filename):
    database = pd.read_csv(filename)
    close =  np.array(database.close)
    high =  np.array(database.high)
    low =  np.array(database.low)
    volume =  np.array(database.volume)
    o =  np.array(database.open)
    #简单区分其到底处于什么区间内
    Add('OPEN',o)
    Add('HIGH',high)
    Add('LOW',low)
    Add('CLOSE',close)
    Add('VOLUME',volume)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    Length = len(upperband)
    increase = []
    for i in range(0,Length):
        if math.isnan(upperband[i]):
            increase.append(np.nan)
        else:
            increase.append(upperband[i]-middleband[i])
    Add('BBANDS',np.asarray(increase))

    real = talib.DEMA(close, timeperiod=10)
    real1 = talib.DEMA(close, timeperiod=20)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('DEMA',real0)

    real = talib.EMA(close, timeperiod=5)
    real1 = talib.EMA(close, timeperiod=10)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('EMA',real0)

    #暂时不会用这个指标
    real = talib.HT_TRENDLINE(close)
    Add('HT_TRENDLINE',real)

    real = talib.KAMA(close, timeperiod=30)
    real1 = talib.KAMA(close, timeperiod=60)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('KAMA',real0)

    real = talib.MA(close, timeperiod=7, matype=0)
    real1 = talib.MA(close, timeperiod=14,matype=0)
    real0 = []
    for i in range(Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('MA',real0)

    #暂时没找到怎么去用
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(mama[i] - fama[i])
        else:
            real0.append(np.nan)
    Add('MAMA',np.asarray(real0))

    #没找到
    real = talib.MIDPOINT(close, timeperiod=14)
    Add('MIDPOINT',real)

    #没找到
    real = talib.MIDPRICE(high, low, timeperiod=14)
    Add('MIDPRICE',real)


    real = talib.SAR(high, low, acceleration=0, maximum=0)
    real0 = []
    for i in range(0,Length):
        if not math.isnan(real[i]):
            real0.append(close[i]-real[i])
        else:
            real0.append(np.nan)
    Add('SAR',real0)

    #暂时不会
    real = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    Add('SAREXT',real)

    real = talib.SMA(close, timeperiod=3)
    real1 = talib.SMA(close, timeperiod=5)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('SMA',real0)

    #暂时不懂
    real = talib.T3(close, timeperiod=5, vfactor=0)
    Add('T3',real)

    real = talib.TEMA(close, timeperiod=7)
    real1 = talib.TEMA(close, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('TEMA',real0)

    real = talib.TRIMA(close, timeperiod=7)
    real1 = talib.TRIMA(close, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('TRIMA',real0)

    real = talib.WMA(close, timeperiod=7)
    real1 = talib.WMA(close, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('WMA',real0)

    #ADX与ADXR的关系需要注意一下
    real = talib.ADX(high, low, close, timeperiod=14)
    Add('ADX',real)

    real = talib.ADXR(high, low, close, timeperiod=14)
    Add('ADXR',real)

    #12个和26个简单移动平均线的差值
    real = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    Add('APO',real)

    '''
    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not(math.isnan(aroondown) or math.isnan(aroonup)):
            real0.append(aroonup[i] - aroondown[i])
        else:
            real0.append(numpy.nan)
    Add('AROON',numpy.asarray(real0))
    '''
    #AROONOSC就是Aroonup-aroondown
    real = talib.AROONOSC(high, low, timeperiod=14)
    Add('AROONOSC',real)

    #不懂
    real = talib.BOP(o, high, low, close)
    Add('BOP',real)

    #
    real = talib.CCI(high, low, close, timeperiod=14)
    Add('CCI',real)

    real = talib.CMO(close, timeperiod=14)
    Add('CMO',real)

    #需要再考虑一下，因为DX代表的市场的活跃度
    real = talib.DX(high, low, close, timeperiod=14)
    Add('DX',real)

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    Add('MACD',macdhist)

    macd, macdsignal, macdhist = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    Add('MACDEXT',macdhist)

    macd, macdsignal, macdhist = talib.MACDFIX(close, signalperiod=9)
    Add('MACDFIX',macdhist)

    real = talib.MFI(high, low, close, volume, timeperiod=14)
    real1 = talib.MA(real,7)
    real0 = []
    for i in range(0,Length):
        if not(math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('MFI',np.asarray(real0))

    real = talib.MINUS_DI(high, low, close, timeperiod=14)
    real1 = talib.PLUS_DI(high, low, close, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not(math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real1[i] - real[i])
        else:
            real0.append(np.nan)
    Add('PLUS_DI',np.asarray(real0))

    real = talib.MINUS_DM(high, low, timeperiod=14)
    Add('MINUS_DM',real)

    #虽然大概了解了规则，但在标普500上怎么用还不是很清楚
    real = talib.MOM(close, timeperiod=14)
    Add('MOM',real)

    real = talib.PLUS_DM(high, low, timeperiod=14)
    Add('PLUS_DM',real)

    #暂时不用
    real = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    Add('PPO',real)

    real = talib.ROC(close, timeperiod=14)
    Add('ROC',real)

    real = talib.ROCP(close, timeperiod=14)
    Add('ROCP',real)

    real = talib.ROCR(close, timeperiod=14)
    Add('ROCR',real)

    real = talib.ROCR100(close, timeperiod=14)
    Add('ROCR100',real)

    real = talib.RSI(close, timeperiod=14)
    Add('RSI',real)
    
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowj = []
    for i in range(Length):
        if not (math.isnan(slowk[i]) or math.isnan(slowd[i])):
            slowj.append(3*slowk[i] - 2*slowd[i])
        else:
            slowj.append(np.nan)
    Add('STOCH',np.asarray(slowj))

    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastj = []
    for i in range(Length):
        if not(math.isnan(fastk[i]) or math.isnan(fastd[i])):
            fastj.append(3*fastk[i] - 2*fastd[i])
        else:
            fastj.append(np.nan)
    Add('STOCHF',np.asarray(fastj))

    fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastj = []
    for i in range(Length):
        if not(math.isnan(fastk[i]) or math.isnan(fastd[i])):
            fastj.append(3*fastk[i] - 2*fastd[i])
        else:
            fastj.append(np.nan)
    Add('STOCHRSI',np.asarray(fastj))

    real = talib.TRIX(close, timeperiod=30)
    real1 = talib.MA(real,6)
    real0 = []
    for i in range(0,Length):
        if not (math.isnan(real[i] or math.isnan(real1[i]))):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('TRIX',real)

    real = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    Add('ULTOSC',real)

    real = talib.WILLR(high, low, close, timeperiod=14)
    real0 = []
    for i in range(0,Length):
        if not math.isnan(real[i]):
            if real[i] > -20:
                real0.append(1.0)
            elif real[i] < -80:
                real0.append(-1.0)
            else:
                real0.append(0.0)
        else:
            real0.append(np.nan)
    Add('WILLR',np.asarray(real0))

    real = talib.AD(high, low, close, volume)
    real1 = talib.MA(real,6)
    real0 = []
    for i in range(0,Length):
        if not(math.isnan(real[i]) or math.isnan(real1[i])):
            real0.append(real[i] - real1[i])
        else:
            real0.append(np.nan)
    Add('AD',np.asarray(real0))

    real = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    Add('ADOSC',real)

    #对于每个指标的处理还是很有问题的呀
    real = talib.OBV(close, volume)
    Add('OBV',real)

    real = talib.ATR(high, low, close, timeperiod=14)
    Add('ATR',real)

    real = talib.NATR(high, low, close, timeperiod=14)
    Add('NATR',real)

    real = talib.TRANGE(high, low, close)
    Add('TRANGE',real)

    integer = talib.HT_TRENDMODE(close)
    Add('HT_TRENDMODE',integer)

    real = talib.LINEARREG_SLOPE(close, timeperiod=14)
    Add('LINEARREG_SLOPE',real)

    real = talib.STDDEV(close, timeperiod=5, nbdev=1)
    Add('STDDEV',real)

    real = talib.TSF(close, timeperiod=14)
    Add('TSF',real)

    real = talib.VAR(close, timeperiod=5, nbdev=1)
    Add('VAR',real)

    real = talib.MEDPRICE(high, low)
    Add('MEDPRICE',real)

    real = talib.TYPPRICE(high, low, close)
    Add('TYPPRICE',real)

    real = talib.WCLPRICE(high, low, close)
    Add('WCLPRICE',real)

    real = talib.DIV(high, low)
    Add('DIV',real)

    real = talib.MAX(close, timeperiod=30)
    Add('MAX',real)

    real = talib.MIN(close, timeperiod=30)
    Add('MIN',real)

    real = talib.SUB(high, low)
    Add('SUB',real)

    real = talib.SUM(close, timeperiod=30)
    Add('SUM',real)
    
    return [dict1,dict2]

def getNum(filename):
    dict0 = indicator(filename)
    dict1 = dict0[0]
    dict2 = dict0[1]

    mmax = 0
    for i in range(num):
        ind = dict1[i]
        value = dict2[ind]
        length = len(value)
        tmp_max = 0
        for j in range(length):
            if not math.isnan(value[j]):
                tmp_max = j
                break

        if tmp_max>mmax:
            mmax = tmp_max

    return dict0,mmax,num


if __name__ == '__main__':
    dict0,mmax,num = getNum("data/600101.XSHG.csv")
    data = pd.DataFrame(dict0[1])[num:]
    print(data.shape)
