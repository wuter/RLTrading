import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from random import sample
from indicator import getNum
#from data_process import ind


class Market(object):
    def __init__(self, filename, size, train_size, test_size, valid_size, flag):
        self.time = 0
        self.state = 0
        self.panish = 0.0001
        self.done = False
        self.cash = [10000]
        self.action_space = [0,1,2]
        self.opflag = flag

        self.filename = 'data/'+filename
        self.size = size
        self.train_size = train_size
        self.test_size = test_size
        self.valid_size = valid_size

        self.train_observation, self.train_close, self.test_observation, self.test_close, self.valid_observation, self.valid_close= self.get_data()
        self.process()

        self.observation_space = self.train_observation
        self.close = self.train_close
        #print("train_len:",len(self.train_observation))
        #print("test_len:",len(self.test_observation))
        #print("valid_len:",len(self.valid_observation))
        

    def get_data(self):
        dict0,mmax,num = getNum(self.filename)
        data = pd.DataFrame(dict0[1])[mmax:]
        close = np.array(data.CLOSE)[:int(len(data)*self.size)]
        #print(self.size)
        X = data.values[:int(len(data)*self.size)]
        train_start = 0
        train_end = int(len(X)*self.train_size)
        test_end = int(len(X)*self.train_size)+ int(len(X)*self.test_size)
        valid_end = int(len(X)*self.train_size)+ int(len(X)*self.test_size)+int(len(X)*self.valid_size)

        return X[train_start:train_end],close[train_start:train_end], X[train_end:test_end],close[train_end:test_end],X[test_end:valid_end],close[test_end:valid_end]

    def process(self):
        self.stander = StandardScaler().fit(self.train_observation)
        self.train_observation = self.stander.transform(self.train_observation)
        self.test_observation = self.stander.transform(self.test_observation)
        self.valid_observation = self.stander.transform(self.valid_observation)
    
    def step(self, a):
        self.time += 1
        if self.opflag:
            return self.future_step(a)
        else:
            return self.stock_step(a)

    def future_step(self,a):
        if a==2 :
            reward = (self.close[self.time] - self.close[self.time-1])/self.close[self.time-1] - self.panish*(1-self.state)
            self.state = 1
        elif a==0 :
            reward = -(self.close[self.time] - self.close[self.time-1])/self.close[self.time-1] - self.panish*(self.state+1)
            self.state = -1
        else:
            reward = -self.panish * abs(self.state)
            self.state = 0

        if self.time==len(self.observation_space)-1:
            self.done = True

        self.cash.append(self.cash[-1]*(1+reward))

        return np.append(self.observation_space[self.time], self.state),reward,self.done,self.state

    def stock_step(self,a):
        if a==2 and self.state==0:
            self.state = 1
            reward = (self.close[self.time] - self.close[self.time-1])/self.close[self.time-1]-self.panish
        elif a==0 and self.state==1:
            self.state = 0
            reward = -self.panish
        elif self.state==1:
            reward = (self.close[self.time] - self.close[self.time-1])/self.close[self.time-1]
        else:
            reward = 0

        if self.time==len(self.observation_space)-1:
            self.done = True

        self.cash.append(self.cash[-1]*(1+reward))

        return np.append(self.observation_space[self.time], self.state),reward,self.done,self.state


    def reset(self):
        self.time = 0
        self.done = False
        self.state =0
        self.cash = [10000]

        return np.append(self.observation_space[self.time], self.state)

    def reset_for_test(self,flag="test"):
        self.time = 0
        self.done = False
        self.state =0
        self.cash = [10000]
        if flag=="test":
            self.observation_space = self.test_observation
            self.close = self.test_close
        elif flag=="valid":
            self.observation_space = self.valid_observation
            self.close = self.valid_close
        else:
            self.observation_space = self.train_observation
            self.close = self.train_close
        return np.append(self.observation_space[self.time], self.state)

    def set_env(self,start,end):
        if end > len(self.observation_space)-1:
            return False
        self.observation_space = self.observation_space[start:end]
        self.close = self.close[start:end]
        return True
