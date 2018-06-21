import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from time import time
from sklearn.preprocessing import StandardScaler
from random import sample
from indicator import getNum
from Market import Market

parser = argparse.ArgumentParser()
parser.add_argument('-log_dir',dest='log_dir',type=str )
parser.add_argument('-record_dir',dest='record_dir',type=str)
parser.add_argument('-filename',dest='filename',type=str)
parser.add_argument('-category',dest='category',default='stock',type=str)
parser.add_argument('-iters',dest='iters',default=500, type=int)
parser.add_argument('-entropy_beta',dest='entropy_beta',default=0.0,type=float)
parser.add_argument('-reuse',dest='reuse',default=False, type=bool)
parser.add_argument('-dropout',dest='dropout',default=1.0, type=float)
result = parser.parse_args()

if result.category=="future":
    CATEGORY = True
else:
    CATEGORY = False

OUTPUT_GRAPH = False
LOG_DIR = result.log_dir
RECORD_DIR = result.record_dir
N_WORKERS = int(multiprocessing.cpu_count()/3)
MAX_GLOBAL_EP = 1
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = result.iters
GAMMA = 0.97
ENTROPY_BETA = result.entropy_beta
LR_A = 0.00001    # learning rate for actor
LR_C = 0.00001    # learning rate for critic
GLOBAL_RUNNING_R = {}
GLOBAL_EP = 0
keep_prob=result.dropout
REUSE = result.reuse

tf.set_random_seed(2)
np.random.seed(2)

filename = result.filename
size=0.4
train_size=0.6
test_size=0.3
valid_size=0.1

env = Market(filename,size,train_size,test_size,valid_size,CATEGORY)
N_S = env.observation_space.shape[1]+1
N_A = len(env.action_space)

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob+1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.random_uniform_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 64, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='la')
            dp_a = tf.nn.dropout(l_a, keep_prob)
            l_a = tf.layers.dense(dp_a, 64, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='la1')
            dp_a = tf.nn.dropout(l_a, keep_prob)
            l_a = tf.layers.dense(dp_a, 64, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='la2')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 32, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='lc')
            dp_c = tf.nn.dropout(l_c, keep_prob)
            l_c = tf.layers.dense(dp_c, 32, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='lc1')
            dp_c = tf.nn.dropout(l_c, keep_prob)
            l_c = tf.layers.dense(dp_c, 32, tf.nn.tanh, bias_initializer=b_init, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, test=False):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        #print(prob_weights)
        if not test:
            action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())  # select action w.r.t the actions prob
        else:
            action = np.argmax(prob_weights.ravel())
        return action

class Worker(object):
    def __init__(self, name, globalAC, step=60):
        self.env = Market(filename,size,train_size,test_size,valid_size,CATEGORY)
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.step = step

    def set_scope(self,scope):
        return self.env.set_env(scope[0],scope[1])


    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            R = [1]
            while True:
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                #print("a:",a,"r:",r,"time:",self.env.time,"len:",len(self.env.observation_space))
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                R.append((r+1)*R[-1])

                if total_step % self.step == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1   
                if done:
                    GLOBAL_RUNNING_R[self.name].append(R[-1])
                    GLOBAL_EP += 1
                    print(self.name,"Ep:", GLOBAL_EP, "prof:",R[-1],"len",len(R))  
                    #for temp in R:
                        #print(temp+1)                                                 

                    break


class TestWorker(object):
    def __init__(self, name, globalAC):
        self.env = Market(filename,size,train_size,test_size,valid_size,CATEGORY)
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.AC.pull_global()

    def reset(self):
        self.AC.pull_global()

    def work(self, flag="test"):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        s = self.env.reset_for_test(flag)
        while True:
            a = self.AC.choose_action(s,test=True)
            s_, r, done, info = self.env.step(a)
            #print("a:",a,"r:",r,"time:",self.env.time,"len:",len(self.env.observation_space))
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

            s = s_
            total_step += 1
            if done:
                prob = np.array(buffer_r) + 1
                print("prof:",prob.prod(),"len",len(prob))
                break
        return prob.prod()



if __name__ == "__main__":
    SESS = tf.Session()
    start_time = time()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.AdamOptimizer(LR_A, name='AdamA')
        OPT_C = tf.train.AdamOptimizer(LR_C, name='AdamC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        i = 0
        while i<10:
            i_name = 'W_%i'%i
            GLOBAL_RUNNING_R[i_name] = []
            worker = Worker(i_name, GLOBAL_AC)
            #if not worker.set_scope([i,i+30]):
            #    break
            workers.append(worker)
            i = i+1
            #if i>20:
            #     break

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if REUSE:
        saver.restore(SESS,RECORD_DIR+"/model0.ckpt")

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        writer = tf.summary.FileWriter(LOG_DIR, SESS.graph)

    test = TestWorker('test',GLOBAL_AC)
    pr = []
    TR= []
    for stp in range(UPDATE_GLOBAL_ITER):
        GLOBAL_EP = 0
        worker_threads = []
        for worker in workers:#sample(workers, N_WORKERS):
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        for worker in workers:
            worker.AC.pull_global()

        test.AC.pull_global()
        tp = test.work(flag="train")
        TR.append(tp)
        print("train iter: ",stp,"prof",tp)

        if stp%5==0:
            prof = test.work(flag="test")
            pr.append(prof)
            print("test iter:",stp,"test prof:",prof)

    end_time = time()
    print("time:",end_time-start_time)
    saver.save(SESS,RECORD_DIR+"/model0.ckpt") 
    with open(LOG_DIR+"/train_r.txt",'w') as f:
        f.write(str(TR))
    with open(LOG_DIR+"/test_r.txt","w") as f:
        f.write(str(pr)) 
