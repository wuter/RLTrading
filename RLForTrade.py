import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Market import Market
np.random.seed(5)
tf.set_random_seed(5)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1000
RENDER = False  # rendering wastes time
GAMMA = 0.9    # reward discount in TD error
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic
REG = 0.0001


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        #这里稍微注意：因为AC框架可以使用单步更新，所以s的大小为1*n_features
        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # 1*n_features
        self.a = tf.placeholder(tf.int32, None, "act") #
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1',
                kernel_regularizer= tf.contrib.layers.l1_regularizer(REG)
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2',
                kernel_regularizer= tf.contrib.layers.l1_regularizer(REG)
            )
            l3 = tf.layers.dense(
                inputs=l2,
                units=32,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3',
                kernel_regularizer= tf.contrib.layers.l1_regularizer(REG)
            )

            self.acts_nextprob = tf.layers.dense(
                inputs=l3,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_nextprob',
                kernel_regularizer = tf.contrib.layers.l1_regularizer(REG)
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_nextprob[0, self.a]+1e-3)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_nextprob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1',
                kernel_regularizer=tf.contrib.layers.l1_regularizer(REG)
            )
            l2 = tf.layers.dense(
                inputs=l1,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2',
                kernel_regularizer=tf.contrib.layers.l1_regularizer(REG)
            )
            l3 = tf.layers.dense(
                inputs=l2,
                units=32,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l3',
                kernel_regularizer=tf.contrib.layers.l1_regularizer(REG)
            )

            self.v = tf.layers.dense(
                inputs=l3,
                units=1,  # 这里输出表示当前state下动作的值函数
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V',
                kernel_regularizer=tf.contrib.layers.l1_regularizer(REG)
            )

        with tf.variable_scope('squared_TD_error'):
            # self.v 当前state下的值函数
            # self.v_ 下一个状态的值函数
            # self.r 当前状态下reward
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_next):
        s, s_next = s[np.newaxis, :], s_next[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


def test(env,actor,critic,op_path):
    s = env.reset_for_test()
    t = 0
    track_r = []
    prob = []
    while True:
        a = actor.choose_action(s)

        s_next, r, done, info = env.step(a)
        op_path.append((a, env.state, r, env.close[env.time], env.close[env.time - 1]))

        track_r.append(r)
        # actor 将在s状态下计算得到的r和s_next传入个给critic,  分别计算出S和S_next对应的value(V和V_)
        # 将计算得到的奖励至td_error传递给actor，代替police gradient中的tf_vt
        # td_error = critic.learn(s, r, s_next)  # gradient = grad[r + gamma * V(s_next) - V(s)]
        # actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_next
        t += 1

        if done :
            prob.append((np.array(track_r)+1).prod())

            print('测试轮的收益是%f'%(prob[-1]))
            break

env = Market()

N_F = env.observation_space.shape[1]+1
N_A = len(env.action_space)


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

prob = []
ep_rs_nextsum = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    path = []
    loss = []
    while True:
        a = actor.choose_action(s)

        s_next, r, done, info = env.step(a)
        # print(( a, env.state, r, env.close[env.time],env.close[env.time-1]))

        track_r.append(r)
        # actor 将在s状态下计算得到的r和s_next传入个给critic,  分别计算出S和S_next对应的value(V和V_)
        # 将计算得到的奖励至td_error传递给actor，代替police gradient中的tf_vt
        td_error = critic.learn(s, r, s_next)  # gradient = grad[r + gamma * V(s_next) - V(s)]
        loss.append(td_error)
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_next
        t += 1



        if done :
            ep_rs_nextsum.append(env.cash)

            prob.append((np.array(track_r)+1).prod())
            print('loss', np.mean(td_error))

            print('第%d轮的收益是%f'%(i_episode, prob[-1]))
            break
op_path = []
test(env,actor,critic,op_path)
plt.figure()
plt.plot(range(len(prob)),prob)
plt.show()

for x in op_path:
    print(x)
