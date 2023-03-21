import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from YuanYangEnv import *
from YuanYangEnv import YuanYangEnv

class TD_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang

        # value initialization
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))

    def greedy_policy(self, qfun, state):
        amax=qfun[state, :].argmax()
        return self.yuanyang.actions[amax]
    
    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        if np.random.uniform() < 1 - epsilon:
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]
        
    def find_anum(self, a):
        for i in range(len(self.yuanyang.actions)):
            if a == self.yuanyang.actions[i]:
                return i
            
    def greedy_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1

        if s == 9:
            flag = 1
        if s == 9 and step_num < 21:
            flag = 2
        return flag
    
    def sarsa(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        for iter in range(num_iter):
            epsilon = epsilon * 0.99
            s_sample = []

            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("SARSA the number need to finish first task:", iter_num[0])
            if flag == 2:
                print("SARSA the number need to finish first shortest task:", iter)
                break

            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0

            # s0-s1-s2-s1-s2-s_ternimate
            while False == t and count < 30:
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    a1 = self.epsilon_greedy_policy(self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(a1)
                    # Q-Learning
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                s = s_next
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
            return self.qvalue
