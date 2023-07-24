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
    def qlearning(self, num_iter, alpha, epsilon):
        iter_num = []
        self.qvalue = np.zeros((len(self.yuanyang.states), len(self.yuanyang.actions)))
        for iter in range(num_iter):
            s = 0
            flag = self.greedy_test()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print("qlearning the number need to finish first task:", iter_num[0])
            if flag == 2:
                print("qlearning the number need to finish first shortest task:", iter)
                break
            s_sample = []
            a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
            t = False
            count = 0
            while False == t and count < 30:
                s_next, r, t = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    a1 = self.greedy_policy(self.qvalue, s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * self.qvalue[s_next, a1_num]
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                s = s_next
                a = self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                count += 1
        return self.qvalue

if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    brain = TD_RL(yuanyang)
    qvalue1 = brain.sarsa(num_iter = 5000, alpha = 0.1, epsilon = 0.8)
    # qvalue2 = brain.qlearning(num_iter = 5000, alpha = 0.1, epsilon = 0.1)

    # print(qvalue1)
    yuanyang.action_value = qvalue1

    # print(qvalue2)
    # yuanyang.action_value = qvalue2

    flag = 1
    s = 0
    step_num = 0
    path = []

    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue1, s)
        print('%d->%s\t' % (s, a), qvalue1[s, 0], qvalue1[s, 1], qvalue1[s, 2], qvalue1[s, 3])
        # a = brain.greedy_policy(qvalue2, s)
        # print('%d->%s\t' % (s, a), qvalue2[s, 0], qvalue2[s, 1], qvalue2[s, 2], qvalue2[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_
    
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()           
