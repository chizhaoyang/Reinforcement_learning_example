import random
import time
import sys
sys.path.append("../ch2")
from YuanYangEnv import YuanYangEnv

class DP_Policy_Iter:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states) + 1)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

        # initial strategy
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]
    
    def policy_evaluate(self):
        for i in range(100):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = yuanyang.collide(yuanyang.state_to_position(state))
                flag2 = yuanyang.find(yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1:
                    continue
                action = self.pi[state]
                s, r, t = yuanyang.transform(state, action)

                new_v = r + self.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < 1e-6:
                print("Num of evaluation iterator", i)
                break
    
    def policy_improve(self):
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1:
                continue
            a1 = self.actions[0]
            s, r, t = yuanyang.transform(state, a1)
            v1 = r + self.gamma * self.v[s]

            for action in self.actions:
                s, r, t = yuanyang.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]
                
                self.pi[state] = a1

    def policy_iterate(self):
        for i in range(100):
            self.policy_evaluate()
            pi_old = self.pi.copy()
            self.policy_improve()
            if (self.pi == pi_old):
                print("Num of policy improve:", i)
                break

if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = DP_Policy_Iter(yuanyang)
    policy_value.policy_iterate()
    flag = 1
    s = 0
    path = []
    for state in range(100):
        i = int(state / 10)
        j = state % 10
        yuanyang.value[j, i] = policy_value.v[state]
    step_num = 0
    while flag:
        path.append(s)
        yuanyang.path = path
        a = policy_value.pi[s]
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    
    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()