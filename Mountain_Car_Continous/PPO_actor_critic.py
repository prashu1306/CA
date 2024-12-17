# %matplotlib inline
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical
import numpy as np
from numpy.random import randint
from numpy.random import choice
from matplotlib import pyplot as plt
import gym
import random as rnd
from gym.wrappers import FlattenObservation
import math
from collections import deque
import statistics
import time
class valuefunction(nn.Module):
    def __init__(self,n1,n2,n3):
        super(valuefunction,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
        x =self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x
    
class policyparameter(nn.Module):
    def __init__(self,n1,n2,n3):
        super(policyparameter,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
         x =self.l1(x)
         x = torch.relu(x)
         x = self.l2(x)
         x = torch.softmax(x, dim=0)
         return x
    
class QNN(nn.Module):
    def __init__(self,n1,n2,n3):
        super(QNN,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        self.lossfn = nn.MSELoss()

    def forward(self,x):
        x =self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x
    
 
env = gym.make("MountainCarContinuous-v0")

d1 = 2
d2 = 64
d3 =  1
nA = 4
N = 5000
n_train = 10000
batch_length = 50


    
def feat(state):
    res = torch.zeros(d1)
    for i in range(d1):
        res[i]=torch.as_tensor(state[i])
    return res
        

def getaction(action):
    a = 0
    b = 0
    if(action == 0):
        a = rnd.uniform(-1,-0.5)
        #b = rnd.uniform(-1,-0.5)
    if(action ==1):
        a = rnd.uniform(-0.5,0)
        #b = rnd.uniform(-0.5,0)
    if(action ==2 ):
        a = rnd.uniform(0,0.5)
        #b = rnd.uniform(0,0.5)
    if(action == 3):
        a = rnd.uniform(0.5,1)
        #b = rnd.uniform(0.5,1)
    
    return torch.tensor([a])

def find_old_log_probs(policy , value):
  states = []
  actions = []
  old_logprobs =[]
  advantage_function = torch.zeros([batch_length], dtype=torch.float64)
  state,info = env.reset()
  L = 0
  count = 1
  while(count <=batch_length):
    probs=policy(feat(state))
    if(torch.isnan(probs).any()):
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True)
    action_ = Categorical(probs).sample()
    action = getaction(action_)
    next_state,reward,terminated,truncated ,info= env.step(action)
    a = 1.5/(count**0.4)
    L += a*(reward-L)
    advantage_function[count-1] = reward - L + value(feat(next_state)).detach()-value(feat(state))
    #print(next_state)  
    if(terminated or truncated):
        next_state,info = env.reset()
    states.append(state)
    actions.append(action_)
    old_logprobs.append(probs[action_])
    state = next_state
    count +=1
    return states,actions,torch.log(torch.tensor(old_logprobs)),advantage_function
  


def train():
    #np.random.seed(seed)
    value = valuefunction(d1,d2,d3)
    policy = policyparameter(d1,d2,nA)
    voptim = torch.optim.SGD(value.parameters(),lr = 1.5)
    poptim = torch.optim.SGD(policy.parameters(),lr = 1.5)
    lambda1 = lambda epoch : (1 + epoch)**(-0.4)
    lambda2 = lambda epoch : (1 + epoch)**(-0.6)  
    vscheduler = LambdaLR(voptim,lambda1)
    pscheduler = LambdaLR(poptim,lambda2)
    state , info= env.reset()
    J = 0
    

    n = 1

    L = 0
    cc = 1
    state , info= env.reset()
    while n <= n_train:
        batch_states,batch_actions ,old_log_probs,advantage = find_old_log_probs(policy,value)
        m = 1
        i = 0
        probs=policy(feat(state))
        if(torch.isnan(probs).any()):
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True)
        action_ = Categorical(probs).sample()
        action = getaction(action_)
        next_state,reward,terminated,truncated ,info= env.step(action)
        if(terminated or truncated):
                next_state,info = env.reset()
        while(m < 5):
            new_probs = []
            for k in range(0,len(batch_states)):
                list_prob = policy(feat(batch_states[k])).clone().tolist()
                new_probs.append(list_prob[batch_actions[k]])
            new_logprobs = torch.log(torch.tensor(new_probs).clone())
            a = 1.5/(m**0.4)
            
            #delta= reward - L + value(feat(next_state)).detach()-value(feat(state))
            L += a*(reward-L)
            #vloss=0.5*delta**2
            #print(old_log_probs)
            ratios = torch.exp(new_logprobs.clone() - old_log_probs.clone().detach())
            #print(ratios)
            #advantage = (reward - L + value(feat(next_state)).detach().clone()-value(feat(state))).item()
            surr1 = ratios.clone()*advantage
            surr2 = torch.clamp(ratios.clone(), 0.8, 1.2)* advantage
            surr1 = torch.tensor(surr1, requires_grad=True)
            surr2 = torch.tensor(surr2, requires_grad=True)

            ploss = -torch.min(surr1.clone(), surr2.clone()).mean()
            #ploss = -torch.tensor([1])
            #voptim.zero_grad()
            #vloss.backward()
            #voptim.step()
            poptim.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            ploss.backward(retain_graph=True)
            poptim.step()
            #vscheduler.step()
            
            m += 1
        

            
        pscheduler.step()    
        delta= reward - L + value(feat(next_state)).detach()-value(feat(state))
        vloss=0.5*delta**2
        voptim.zero_grad()
        vloss.backward()
        voptim.step()
        vscheduler.step()
        
        state = next_state
            #indices.append(cc)
        n += 1
    
    
        
    #print(reward_list_final)
    return policy

def PPO_actor_critic(seed):
    np.random.seed(seed)
    start_t = time.time()
    policy = train()
    end_t = time.time()
    print("completed training")
    indices = []
    reward_list =[]
    reward_list_final =[]
    returns = deque(maxlen = 5000000)
    n = 1
    state , info= env.reset()
    while n <=N:
        
        probs=policy(feat(state))
        if(torch.isnan(probs).any()):
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True)
        action_ = Categorical(probs).sample()
        action = getaction(action_)
        next_state,reward,terminated,truncated ,info= env.step(action)
        if(terminated or truncated):
                next_state,info = env.reset()

        returns.append(reward)
        reward_list.append(np.mean(returns))
        indices.append(n)
        state = next_state  
        n +=1
    return indices , reward_list , end_t - start_t
        


f = open("plotting_ppoac.txt", "a")
f1 = open("plotting_ppoac_sdt.txt", "a")
    


n_seed = 10
seed = randint(1000,size = (n_seed,1))
for i in range(0,n_seed):
      seed[i] = randint(1000)
reward_list_ppo_ac = np.zeros((n_seed,N))  
compute_time = np.zeros((n_seed,1))

    
for i in range(0,n_seed):
      indices , reward_list_ppo_ac[i] , compute_time[i] = PPO_actor_critic(seed[i][0])
      
reward_ppo_ac = np.mean(reward_list_ppo_ac , axis = 0)
compute_t = np.mean(compute_time)
#reward_ppo_ca = np.mean(reward_list_ppo_ac, axis = 0)
for i in reward_ppo_ac:
    f.write(str(i)  + '\n')
f.close()    




#stdr1 = np.std(reward_list_ac,axis = 0)
#stdr2 = np.std(reward_list_ca,axis = 0)
#stdr3 = np.std(reward_list_q,axis = 0)
stdr4 = np.std(reward_list_ppo_ac ,axis = 0)
#stdr5 = np.std(reward_list_ppo_ca , axis = 0)

for i in stdr4:
    f1.write(str(i)  + '\n')
f1.close()  

#print('avg_reward_ac=',reward_ac[N-1]) 
'''print('avg_reward_ca=',reward_ca[N-1])
print('avg_reward_q=',reward_q[N-1])'''
print('avg_reward_ppo_ac=',reward_ppo_ac[N-1])
'''print('avg_reward_ppo_ca=',reward_ppo_ca[N-1])'''
print('compute_time_ppo_ac=',compute_t)
    
#print('sdt_reward_ac =',stdr1[N-1]) 
'''print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) '''
print('sdt_reward_ppo_ac= ',stdr4[N-1])
'''print('sdt_reward_ppo_ca =',stdr5[N-1]) '''

    


         
