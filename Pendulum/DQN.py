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
    
 
env = gym.make("Pendulum-v1")

d1 = 3
d2 = 64
d3 =  1
nA = 4
N = 5000
n_train = 10000


    
def feat(state):
    res = torch.zeros(d1)
    for i in range(d1):
        res[i]=torch.as_tensor(state[i])
    return res
        

def getaction(action):
    a = 0
    b = 0
    if(action == 0):
        a = rnd.uniform(-2,-1)
        b = rnd.uniform(-2,-1)
    if(action ==1):
        a = rnd.uniform(-1,0)
        b = rnd.uniform(-1,0)
    if(action ==2 ):
        a = rnd.uniform(0,1)
        b = rnd.uniform(0,1)
    if(action == 3):
        a = rnd.uniform(1,2)
        b = rnd.uniform(1,2)
    
    return torch.tensor([a,b])

def find_old_log_probs(policy):
  states = []
  actions = []
  old_logprobs =[]
  state,info = env.reset()
  count = 1
  while(count <50):
    probs=policy(feat(state))
    action = Categorical(probs).sample()
    next_state,reward,terminated,truncated ,info= env.step(int(action))
    #print(next_state)  
    if(terminated or truncated):
        next_state,info = env.reset()
    states.append(state)
    actions.append(action)
    old_logprobs.append(probs[action])
    state = next_state
    count +=1
    return states,actions,torch.log(torch.tensor(old_logprobs))
  


def train_dqn():
   Qvalue = QNN(d1,d2,nA)
   qoptim = torch.optim.AdamW(Qvalue.parameters(),lr = 0.5,amsgrad=True)
   state , info= env.reset()
   J = 0
   const = 0
  

   n = 1

   L = 0
   epsilon = 0.4
   Qvaluee = Qvalue(feat(state)).detach().numpy()
   while n <= N:
       if np.random.uniform(0,1) < epsilon:
           action = np.random.choice(range(nA))
       else:
           action = np.argmax(Qvaluee)
       action_ = getaction(action)
       next_state,reward,terminated,truncated ,info= env.step(action_)
       if(terminated or truncated):
            #print('obstacle reached' , n)
            path = []
            next_state,info = env.reset()
            path.append(next_state)

       J = (J + reward)/n #average reward
       L += 0.6*(reward-L)
       Qvalue_next = Qvalue(feat(next_state))
       qv = torch.max(Qvalue_next)
       Qvaluee_next = torch.tensor([qv])
       reward_tensor = torch.tensor([reward])
       L_tensor = torch.tensor([L])
       criterion = nn.SmoothL1Loss()
       state_action_values = Qvalue(feat(state))[action]
       expected_state_action_values = reward_tensor - L_tensor + Qvaluee_next
       qloss = criterion(state_action_values,expected_state_action_values)
       #print(qloss.mean().shape)
       qoptim.zero_grad()
       qloss.backward()
       qoptim.step()
       n += 1
       state = next_state
   return Qvaluee  

   

def DQN(seed):
   np.random.seed(seed)
   start_time = time.time()
   Qvaluee = train_dqn()
   end_time = time.time()
   indices = []
   reward_list =[]
   returns = deque(maxlen = N)
   epsilon = 0.4
   m = 1
   while m <= N:
       if np.random.uniform(0,1) < epsilon:
           action = np.random.choice(range(nA))
       else:
           action = np.argmax(Qvaluee)
       action_ = getaction(action)
       next_state,reward,terminated,truncated ,info= env.step(action_)
       if(terminated or truncated):
            #print('obstacle reached' , n)
            next_state,info = env.reset()
       returns.append(reward)
       reward_list.append(np.mean(returns))
       #print(J)
       indices.append(m)
       
       m += 1
       state = next_state
   return reward_list,indices, end_time-start_time


  
f = open("plotting_dqn.txt", "a")
f1 = open("plotting_dqn_sdt.txt", "a") 

n_seed = 10
seed = randint(1000,size = (n_seed,1))
for i in range(0,n_seed):
      seed[i] = randint(1000)
reward_list_ac = np.zeros((n_seed,N))
reward_list_ca = np.zeros((n_seed,N))
reward_list_q = np.zeros((n_seed,N)) 
reward_list_ppo_ac = np.zeros((n_seed,N))  
reward_list_ppo_ca = np.zeros((n_seed,N))  
compute_time = np.zeros((n_seed,1))    

for i in range(0,n_seed):
      reward_list_q[i] , indices ,compute_time[i]= DQN(seed[i][0])
      #reward_list_ac[i], indices = actor_critic(seed[i][0])
      #reward_list_ca[i],indices = critic_actor(seed[i][0])
      #indices , reward_list_ppo_ac[i] = PPO_actor_critic(seed[i][0])
      #indices , reward_list_ppo_ca[i] = PPO_critic_actor(seed[i][0])
      #print(i)
#reward_ac = np.mean(reward_list_ac,axis = 0)
#reward_ca = np.mean(reward_list_ca,axis = 0)
reward_q = np.mean(reward_list_q,axis = 0)
compute_t = np.mean(compute_time,axis = 0)
for i in reward_q:
    f.write(str(i) + '\n')
f.close()


#reward_ppo_ac = np.mean(reward_list_ppo_ac , axis = 0)
#reward_ppo_ca = np.mean(reward_list_ppo_ac, axis = 0)

#stdr1 = np.std(reward_list_ac,axis = 0)
#stdr2 = np.std(reward_list_ca,axis = 0)
stdr3 = np.std(reward_list_q,axis = 0)
#stdr4 = np.std(reward_list_ppo_ac ,axis = 0)
#stdr5 = np.std(reward_list_ppo_ca , axis = 0)

for i in stdr3:
    f1.write(str(i) + '\n')
f1.close()


#print('avg_reward_ac=',reward_ac[N-1]) 
#print('avg_reward_ca=',reward_ca[N-1])
print('avg_reward_q=',reward_q[N-1])
'''print('avg_reward_ppo_ac=',reward_ppo_ac[N-1])
print('avg_reward_ppo_ca=',reward_ppo_ca[N-1])'''
    
print('Compute time = ', compute_t[0])    
#print('sdt_reward_ac =',stdr1[N-1]) 
#print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) 
'''print('sdt_reward_ppo_ac= ',stdr4[N-1])
print('sdt_reward_ppo_ca =',stdr5[N-1]) '''




