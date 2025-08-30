# %matplotlib inline
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical
import numpy as np
from numpy.random import randint
from numpy.random import choice
from matplotlib import pyplot as plt
import gymnasium as gym
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
        # Initialize weights with smaller values
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        
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
        # Initialize weights with smaller values
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        
    def forward(self,x):
         x = self.l1(x)
         x = torch.relu(x)
         x = self.l2(x)
         # Clamp logits to prevent extreme values
         x = torch.clamp(x, min=-10, max=10)
         x = torch.softmax(x, dim=0)
         # Add small epsilon to prevent exact zeros
         x = x + 1e-8
         # Renormalize to ensure it sums to 1
         x = x / torch.sum(x)
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
  




def train_ac():
    #np.random.seed(seed)
    value = valuefunction(d1,d2,d3)
    policy = policyparameter(d1,d2,nA)
    # Reduced learning rates for stability
    voptim = torch.optim.SGD(value.parameters(),lr = 0.01)
    poptim = torch.optim.SGD(policy.parameters(),lr = 0.01)
    lambda1 = lambda epoch : (1 + epoch)**(-0.4)
    lambda2 = lambda epoch : (1 + epoch)**(-0.6)  
    vscheduler = LambdaLR(voptim,lambda1)
    pscheduler = LambdaLR(poptim,lambda2)
    state , info= env.reset()
    J = 0
    const = 0
    

    n = 1

    L = 0
    while n <= n_train:

        #Actor critic learning------------
        probs=policy(feat(state))
        #print(probs)
        # Check for NaN/Inf and handle it properly
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Warning: NaN/Inf detected in probs at iteration {n}")
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True)
        
        # Ensure probs are valid probabilities
        if torch.sum(probs) == 0:
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=True)
            
        action_ = Categorical(probs).sample()
        action = getaction(action_)
        next_state,reward,terminated,truncated ,info= env.step(action)
        if(terminated or truncated):
            next_state,info = env.reset()

        J = (J + reward)/n #average reward
        #returns.append(reward)
        #reward_list.append(np.mean(returns))
        #print(J)
        a = 1.5/(n**0.4)


        delta=reward - L + value(feat(next_state)).detach()-value(feat(state))
        #delta=reward  + 0.9*value(feat(next_state)).detach()-value(feat(state))
        L += a*(reward-L)
        vloss=0.5*delta**2
        ploss=-Categorical(probs).log_prob(action_)*delta.detach()
        #print(vloss)
        #print(ploss.shape)
        
        # Value function update with gradient clipping
        voptim.zero_grad()
        vloss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=1.0)
        voptim.step()
        
        # Policy update with gradient clipping
        poptim.zero_grad()
        ploss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        poptim.step()
        vscheduler.step()
        pscheduler.step()


        n += 1
        #print(n)
        #print('state= ', state, 'action = ',int(action),'next state = ',next_state)
        state = next_state
    return policy

def actor_critic(seed):  
    np.random.seed(seed) 
    start_time = time.time()
    policy = train_ac()
    end_time = time.time()
    state ,info = env.reset()
    indices = []
    reward_list =[]
    returns = deque(maxlen = N)     
    m = 1
    while(m <= N):
        probs=policy(feat(state))
        # Check for NaN/Inf and handle it properly
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Warning: NaN/Inf detected in probs at evaluation step {m}")
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=False)
        
        # Ensure probs are valid probabilities
        if torch.sum(probs) == 0:
            probs = torch.tensor([0.25,0.25,0.25,0.25], requires_grad=False)
            
        action_ = Categorical(probs).sample()
        action = getaction(action_)
        next_state,reward,terminated,truncated ,info= env.step(action)
        if(terminated or truncated):
            next_state,info = env.reset()

        returns.append(reward)
        indices.append(m)
        reward_list.append(np.mean(returns))
        print('m=', m)
        m += 1
        state = next_state 
    return reward_list,indices,end_time-start_time
        



  
       

f = open("plotting_ac.txt", "a")
f1 = open("plotting_ac_sdt.txt", "a")
#f2 = open("plotting_ac_min.txt", "a")
n_seed = 10
seed = randint(1000,size = (n_seed,1))
policy= train_ac()
for i in range(0,n_seed):
      seed[i] = randint(1000)
reward_list_ac = np.zeros((n_seed,N))
compute_time = np.zeros((n_seed,1))
reward_list_ca = np.zeros((n_seed,N))
reward_list_q = np.zeros((n_seed,N)) 
reward_list_ppo_ac = np.zeros((n_seed,N))  
reward_list_ppo_ca = np.zeros((n_seed,N))  
    
for i in range(0,n_seed):
      #reward_list_q[i] , indices = DQN(seed[i][0])
      reward_list_ac[i], indices,compute_time[i] = actor_critic(seed[i][0])
      #reward_list_ca[i],indices = critic_actor(seed[i][0])
      #indices , reward_list_ppo_ac[i] = PPO_actor_critic(seed[i][0])
      #indices , reward_list_ppo_ca[i] = PPO_critic_actor(seed[i][0])
      print('i=', i)
reward_ac = np.mean(reward_list_ac,axis = 0)
compute_t = np.mean(compute_time , axis =0)
for i in reward_ac:
    f.write(str(i) + '\n')
f.close() 


#reward_ca = np.mean(reward_list_ca,axis = 0)
#reward_q = np.mean(reward_list_q,axis = 0)
#reward_ppo_ac = np.mean(reward_list_ppo_ac , axis = 0)
#reward_ppo_ca = np.mean(reward_list_ppo_ac, axis = 0)

stdr1 = np.std(reward_list_ac,axis = 0)
#stdr2 = np.std(reward_list_ca,axis = 0)
#stdr3 = np.std(reward_list_q,axis = 0)
#stdr4 = np.std(reward_list_ppo_ac ,axis = 0)
#stdr5 = np.std(reward_list_ppo_ca , axis = 0)


for i in stdr1:
    f1.write(str(i) + '\n')
f1.close()



print('avg_reward_ac=',reward_ac[N-1]) 
'''print('avg_reward_ca=',reward_ca[N-1])
print('avg_reward_q=',reward_q[N-1])
print('avg_reward_ppo_ac=',reward_ppo_ac[N-1])
print('avg_reward_ppo_ca=',reward_ppo_ca[N-1])'''
    
print('sdt_reward_ac =',stdr1[N-1]) 
'''print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) 
print('sdt_reward_ppo_ac= ',stdr4[N-1])
print('sdt_reward_ppo_ca =',stdr5[N-1]) '''

print('Compute time = ', compute_t[0]) 



       
'''plt.plot(indices, reward_ac, color='r', label=' actor critic')
#plt.fill_between(indices, min_all, max_all, color='skyblue', alpha=0.4)'''
'''plt.plot(indices, reward_ca, color='b', label='  critic actor')
plt.plot(indices,reward_q,color = 'orange',label = 'DQN')
plt.plot(indices,reward_ppo_ac,color = 'green',label = 'PPO actor critic')
plt.plot(indices,reward_ppo_ca,color = 'pink',label = 'PPO critic actor')'''

'''plt.plot(indices,stdr1,color = 'r',label = 'actor critic')
#plt.plot(indices,stdr2,color = 'b',label = 'critic actor')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')

#plt.ylabel('Standard Deviation for average reward')
plt.legend()
plt.savefig('plot_frozen_lake_ac.png')
plt.show()'''