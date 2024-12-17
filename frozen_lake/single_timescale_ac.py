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
        x = torch.tanh(x)
        x = self.l2(x)
        return x
    
class policyparameter(nn.Module):
    def __init__(self,n1,n2,n3):
        super(policyparameter,self).__init__()
        self.l1 = nn.Linear(n1,n2)
        self.l2 = nn.Linear(n2,n3)
        
    def forward(self,x):
         x =self.l1(x)
         x = torch.tanh(x)
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
        x = torch.tanh(x)
        x = self.l2(x)
        return x
    
 
env = gym.make("FrozenLake-v1", is_slippery=True)


wrapped_env = FlattenObservation(env)

nS = wrapped_env.observation_space.shape[0]
size = math.sqrt(nS)

d1 = 2
d2 = 20
d3 =  1
nA = 4
N = 10000


    
def feat(state):
    res=torch.zeros(d1)
    res[0]=state/size 
    res[1] = state - res[0]*size
    return res
        

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
    voptim = torch.optim.SGD(value.parameters(),lr = 1.5)
    poptim = torch.optim.SGD(policy.parameters(),lr = 1.5)
    lambda1 = lambda epoch : (1 + epoch)**(-0.6)
    lambda2 = lambda epoch : (1 + epoch)**(-0.6)  
    vscheduler = LambdaLR(voptim,lambda1)
    pscheduler = LambdaLR(poptim,lambda2)
    state , info= env.reset()
    J = 0
    const = 0
    

    n = 1

    L = 0
    while n <= N:

        #Actor critic learning------------
        probs=policy(feat(state))
        action = Categorical(probs).sample()
        next_state,reward,terminated,truncated ,info= env.step(int(action))
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
        ploss=-Categorical(probs).log_prob(action)*delta.detach()
        #print(vloss)
        #print(ploss.shape)
        voptim.zero_grad()
        vloss.backward()
        voptim.step()
        poptim.zero_grad()
        ploss.backward()
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
    start_t = time.time()
    policy = train_ac()
    end_t = time.time()
    state ,info = env.reset()
    indices = []
    reward_list =[]
    returns = deque(maxlen = N)     
    m = 1
    while(m <= N):
        probs=policy(feat(state))
        action = Categorical(probs).sample()
        next_state,reward,terminated,truncated ,info= env.step(int(action))
        if(terminated or truncated):
            next_state,info = env.reset()

        returns.append(reward)
        indices.append(m)
        reward_list.append(np.mean(returns))
        
        m += 1
        state = next_state 
    return reward_list,indices,end_t-start_t
        



  
       

f = open("plotting_single_ac.txt", "a")
f1 = open("plotting_single_ac_sdt.txt", "a")
#f2 = open("plotting_ac_min.txt", "a")
n_seed = 10
seed = randint(1000,size = (n_seed,1))
policy= train_ac()
for i in range(0,n_seed):
      seed[i] = randint(1000)
reward_list_ac = np.zeros((n_seed,N))
reward_list_ca = np.zeros((n_seed,N))
reward_list_q = np.zeros((n_seed,N)) 
reward_list_ppo_ac = np.zeros((n_seed,N))  
reward_list_ppo_ca = np.zeros((n_seed,N))  
compute_time = np.zeros((n_seed , 1))
    
for i in range(0,n_seed):
      #reward_list_q[i] , indices = DQN(seed[i][0])
      reward_list_ac[i], indices , compute_time[i] = actor_critic(seed[i][0])
      #reward_list_ca[i],indices = critic_actor(seed[i][0])
      #indices , reward_list_ppo_ac[i] = PPO_actor_critic(seed[i][0])
      #indices , reward_list_ppo_ca[i] = PPO_critic_actor(seed[i][0])
      #print(i)
reward_ac = np.mean(reward_list_ac,axis = 0)
compute_t = np.mean(compute_time,axis = 0)
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
print('compute_time_ac =',compute_t[0])
    
print('sdt_reward_ac =',stdr1[N-1]) 
'''print('sdt_reward_ca = ',stdr2[N-1])
print('sdt_reward_q =',stdr3[N-1]) 
print('sdt_reward_ppo_ac= ',stdr4[N-1])
print('sdt_reward_ppo_ca =',stdr5[N-1]) '''

    



       
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