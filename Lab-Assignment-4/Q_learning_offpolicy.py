# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:06:49 2022

@author: Sumanth Raikar
"""
import numpy as np
import gym
from matplotlib import pyplot as plt
import matplotlib
import itertools

import random 
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import sys
#if "../" not in sys.path:
#  sys.path.append("../") 
  
matplotlib.style.use('ggplot')  


#env=gym.make('Blackjack-v0')
#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
#env = gym.make('Acrobot-v1')

def discretization(state):
    resolution = (env.observation_space.high-env.observation_space.low)/40
    #calculate what bin a state should go after resolution
    discrete_state  = (state-env.observation_space.low)//resolution
    return tuple(discrete_state.astype(int))
    



def epsilon_greedy_policy(Q, epsilon,nA):
    def policy(observation):
        A = np.ones(nA)*epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action]+=(1-epsilon)
        return A
    return policy

def Q_learning(env,no_of_episodes,discount=1,epsilon=0.1,alpha=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = epsilon_greedy_policy(Q, epsilon,env.action_space.n)
    render_state=0
    rewards=[]
    
    for episode in range(1,no_of_episodes+1):
        epi_reward=0.0
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, no_of_episodes), end="")
            sys.stdout.flush()
            render_state=0
            
        if render_state:
            env.render()
            
        state=discretization(env.reset())
        #state=env.reset()
        
        
        for t in itertools.count():
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)),p=action_prob)
            next_state,reward,done,_ = env.step(action)
            next_state = discretization(next_state)
            #next_action_probs = policy(state)
            #next_action = np.random.choice(np.arange(len(next_action_probs)),p=next_action_probs)
            best_action = np.argmax(Q[next_state])
            Q[state][action]+=alpha*(reward+discount*Q[next_state][best_action]-Q[state][action])
            epi_reward+=reward
            if done:
                rewards.append(epi_reward)
                break
            state=next_state
    env.close()    
    return Q,rewards

def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           #cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
    surf=ax.plot_wireframe(X,Y,Z)                       
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    #fig.colorbar(surf)
    plt.show()

#plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
#plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

if __name__=="__main__":
    Q,rews = Q_learning(env,5000)
    moving_avg=np.convolve(rews, np.ones((200,))/200, mode="Valid")
    plt.plot([i for i in range(len(moving_avg))],moving_avg)            
    plt.ylabel(f"Reward every 200 eps") 
    plt.xlabel(f"episode #")
    plt.show()

    state = env.reset()
    state =discretization(state)
    #env.render()
    while True:
        env.render()
        act = np.argmax(Q[state])
        next_state,_,done,_ = env.step(act)
        next_state = discretization(next_state)
        if done:
            env.reset()
            #env.close()
        state = next_state
    