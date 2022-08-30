# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:14:32 2022

@author: Sumanth Raikar
"""
import numpy as np
import gym
from matplotlib import pyplot as plt
import matplotlib
import keyboard
import random 
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import sys
#if "../" not in sys.path:
#  sys.path.append("../") 
  
matplotlib.style.use('ggplot')  


#env=gym.make('Blackjack-v0')
env = gym.make('MountainCar-v0')
#env = gym.make('CartPole-v0')

def discretization(state):
    resolution = (env.observation_space.high-env.observation_space.low)/5
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


def MC_control__online_greedy(env,no_of_episodes,discount,epsilon):
    final_G = defaultdict(float)
    final_N = defaultdict(float)
    #Q initialized to zeros
    Q = defaultdict(lambda:np.zeros(env.action_space.n))
    
    for episode in range(1,no_of_episodes+1):
        render_state=0
        
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, no_of_episodes), end="")
            sys.stdout.flush()
            render_state=1
        
        episodes=[]
        state=discretization(env.reset())
        policy=epsilon_greedy_policy(Q,epsilon,env.action_space.n)
        for t in range(200):
            action_prob=policy(state)
            action = np.random.choice(np.arange(len(action_prob)),p=action_prob)
            next_state,reward,done,_ = env.step(action)
            next_state = discretization(next_state)
            if next_state[0]>=-0.1:
                reward+=2.0
            episodes.append((state,action,reward))
            if done:
                break
            state=next_state
         
        state_action_pair=set([(tuple(x[0]), x[1]) for x in episodes]) #change to tuple for blackjack
        for s,a in state_action_pair:
            s_a_pair=(s,a)
            first_index = next(i for i,x in enumerate(episodes) if x[0]==s and x[1]==a)
            G = sum([x[2]*(discount**i) for i,x in enumerate(episodes[first_index:])])
            final_G[s_a_pair]+=G
            final_N[s_a_pair]+=1.0
            Q[s][a]=final_G[s_a_pair]/final_N[s_a_pair]
            
  
    return Q,policy


def plot_value_function(V,title="Value Function"):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

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

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))

if __name__=="__main__":
    Q,policy = MC_control__online_greedy(env, 50000, 1, 0.1)
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    #plot_value_function(V, title="Optimal Value Function")
    
    state = env.reset()
    state = discretization(state)
    epi_rews=0.0
    while True: 
        env.render()
        
        act = np.argmax(Q[state])
        next_state,reward,done,_ = env.step(act)
        next_state = discretization(next_state)
        epi_rews+=reward
        if done:
            #env.reset()
            print('Reward is',epi_rews)
            epi_rews=0.0
        state = next_state
        if keyboard.is_pressed('c'):
            env.close()
            break

