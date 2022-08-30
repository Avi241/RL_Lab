# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:40:03 2022

@author: Sumanth Raikar 
"""
import numpy as np
from PIL import Image 
import cv2
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


#PARAMETERS
SIZE=10
N_EPISODES=25000
MOVE_PENALTY=1
ENEMY_PENALTY=300
FOOD_REWARD=100
EPSILON=0.9
EPSILON_DECAY=0.998
SHOW_EVERY=1000

start_q_table=None
LEARNING_RATE=0.1
DISCOUNT=0.95

PLAYER=1
FOOD=2
ENEMY=3

#Dictionary for coloring player,food and enemy blobs(BGR FORMAT)
d={1:(255,175,0),
   2:(0,255,0),
   3:(0,0,255)}


#BLOB CLASS
class BLOB:
    def __init__(self):
        #Randomly initialize locations of blobs
        self.x=np.random.randint(0,SIZE)
        self.y=np.random.randint(0,SIZE)
        
    def __str__(self):
        return (self.x,self.y)
    
    def __sub__(self,other_blob):
        return (self.x-other_blob.x,self.y-other_blob.y)
    
    def action(self,choice):
        #Simulates diagonal movement of blobs
        if choice==0:
            self.move(x=1,y=1)
        elif choice==1:
            self.move(x=1,y=-1)
        elif choice==2:
            self.move(x=-1,y=1)
        elif choice==3:
            self.move(x=-1,y=-1)
     
    def move(self,x=False,y=False):
        if not x:
            self.x+=np.random.randint(-1,2)
        else:
            self.x+=x
        
        if not y:
            self.y+=np.random.randint(-1,2)
        else:
            self.y+=y
            
        #Thwart the movement outside grid
        if self.x<0:
            self.x=0
        elif self.x>SIZE-1:
            self.x=SIZE-1
            
        if self.y<0:
            self.y=0
        elif self.y>SIZE-1:
            self.y=SIZE-1
            
#create Q Table if not present
if start_q_table is None:
    q_table={}
    for x1 in range(-SIZE+1,SIZE):
        for y1 in range(-SIZE+1,SIZE):
            for x2 in range(-SIZE+1,SIZE):
                for y2 in range(-SIZE+1,SIZE):
                    #There are 4 actions for each observation
                    q_table[((x1,y1),(x2,y2))]=[np.random.uniform(-5,0) for i in range(4)]
            
else:
    with open(start_q_table,"rb") as f:
        q_table=pickle.load(f)
        
episode_rewards=[]        
        
for episode in range(N_EPISODES):
    player=BLOB()
    enemy=BLOB()
    food=BLOB()
    
    if episode%SHOW_EVERY==0:
        print(f'Episode {episode} epsilon: {EPSILON}')
        print(f'{SHOW_EVERY} episode mean reward {np.mean(episode_rewards[-SHOW_EVERY:])}')
        show=True
    else:
        show=False
        
    episode_reward=0
    for i in range(200):#This could be varied
        obs = (player-food,player-enemy) #operator overloading
        if np.random.random()>EPSILON:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
            
        player.action(action) #take actions and move player blob
        
        #For enemy and food movements
        #enemy.move()
        #food.move()
        
        if player.x==enemy.x and player.y==enemy.y:
            reward=-ENEMY_PENALTY #remove minus (LOVE THY ENEMY)
            
        elif player.x==food.x and player.y==food.y:
            reward=FOOD_REWARD
            
        else: 
            reward=-MOVE_PENALTY
            
        new_obs = (player-food,player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[new_obs][action]
        
        if reward==FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward==-ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q=(1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+ DISCOUNT*max_future_q)
            
        q_table[obs][action]=new_q
        
        if show:
            env=np.zeros((SIZE,SIZE,3),dtype=np.uint8)
            env[food.y][food.x]=d[FOOD]
            env[player.y][player.x]=d[PLAYER]
            env[enemy.y][enemy.x]=d[ENEMY]
            
            img=Image.fromarray(env,"RGB")
            img=img.resize((300,300))
            cv2.imshow("",np.array(img))
            if reward==FOOD_REWARD or reward==-ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF==ord("q"):
                    break
                
            else:
                if cv2.waitKey(1) & 0XFF==ord("q"):
                    break
                
        episode_reward+=reward
        if reward==FOOD_REWARD or reward==-ENEMY_PENALTY:
            break
        
    episode_rewards.append(episode_reward)
    EPSILON*=EPSILON_DECAY
    
moving_avg=np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="Valid")

plt.plot([i for i in range(len(moving_avg))],moving_avg)            
plt.ylabel(f"Reward {SHOW_EVERY}ma") 
plt.xlabel(f"episode #")
plt.show()

with open(f"q-table{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table,f)
    
    
