#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:47:21 2023

@author: josh
"""


import os
import time
import math
from datetime import datetime
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import gym

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

from GPyOpt.methods import BayesianOptimization

from BO_CMAES_Coding_1_2 import BO_CMAES
from BO_CMAES_Coding_1_2 import set_bounds

seed = 8902077161928034768
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
model = DDPG.load("ddpg_mountain")

from gym import spaces
def compute_traj(x):
    env.reset()
    ob = x[0:2]
    env.env.state = ob
    #gp = kwargs['goal_pos']
    #env.env.goal_position = gp
    ms = x[2]
    env.env.max_speed = ms
    env.env.low_state = \
        np.array([env.env.min_position, - env.env.max_speed])
    env.env.high_state = \
        np.array([env.env.max_position, env.env.max_speed])
    env.env.observation_space = \
        spaces.Box(env.env.low_state, env.env.high_state)
    pow = x[3]
    env.env.power = pow
    max_steps = 200
    #max_steps = np.inf

    iter_time = 0
    reward = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        action, _states = model.predict(ob)
        ob, rewards, dones, info = env.step(action)
        
        traj.append(ob)
        #reward += r
        done = done or (iter_time >= max_steps)
        if done:
            break
    return traj

def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness=[]
        for i in range (len(traj)):
            #x_pos=np.array(traj1[0]).T[i]
            #angle=np.array(traj[0]).T[i]
            x_pos=traj[i][0]
            velocity=traj[i][1]
            if x_pos <= -1.1 or x_pos>=0.5:
                Robustness.append(0.0735-abs(velocity))
            if x_pos>-1.1 and x_pos<0.5:
                Robustness.append(1/abs(x_pos))
        Y.append(min(Robustness))
    return np.array(Y)

def pred2(x):
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness=[]
        Until_Con=0
        for i in range (len(traj)):
            #x_pos=np.array(traj1[0]).T[i]
            #angle=np.array(traj[0]).T[i]
            x_pos=traj[i][0]
            velocity=traj[i][1]
            if x_pos>0.1:
                Until_Con+=1
            if x_pos<=0.1:
                Until_Con+=0
            if Until_Con<1:
                Robustness.append(0.055-abs(velocity))
            if Until_Con>=1:
                Robustness.append(1)
        Y.append(min(Robustness))
    return np.array(Y)

def pred3(x):
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness1=[]
        Robustness2=[]
        Until_Con=0
        for i in range (len(traj)):
            #x_pos=np.array(traj1[0]).T[i]
            #angle=np.array(traj[0]).T[i]
            x_pos=traj[i][0]
            velocity=traj[i][1]
            if x_pos <= -1.1 or x_pos>=0.5:
                Robustness1.append(0.0735-abs(velocity))
            if x_pos>-1.1 and x_pos<0.5:
                Robustness1.append(1/abs(x_pos))
                #x_pos=np.array(traj1[0]).T[i]
                #angle=np.array(traj[0]).T[i]
            if x_pos>0.1:
                Until_Con+=1
            if x_pos<=0.1:
                Until_Con+=0
            if Until_Con<1:
                Robustness2.append(0.055-abs(velocity))
            if Until_Con>=1:
                Robustness2.append(1)     
        A=min(Robustness1)
        B=min(Robustness2)
        Y.append(min(A,B))
    return np.array(Y)

bounds=np.array([[-0.6,-0.4],[-0.025, 0.025],[0.040, 0.075],[0.0005, 0.0025]])
B=set_bounds(bounds)

#rand_num=[16245,18762,3199,921,2817];
#rand_num=[6245,8762,399,9212,17281];
#rand_num=[3,983,80231,9982,45007]
#rand_num=[92311,4671,47336,34009,61722]
#rand_num=[298371,2873,981182,64783,472615];

"""
[ 

"
 """

rand_num=[39884,
 7416477, 6937979,
 362630,
 8343444, 7352771,
 72065,
 4620367,
 8374652,
 8937830,
 7394200, 7440187,
 47902, 1419350,
  672765,
  1667189,
  4042873,
  3923469,
 70453, 7601482,
 376235,
 7243576,
 1140050, 
 5067883,
 1602627,
 8527042,
 42931,
 2206796,
 7066737,
 5501058,
 1819615,
 1422982,  421342,
 3244436,
 7888969,
 5256551,5820101,
 5149514,
 6287275,
 1785174,
 863562,
 9645204,
  42631,
 6286605,
 9266879,
 37273,
 8466385,
 2298580,
 1737255,
 6693299,
 6977644,
 8324792,
 4307562,
 4601977,
 3629290,
 6877172,
 8432515,
 1097046,
 7086261,
 3407994,
 51518,
 3660088,
 2042563,
 9620812,
 93757,
 6894255,
 205315,
 7931212,
 9892136,
 85646,
 2292711,
 5485653,
 512924,
 101859,
 71920,
 1753255,
 236167,
 4894010,
 4451580,
 2965513,
 5005467,
 2297190,
 1820036,
 7827962,
 8040956,
 7000558,
 8142030,
 7953048,
 9012635,
 107525,
 5234854,
 4503544,
 3768285,
 7879863,
 9401,
 4559014,
 9754585,
 8835131, 
 4276492,
 3798329,
 5967911,
 4734700,
 2495034,
 3290624,
 1758134,
 7645142, 
 1694751,
 1484072,
 6814805,
 9046141,
 4028556,
 5971602,
 401210,
 1791682,
 9730616, 
 3608853,
 3627847,
 8762286,
 4360604,
 5405567,
 6040274,
 2292442,
 1667008,
 4400069,
 3336946,
 8036906,
 9362987,
 3066,
 5816064,
 1117882,
 1826281,
 4126728, 
 75914,
 7578678,
 4515548,
 9928210,
 4154043,
 9386424,
 3401793,
 8549955,
 9040403,
 752861,
 2623,
 66964129,
 882812,
 2850231,
 8169720,
 825940,
 895867,
 168869
]
 

"""
"""


BO_Robust=[]
BO_Counter=[]
BCMA_Robb=[]
BCMA_Counter=[]
BCMA_Jump=[]
for r in rand_num:
    np.random.seed(r)

    Test=BO_CMAES(spec=pred3,boundary=bounds,budget=500,population_size=20,jump_con=2)
    Test.initialize()
    Test.run_BO_CMA()
    Test.get_violation_count()
    BCMA_Counter.append(Test.Violation_Count)
    BCMA_Robb.append(Test.Global_Min_Robust)
    BCMA_Jump.append(Test.Record_jump)
    print('------------------------------------------------------------------')
    
    BO_Test=BayesianOptimization(f=pred3,domain=B, acquisition_type='LCB',initial_design_numdata=20)
    BO_Test.run_optimization(max_iter=480,eps=-1)
    BO_Counter.append(np.sum(BO_Test.Y<0))
    BO_Robust.append(BO_Test.Y.min())
    print('******************************************************************')
    

