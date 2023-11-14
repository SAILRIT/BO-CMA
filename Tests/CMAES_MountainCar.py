#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:53:47 2022

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
from cmaes import CMA

import logging
import numpy as np
from gym import spaces
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

seed = 8902077161928034768
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
model = DDPG.load("ddpg_mountain")

def compute_traj(**kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        env.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        env.env.max_speed = ms
        env.env.low_state = \
            np.array([env.env.min_position, - env.env.max_speed])
        env.env.high_state = \
            np.array([env.env.max_position, env.env.max_speed])
        env.env.observation_space = \
            spaces.Box(env.env.low_state, env.env.high_state)
    if 'power' in kwargs:
        pow = kwargs['power']
        env.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

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

def pred1(traj):
    traj1 = traj
    Robustness=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        velocity=traj1[i][1]
        if x_pos <= -1.1 or x_pos>=0.5:
            Robustness.append(0.0735-abs(velocity))
        if x_pos>-1.1 and x_pos<0.5:
            Robustness.append(1/abs(x_pos))
    return min(Robustness)

def pred2(traj):
    traj1 = traj
    Robustness=[]
    Until_Con=0
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        velocity=traj1[i][1]
        if x_pos>0.1:
            Until_Con+=1
        if x_pos<=0.1:
            Until_Con+=0
        if Until_Con<1:
            Robustness.append(0.055-abs(velocity))
        if Until_Con>=1:
            Robustness.append(1)
    return min(Robustness)

bounds=np.array([[-0.6,-0.4],[-0.025, 0.025],[0.040, 0.075],[0.0005, 0.0025]])

#optimizer=CMA(mean=np.array([-0.5,0,0.0575,0.0015]), sigma=1.3,bounds=bounds,population_size=(5))
print(" g    f(x1,x2)     x1      x2  ")
print("===  ==========  ======  ======")

C=[rand_num]
Exp_First_Failure_Gen=[] #Gen first Failure is found in test
Exp_Min_Rob=[]
Exp_Min_Gen=[]
Exp_Min_Params=[] #params for lowest robustness in test
Exp_Failures_Found=[] #number of generations where failures were found
Exp_Failures_Count=[] 
Exp_Gens_till_Stop=[]

for a in range(len(C)):
    Test_min_gen=[] #Gen where lowest robustness was found in test
    Test_min_rob=[] #lowest Robustness Score in test
    Test_Params=[] #list of params for worst one in each gen
    First_Failure_Gen=[] #Gen first Failure is found in test
    Test_Robust_Gens=[] #lowest robustness in each gen
    Test_Min_Params=[] #params for lowest robustness in test
    Test_Failure_Count=[] #number of counter examples found
    Test_Failures_Found=[] #number of generations with counter examples
    Gens_till_Stop=[]
    I=0
    
    for r in C[a]:
        np.random.seed(r)
        optimizer=CMA(mean=np.array([-0.5,0,0.0575,0.0015]), sigma=1.3,bounds=bounds,population_size=(20),seed=r)
        Generation=[]
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                traj=compute_traj(max_steps=200,init_state=x[0:2],max_speed=x[2],power=x[3])
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                solutions.append((x, value))
                print(
                    f"{optimizer.generation:3d}  {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]})"
                )
            optimizer.tell(solutions)
            Generation.append(solutions)
    
            if optimizer.should_stop():
                break
        Robust=[]
        Min_Param=[]
        Details=[]
        for a in range(len(Generation)):
            Robust.append(np.array(Generation[a]).T[1].min())
            min_arg=np.array(Generation[a]).T[1].argmin()
            Min_Param.append(np.array(Generation[a]).T[0][min_arg])
            for s in range(optimizer.population_size):
                Details.append(np.array(Generation[a]).T[1][s])
        Failure_Count=np.sum(np.array(Details)<0)
        Test_Failure_Count.append(Failure_Count)
        Failures_Found=np.sum(np.array(Robust)<0)
        Test_Failures_Found.append(Failures_Found)
        Test_Robust_Gens.append(Robust)
        Test_Params.append(Min_Param)
        Test_min_rob.append(np.array(Robust).min())
        Test_min_gen.append(np.array(Robust).argmin())
        Test_Min_Params.append(Test_Params[I][Test_min_gen[I]])
        if np.sum(np.array(Robust)<0)==0:
            First_Failure_Gen.append(0)
        else:
            First_Failure_Gen.append(np.where(np.array(Robust)<0)[0][0])
        Gens_till_Stop.append(len(Generation))
        I+=1
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    
    
    #Running with same number of simulations or designated # of generations
    """
    Test_min_gen=[]
    Test_min_rob=[]
    First_Failure_Gen=[]
    Test_Params=[]
    Test_Robust_Gens=[]
    Test_Min_Params=[]
    """
    """
    for r in C[a]:
        np.random.seed(r)
        optimizer=CMA(mean=np.array([-0.5,0,0.0575,0.0015]), sigma=1.3,bounds=bounds,population_size=(5),seed=r)
        Generation=[]
        for generation in range(30):
            solutions=[]
            for i in range(optimizer.population_size):
                x = optimizer.ask()
                traj=compute_traj(max_steps=200,init_state=x[0:2],max_speed=x[2],power=x[3])
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                solutions.append((x, value))
                print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]})")
            optimizer.tell(solutions)
            Generation.append(solutions)
    #Min robustness and parameters per generation
        Robust=[]
        Min_Param=[]
        Details=[]
        for a in range(len(Generation)):
            Robust.append(np.array(Generation[a]).T[1].min())
            min_arg=np.array(Generation[a]).T[1].argmin()
            Min_Param.append(np.array(Generation[a]).T[0][min_arg])
            for s in range(optimizer.population_size):
                Details.append(np.array(Generation[a]).T[1][s])
        Failure_Count=np.sum(np.array(Details)<0)
        Test_Failure_Count.append(Failure_Count)
        Failures_Found=np.sum(np.array(Robust)<0)
        Test_Failures_Found.append(Failures_Found)
        Test_Robust_Gens.append(Robust)
        Test_Params.append(Min_Param)
        Test_min_rob.append(np.array(Robust).min())
        Test_min_gen.append(np.array(Robust).argmin())
        Test_Min_Params.append(Test_Params[I][Test_min_gen[I]])
        if np.sum(np.array(Robust)<0)==0:
            First_Failure_Gen.append(0)
        else:
            First_Failure_Gen.append(np.where(np.array(Robust)<0)[0][0])
        Gens_till_Stop.append(len(Generation))
        I+=1
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    """