#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:28:19 2023

@author: josh
"""

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine

import os
import time
import math
from datetime import datetime
import argparse
import re
import operator
from cmaes import CMA
import logging

from GPyOpt.methods import BayesianOptimization

from BO_CMAES_Coding_1_2 import BO_CMAES
from BO_CMAES_Coding_1_2 import set_bounds

eng = matlab.engine.start_matlab() #connect matlab

def compute_traj(**kwargs):
    if 'param' in kwargs:
        inp = kwargs['param']
    traj=[]
    param_convert=inp.tolist() #divide by 100 for correct scale
    Input=matlab.double(param_convert) #convert to matlab format
    Out=eng.Sim_AFC2(Input,nargout=4) #run sim
    time=np.array(Out[1]) #starting at time[172]=1 sec, every 25 iters=1sec
    AF=np.array(Out[0])
    Ped_Ang=np.array(Out[2])
    Eng_Spd=np.array(Out[3])
    traj.append(time)
    traj.append(AF)
    traj.append(Ped_Ang)
    traj.append(Eng_Spd)
    return traj

def sut(x0):
    return compute_traj(param=x0[0:11])

def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #201=10s
        AF=np.asarray(traj[1])
        Pedal=np.asarray(traj[2])
        Eng_Spd=np.asarray(traj[3])
        tol=.007
        Robustness=[]
        for i in range(len(time)-4070):
            Robustness.append(tol*14.7-abs(AF[i+4070]-14.7))
        Y.append(min(Robustness))
    return np.array(Y)

bounds=np.array([[0,61.1],[0,61.1],[0,61.1],[0,61.1],[0,61.1],[0,61.1],[0,61.1],
                 [0,61.1],[0,61.1],[0,61.1],[900,1100]])
B=set_bounds(bounds)

"""
"""

rand_num=[
456472,
  1490,  
  62376765,
  17189, 
  4042873,
  3923469, 
  70453,
 7601482,
 376235,
 7243576,
 1140050, 
 5067883,
 1602627,
 8527042,
 42931,
 2206796,
 7066737, 
 51058,
 1819615,
 1422982, 
 421342,
 3244436,
 7888969,
 5256551,
 5820101,
 5149514,
 6287275,
 1785174, 863562,
 9645204,
  42631,
 6286605,
 96879,
 37273,
 846638,
 52298580,
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
 44580,
 25513,
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
 9401,4559014,
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
 168869,
 83574763,
 4756283,
 387462,
 83743,
 9302874,
 475623,
 548254,
 71222635,
 234818344,
 1986354,
 219837,
 645524]
"""

rand_num=[
 
 ]
"""

 
 

#Performs BO-CMA and BO tests, change budget and max iter for simulation budget.
#For BO, max iters does not include initial sampling size of 20

BO_Robust=[]
BO_Counter=[]
BCMA_Robb=[]
BCMA_Counter=[]
BCMA_Jump=[]
for r in rand_num:
    np.random.seed(r)
    
    Test=BO_CMAES(spec=pred1,boundary=bounds,budget=500,population_size=20,jump_con=2)
    Test.initialize()
    Test.run_BO_CMA()
    Test.get_violation_count()
    BCMA_Counter.append(Test.Violation_Count)
    BCMA_Robb.append(Test.Global_Min_Robust)
    BCMA_Jump.append(Test.Record_jump)
    
    BO_Test=BayesianOptimization(f=pred1,domain=B, acquisition_type='LCB',initial_design_numdata=20)
    BO_Test.run_optimization(max_iter=480,eps=-1)
    BO_Counter.append(np.sum(BO_Test.Y<0))
    BO_Robust.append(BO_Test.Y.min())
    print('******************************************************************')