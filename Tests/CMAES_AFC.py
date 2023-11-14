#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:02:10 2023

@author: josh
"""

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
    #mode=np.array(Out[5])
    traj.append(time)
    traj.append(AF)
    traj.append(Ped_Ang)
    traj.append(Eng_Spd)
   # traj.append(mode)
    return traj

def sut(x0):
    return compute_traj(param=x0[0:11])

def pred1(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #201=10s
    AF=np.asarray(traj1[1])
    Pedal=np.asarray(traj1[2])
    Eng_Spd=np.asarray(traj1[3])
    tol=.007
    Robustness=[]
    for i in range(len(time)-4070):
        Robustness.append(tol*14.7-abs(AF[i+4070]-14.7))
    return min(Robustness)

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
 1785174,
 863562,
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
 6455244]
"""

rand_num=[
]
"""

 
 

#rand_num=[16245,18762,3199,921,2817];
#rand_num=[65,8762,399,1212,7281];
#rand_num=[3,983,80231,9982,45007]
#rand_num=[92311,4671,436,34009,61722]
#rand_num=[298371,2873,31182,64783,472615];
C=[rand_num]
#C=[rand_nums_test]
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
        optimizer=CMA(mean=np.array([30,30,30,30,30,30,30,30,30,30,1000]), sigma=1.3,bounds=bounds,population_size=(20),seed=r)
        Generation=[]
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                traj=compute_traj(param=x[0:11])
                value1=pred1(traj)
                #solutions.append((x, value))
                solutions.append((x, value1))
                #print(
                 #   f"{optimizer.generation:3d}  {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]})"
                #)
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
        print('****************************************************************')
        print(I)
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    print('####################################################################')