#!/bin/bash
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:06:48 2023

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
    param_convert=np.divide(inp,100) #divide by 100 for correct scale
    param_convert2=param_convert.tolist()
    Input=matlab.double(param_convert2) #convert to matlab format
    Out=eng.sim_AT(Input,nargout=6) #run sim
    time=np.array(Out[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.array(Out[1])
    rpm_out=np.array(Out[2])
    gear_out=np.array(Out[3])
    gear_time=np.array(Out[4])
    traj.append(time)
    traj.append(velocity)
    traj.append(rpm_out)
    traj.append(gear_out)
    traj.append(gear_time)
    return traj

def sut(x0):
    return compute_traj(param=x0[0:4])

def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        Max_Speed=80
        Robustness=[]
        for i in range(len(time)):
            Robustness.append(Max_Speed-velocity[i])
        Y.append(min(Robustness))
    return np.array(Y)
    
#output engine speed should always be below 1400rpm
def pred2(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        Max_RPM=1400
        Robustness=[]
        for i in range(len(time)):
            Robustness.append(Max_RPM-rpm_out[i])
        Y.append(min(Robustness))
    return np.array(Y)

#3rd predicate
def pred_extra(x):
    Y=[]

#If we are not in 1st gear for 30 seconds then shift to it, we must remain in
#1st gear for 2.5 seconds
def pred_3rd(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        gear_time=np.asarray(traj[4])
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        gear_out=np.asarray(traj[3])
        Robustness=[]
        not_gear_1=0
        req1=0
        in_gear=0
        for i in range(len(gear_time)):
            if req1==1 and gear_out[i]==3:
                in_gear+=1                              #record time in gear
            if req1==1 and in_gear>0 and gear_out[i]!=3:
                Robustness.append(in_gear-63)
                in_gear=0
                req1=0
                not_gear_1=0
            if req1==0:
                if gear_out[i]!=2:
                    not_gear_1+=1
                if not_gear_1>=750:
                    req1=1               #determines if first part is met
        if len(Robustness)==0:
            Robustness.append(np.sum(np.array(gear_out)==3))
        Y.append(min(Robustness))
    return np.array(Y)
            
        

def pred3(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        Max_Speed=80
        Max_RPM=1400
        Robustness1=[]
        Robustness2=[]
        for i in range(len(time)):
            Robustness1.append(Max_Speed-velocity[i])
            Robustness2.append(Max_RPM-rpm_out[i])
        A=min(Robustness1)
        B=min(Robustness2)
        Y.append(min(A,B))
    return np.array(Y)
        
bounds=np.array([[0.,100.],[0.,100.],[0.,100.],[0.,100.]])
B=set_bounds(bounds)

"""

"""

rand_num=[456472,
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
 421342,3244436,2298580,
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
 3627847, 8762286,
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
 645524,
 7888969,
 5256551,
 5820101,
 5149514, 6287275,
 1785174,
 863562,
 9645204,
  42631,
 6286605,
 96879,
 37273,
 8466385]
"""

 
 ]
"""
rand_test=[28476]
BO_Robust=[]
BO_Counter=[]
BCMA_Robb=[]
BCMA_Counter=[]
BCMA_Jump=[]
#for r in rand_num:
for r in rand_num:
    np.random.seed(r)
    
    Test=BO_CMAES(spec=pred3,boundary=bounds,budget=300,population_size=20,jump_con=2)
    Test.initialize()
    Test.run_BO_CMA()
    Test.get_violation_count()
    BCMA_Counter.append(Test.Violation_Count)
    BCMA_Robb.append(Test.Global_Min_Robust)
    BCMA_Jump.append(Test.Record_jump)
    
    BO_Test=BayesianOptimization(f=pred3,domain=B, acquisition_type='LCB',initial_design_numdata=20)
    BO_Test.run_optimization(max_iter=280,eps=-1)
    BO_Counter.append(np.sum(BO_Test.Y<0))
    BO_Robust.append(BO_Test.Y.min())
    
    print('****************************************************************')
    