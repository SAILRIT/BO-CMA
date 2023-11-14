#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:44:09 2023

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


eng = matlab.engine.start_matlab() #connect matlab
#matlab.engine.find_matlab()
#eng=matlab.engine.connect_matlab()

def compute_traj(**kwargs):
    if 'param' in kwargs:
        inp = kwargs['param']
    traj=[]
    param_convert=inp.tolist() #divide by 100 for correct scale
    Input=matlab.double(param_convert) #convert to matlab format
    Out=eng.NN_Sim(Input,nargout=3) #run sim
    time=np.array(Out[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.array(Out[1])
    Ref=np.array(Out[2])
    traj.append(time)
    traj.append(Position)
    traj.append(Ref)
    return traj

def sut(x0):
    return compute_traj(param=x0[0:8])
"""

 """
rand_num=[876228,
 436004,
 54567,
 600274, 
 292442,
 667008, 
 404069,
 336946,
 80306,
 932987,
 304266,
 58064,
 117882,
 18281,
 41678, 
 759124,
 75786,
 451538,
 99282210,
 41543043,
 9383724,
 3406793,
 39884,
 7416477,
 6937979,
 362630,
 8343444,
 7352771,
 7206755,
 4620367,
 8374652,
 8937830,
 7394200, 
 7440187,
 47902,
  1419350,
  672765,
  1667189,
  4042873,
  3923469,
 70453,
 7601482,
 3762,
 7243576,
 1140050, 
 5067883,
 1602627,
 8527042,
 42931,
 2206796,
 7066737,
 5501058, 
 181615,
 142982, 
 42142,
 324436,
 7888969,
 525551,
 582101,
 514514,
 628275,
 178455174,
 86356672,
 964520488,
  4263167,
  62866405,
 92668539,
 3727334,
 846635,
 229180,
 17255,
 66299,
 69234444,
 83234792,
 43075182,
 460197,
 362990,
 687172, 
 84515,
 19046,
 70261,
 3994,
 510958, 
 36688,
 20563,
 90812,
 987757,
 69455,
 203185,
 79322,
 9836,
 85646,
 2292711,
 5485653,
 512924,
 101859,
 71920,
 1753255,
 23667,
 4894010,
 445580,
 296513,
 500467,
 229190, 
 182036,
 782962,
 804956,
 700558,
 814030,
 795048, 
 901635,
 10725,
 523854,
 450544,
 376285,
 787863,
 949701, 455914,
 975485,
 883531, 
 427692,
 379829,
 596711,
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
 
 854955,
 904403,
 752861,
 2623,
 66964129,
 882812,
 2850231,
 8169720,
 825940,
 895867,
 168869]

#output height should always remain below 3.9
def pred1(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Max_Height=3.9
    Robustness=[]
    for i in range(len(time)):
        Robustness.append(Max_Height-Position[i])
    return min(Robustness)

  
def pred2(traj):
    traj1=traj
    in_time=[5,10,15,20,25,30,35]
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Settle_Time=100
    Robustness=[]
    for i in range(len(in_time)):
        segment_start=time==in_time[i]
        segment_end=time==in_time[i]+2
        Iter_start=np.argmax(segment_start)
        Iter_end=np.argmax(segment_end)
        settle=0
        for a in range(Iter_start,Iter_end):
            if abs(Position[a]-Ref[a])<=(0.005+0.04*abs(Ref[a])):
                settle+=1
            if abs(Position[a]-Ref[a])>(0.005+0.04*abs(Ref[a])):
                settle+=0
        Robustness.append(settle-Settle_Time)
    Robustness=np.divide(Robustness,100)
    return min(Robustness)

bounds=np.array([[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.]])

print(" g    f(x1,x2)     x1      x2  ")
print("===  ==========  ======  ======")

#C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,
   #rand_nums7,rand_nums8,rand_nums9,rand_nums10]
#C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5]
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
        optimizer=CMA(mean=np.array([2,2,2,2,2,2,2,2]), sigma=1.3,bounds=bounds,population_size=(20),seed=r)
        Generation=[]
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                traj=compute_traj(param=x[0:8])
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                #solutions.append((x, value))
                solutions.append((x, value))
                print(
                    f"{optimizer.generation:3d}  {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]},x5 = {x[4]},x6 = {x[5]},x7 = {x[6]},x8 = {x[7]})"
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