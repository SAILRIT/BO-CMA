#!/bin/bash
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:19:52 2023

@author: josh
"""

import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

import numpy as np
from numpy import array

import pandas as pd

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

from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot

def compute_traj(x):
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = x[3]        # altitude (ft)
    vt = x[4]          # 540 initial velocity (ft/sec)
    phi = x[0]           # Roll angle from wings level (rad)
    theta = x[1]         # Pitch angle from nose level (rad)
    psi = x[2]  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 3.51 # simulation time

    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

    step = 1/30
    traj = run_f16_sim(init, tmax, ap, step=step, extended_states=True)

    print(f"Simulation Completed in {round(traj['runtime'], 3)} seconds")
    return traj

def pred1(x):   #altitude>0
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness=[]
        for i in range (len(traj['states'])):
           Robustness.append(traj['states'][i][11]-0)
        Y.append(min(Robustness))
    return np.array(Y)

#bounds=np.array([[0.2*np.pi,0.2833*np.pi],[-0.5*np.pi,-0.54*np.pi],[0.25*np.pi,0.375*np.pi]])
bounds=np.array([[0.6283,0.8900],[-1.6964,-1.5707],[0.7853,1.17809],[900,4000],[340,740]])
B=set_bounds(bounds)

"""

"""

rand_num=[
 876228,39884,
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
    