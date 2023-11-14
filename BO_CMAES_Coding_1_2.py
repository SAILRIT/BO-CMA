#!/bin/bash
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:17:33 2023

@author: josh
"""

import numpy as np
from GPyOpt.methods import BayesianOptimization
from cmaes import CMA
import matplotlib.pyplot as plt

"""
example problem
def quadratic(x):
    return (x[:,0] - 3) ** 2 + (10 * (x[:,1] + 2)) ** 2
bounds=np.array([[-10,10],[-10,10]])

def six_hump_camel(x):
    return (4-2.1*(x[:,0])**2+((x[:,0]**4)/3))*(x[:,0])**2+x[:,0]*x[:,1]+(-4+4*(x[:,1]**2))*(x[:,1]**2)
    
bounds=np.array([[-3,3],[-2,3]])

"""

#Code for BO_CMAES
class BO_CMAES:
    def __init__(
            self,
            spec,
            boundary,       #bounds=np.array([[-10,10],[-10,10]])
            budget,
            population_size,
            jump_con):
        self.spec=spec              #Robustness function with built in simulation
        self.global_bounds=boundary     #global boundaries of variables
        self.population_size=population_size #number of iterations per localized model
        self.budget=budget
        self.Global_Sim_Results=[]
        self.Record_means=[]
        self.Records_bounds=[]
        self.Record_Min_Rob=[]
        self.Record_jump=[]
        self.jump_con=jump_con
        
        
    def initialize(self):
        #initiate CMAES to calculate variance
        self.Sim_count=0
        self.start_mean=np.zeros(len(self.global_bounds))
        for i in range(len(self.global_bounds)):
            self.start_mean[i]=np.mean(self.global_bounds[i])
        
        self.CMA_optimizer=CMA(mean=self.start_mean,sigma=1.3,
                          bounds=self.global_bounds,
                          population_size=(self.population_size))
        self.CMA_optimizer._mu=self.population_size//4       #set number of best ind for eval
        #initialize fist sampling
        format_glob_bound=set_bounds(self.global_bounds)
        self.global_opt=BayesianOptimization(f=self.spec,domain=format_glob_bound,
                                        acquisition_type='LCB',
                                        initial_design_numdata=self.population_size)
        self.Sim_count+=len(self.global_opt.X)
        solutions=[]
        for _ in range(self.population_size):
            solutions.append((self.global_opt.X[_],self.global_opt.Y[_]))
        self.CMA_optimizer.tell(solutions)
        self.Global_Min_Robust=self.global_opt.Y.min()
        self.Local_Min_Rob=self.Global_Min_Robust
        self.Record_Min_Rob.append(self.Global_Min_Robust)
        self.Global_Sim_Results.append(solutions)
        self.stagnant_count=0   #counts consecutive generations with no improvement in robustness
#-----------------------------------------------------------------------------
    def Local(self):        
        self.sigma=self.CMA_optimizer._sigma
        self.mean=self.CMA_optimizer._mean
        variance=[]
        for i in range(len(self.CMA_optimizer._C)):
            variance.append(self.CMA_optimizer._C[i][i])
        dev=np.sqrt(variance)
        self.local_bounds=[]
        for i in range(len(self.global_bounds)):
            lower_bounds=self.mean[i]-self.sigma*dev[i]
            upper_bounds=self.mean[i]+self.sigma*dev[i]
            self.local_bounds.append([lower_bounds,upper_bounds])
        #set bounds to global bound if value is outside of it
        self.local_bounds=repair_bounds_2(self.local_bounds,self.global_bounds)
        self.Records_bounds.append(self.local_bounds)
        self.Record_means.append(self.mean)
        format_local_bounds=set_bounds(self.local_bounds)
        prior_knowledge(self)
        if len(self.local_prior_param)<(self.population_size//4):
            self.local_opt=BayesianOptimization(f=self.spec,domain=format_local_bounds,
                                            acquisition_type='LCB',
                                            initial_design_numdata=self.population_size//4)
            self.local_opt.run_optimization(max_iter=self.population_size-(self.population_size//4),eps=-1)
            self.Sim_count+=len(self.local_opt.X)
            solutions=[]
            #for _ in range(self.population_size):
            for _ in range(len(self.local_opt.X)):
                solutions.append((self.local_opt.X[_],self.local_opt.Y[_]))
        if len(self.local_prior_param)>=(self.population_size//4):
            self.local_opt=BayesianOptimization(f=self.spec,domain=format_local_bounds,
                                            acquisition_type='LCB',
                                            X=np.array(self.local_prior_param),
                                            Y=np.array(self.local_prior_robust)
                                            )
            self.local_opt.run_optimization(max_iter=self.population_size,eps=-1)
            self.Sim_count+=len(self.local_opt.X)-len(self.local_prior_param)
            solutions=[]
            #for _ in range(self.population_size):
            for _ in range(len(self.local_opt.X)-len(self.local_prior_param)):
                solutions.append((self.local_opt.X[_+len(self.local_prior_param)],self.local_opt.Y[_+len(self.local_prior_param)]))
        """
        self.local_opt=BayesianOptimization(f=self.spec,domain=format_local_bounds,
                                        acquisition_type='LCB',
                                        initial_design_numdata=self.population_size//4)
        self.local_opt.run_optimization(max_iter=self.population_size-(self.population_size//4))
        self.Sim_count+=len(self.local_opt.X)
        solutions=[]
        for _ in range(self.population_size):
            solutions.append((self.local_opt.X[_],self.local_opt.Y[_]))
        """
        if len(solutions)!=self.population_size:
            self.local_opt.run_optimization(max_iter=(self.population_size-len(solutions)),eps=-1)
            self.Sim_count+=self.population_size-len(solutions)
            for _ in range(self.population_size-len(solutions)):
                solutions.append((self.local_opt.X[-1-_],self.local_opt.Y[-1-_]))
   
        self.CMA_optimizer.tell(solutions)
        self.Global_Sim_Results.append(solutions)
        self.Record_Min_Rob.append(self.local_opt.Y.min())
        self.Record_jump.append(0)
        #check robustness
        if self.local_opt.Y.min()<self.Local_Min_Rob:
            self.Local_Min_Rob=self.local_opt.Y.min()
            if self.Local_Min_Rob<self.Global_Min_Robust:
                self.Global_Min_Robust=self.Local_Min_Rob
            self.stagnant_count=0
        else:
            self.stagnant_count+=1
            
    def G_Jump(self):
        #creates updated global model to choose new start point far from search space
        self.Param_Val=[]
        self.Robust_Val=[]
        for _ in range(len(self.Global_Sim_Results)):
            for i in range(len(self.Global_Sim_Results[_])):
                self.Param_Val.append(self.Global_Sim_Results[_][i][0])
                self.Robust_Val.append(self.Global_Sim_Results[_][i][1])
        format_glob_bound=set_bounds(self.global_bounds)
        self.global_model=BayesianOptimization(f=self.spec,domain=format_glob_bound,
                                        acquisition_type='EI',
                                        X=np.array(self.Param_Val),
                                        Y=np.array(self.Robust_Val))
        Next_point=self.global_model.suggest_next_locations()
        self.mean=Next_point[0]
        #create and run local BO
        self.sigma=self.CMA_optimizer._sigma
        variance=[]
        for i in range(len(self.CMA_optimizer._C)):
            variance.append(self.CMA_optimizer._C[i][i])
        dev=np.sqrt(variance)
        self.local_bounds=[]
        for i in range(len(self.global_bounds)):
            lower_bounds=self.mean[i]-self.sigma*dev[i]
            upper_bounds=self.mean[i]+self.sigma*dev[i]
            self.local_bounds.append([lower_bounds,upper_bounds])
        #set bounds to global bound if value is outside of it
        self.local_bounds=repair_bounds_2(self.local_bounds,self.global_bounds)
        self.Records_bounds.append(self.local_bounds)
        self.Record_means.append(self.mean)
        format_local_bounds=set_bounds(self.local_bounds)
        self.local_opt=BayesianOptimization(f=self.spec,domain=format_local_bounds,
                                        acquisition_type='LCB',
                                        initial_design_numdata=self.population_size//4)
        self.local_opt.run_optimization(max_iter=self.population_size-(self.population_size//4),eps=-1)
        self.Sim_count+=len(self.local_opt.X)
        solutions=[]
        self.local_sol=[]
       # for _ in range(self.population_size):
        for _ in range(len(self.local_opt.X)):
            solutions.append((self.local_opt.X[_],self.local_opt.Y[_]))
            self.local_sol.append((self.local_opt.X[_],self.local_opt.Y[_]))
        sol_size=len(solutions)
        if len(solutions)!=self.population_size:
            self.local_opt.run_optimization(max_iter=(self.population_size-len(solutions)),eps=-1)
            self.Sim_count+=self.population_size-len(solutions)
            #for _ in range(self.population_size-len(solutions)):
            for _ in range(len(self.local_opt.X)-len(solutions)):
                solutions.append((self.local_opt.X[-1-_],self.local_opt.Y[-1-_]))
                self.local_sol.append((self.local_opt.X[-1-_],self.local_opt.Y[-1-_]))
        self.CMA_optimizer.tell(solutions)
        self.Global_Sim_Results.append(solutions)
        self.Local_Min_Rob=self.local_opt.Y.min()
        self.Record_Min_Rob.append(self.local_opt.Y.min())
        self.Record_jump.append(1)
        if self.Local_Min_Rob<self.Global_Min_Robust:
            self.Global_Min_Robust=self.Local_Min_Rob
        self.stagnant_count=0
        
    def run_BO_CMA(self):
        while self.Sim_count<self.budget:
            #if self.stagnant_count<=2:
            if self.stagnant_count<=self.jump_con:
                self.Local()
            #if self.stagnant_count>2:
            if self.stagnant_count>self.jump_con:
                self.G_Jump()
      
    def get_violation_count(self):
        Robustness=[]
        for i in range(len(self.Global_Sim_Results)):
            Robustness.append(np.array(self.Global_Sim_Results[i]).T[1])
        self.Violation_Count=np.sum(np.array(Robustness)<0)
    
    def plot_2D_sol(self):
        xx=[]
        yy=[]
        for i in range(len(self.Global_Sim_Results)):
            for c in range(len(self.Global_Sim_Results[i])):

                xx.append(self.Global_Sim_Results[i][c][0][0])
                yy.append(self.Global_Sim_Results[i][c][0][1])
        plt.scatter(xx,yy)
        
    def plot_gens_2D(self):
        for i in range(len(self.Global_Sim_Results)):
            xx=[]
            yy=[]
            for c in range(len(self.Global_Sim_Results[i])):

                xx.append(self.Global_Sim_Results[i][c][0][0])
                yy.append(self.Global_Sim_Results[i][c][0][1])
            plt.scatter(xx,yy)
    def plot_single_gens_2D(self,i):
            xx=[]
            yy=[]
            for c in range(len(self.Global_Sim_Results[i])):

                xx.append(self.Global_Sim_Results[i][c][0][0])
                yy.append(self.Global_Sim_Results[i][c][0][1])
            plt.scatter(xx,yy)
          
    def search_zones_2D(self,i):
        plt.xlim([self.global_bounds[0][0],self.global_bounds[0][1]])
        plt.ylim([self.global_bounds[1][0],self.global_bounds[1][1]])
        #lower line
        plt.axhline(y=self.Records_bounds[i][1][0], xmin=self.Records_bounds[i][0][0],
                        xmax=self.Records_bounds[i][0][1])
        #upper line
        plt.axhline(y=self.Records_bounds[i][1][1], xmin=self.Records_bounds[i][0][0],
                        xmax=self.Records_bounds[i][0][1])
        plt.axvline(x=self.Records_bounds[i][0][0],ymin=self.Records_bounds[i][1][0],
                    ymax=self.Records_bounds[i][1][1])
        plt.axvline(x=self.Records_bounds[i][0][1],ymin=self.Records_bounds[i][1][0],
                    ymax=self.Records_bounds[i][1][1])

#function to define boundaries (both global and local)
def set_bounds(bounds):
    Bounds=[]
    for i in range(len(bounds)):
        Bounds.append({'name':'x'+str(i+1),'type':'continuous','domain':bounds[i]})
    return Bounds
def repair_bounds(A,bounds):
    A=np.where(A<bounds[:,0],bounds[:,0],A)
    A=np.where(A>bounds[:,1],bounds[:,1],A)
    return A
def repair_bounds_2(A1,A2):
    for a in range(len(A2)):
        if A1[a][0]<A2[a][0]:
            A1[a][0]=A2[a][0]
        if A1[a][1]>A2[a][1]:
            A1[a][1]=A2[a][1]
    return A1
def prior_knowledge(self):
    self.local_prior_param=[];
    self.local_prior_robust=[];
    for g in range(len(self.Global_Sim_Results)):       #gen loop
        for i in range(len(self.Global_Sim_Results[g])):    #individual loop
            for x in range(len(self.Global_Sim_Results[g][i][0])):
                A=True;
                if A==True:
                    A=self.Global_Sim_Results[g][i][0][x]>=self.local_bounds[x][0] and self.Global_Sim_Results[g][i][0][x]<=self.local_bounds[x][1];
                else:
                    A=False;
            if A==True:
                self.local_prior_param.append(self.Global_Sim_Results[g][i][0]);
                self.local_prior_robust.append(self.Global_Sim_Results[g][i][1]);
    return self.local_prior_param, self.local_prior_robust

                