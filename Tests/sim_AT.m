function [time_output,velocity_output,rpm_output,gear_out,gear_time,rpm_input] = sim_AT(params)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%create throttle/break data
%load sim
load_system('sdl_car_gear');
%input data
time_input=[0 15 15 50];
%time_input=[0 5 5 10 10 50];
%throttle_input=[params(1) params(1) params(2) params(2) params(3) params(3)]';
%brake_input=[params(4) params(4) params(5) params(5) params(6) params(6)]';
throttle_input=[params(1) params(1) params(2) params(2)]';
brake_input=[params(3) params(3) params(4) params(4)]';

%create time series
Thr=timeseries(throttle_input,time_input,Name='Thr');
Brk=timeseries(brake_input,time_input,Name='Brk');
%put into dataset
ds=Simulink.SimulationData.Dataset;
ds{1}=Brk;
ds{2}=Thr;

%adjust group
signalbuilder('sdl_car_gear/Driver Inputs','set',5,ds);
%set active group
signalbuilder('sdl_car_gear/Driver Inputs','activegroup',5);

%run sim
sim('sdl_car_gear')
time_output=simlog_sdl_car.Inertia_Input_Shaft.w.series.time;
velocity_output=simlog_sdl_car.Vehicle_Body.Vehicle_Body.v.series.values('mph');
rpm_output=simlog_sdl_car.Inertia_Output_Shaft.w.series.values('rpm');
rpm_input=simlog_sdl_car.Inertia_Input_Shaft.w.series.values('rpm');
gear_out=Gear_Out.Data;
gear_time=Gear_Out.Time;
end
