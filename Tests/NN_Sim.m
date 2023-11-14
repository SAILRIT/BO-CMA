function [time_output,position_output,reference_output] = NN_Sim(params)
%UNTITLED Summary of this function goes here
load_system('narmamaglev');
load("Neural_Network_Weights.mat")
%   Detailed explanation goes here
time_in=[0 5 5 10 10 15 15 20 20 25 25 30 30 35 35 40];
%Input_sig=[2.3 2.3 1.4 1.4 1 1 2.8 2.8 2.1 2.1 3 3 1.7 1.7 1.1 1.1]';
Input_sig=[params(1) params(1) params(2) params(2) params(3) params(3) params(4) params(4) params(5) params(5) params(6) params(6) params(7) params(7) params(8) params(8)]';
Input=timeseries(Input_sig,time_in,Name='Signal 1');
ds=Simulink.SimulationData.Dataset;
ds{1}=Input;
signalbuilder('narmamaglev/Signal Builder','set',1,ds);
signalbuilder('narmamaglev/Signal Builder','activegroup',1);

sim('narmamaglev');
Pos=sigsOut{1}.extractTimetable();
Ref_Sig=sigsOut{2}.extractTimetable();
Reference=Ref_Sig.Data;
Time=Pos.Time;
Pos_Out=Pos.Position;
Time_Out=seconds(Time);


time_output=Time_Out;
position_output=Pos_Out;
reference_output=Reference;
end