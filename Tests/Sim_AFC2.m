function [AF_out,Time,Ped_A,Eng_A] = Sim_AFC2(A)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load_system('AbstractFuelControl_M2.slx');
tim_in=[0 5 5 10 10 15 15 20 20 25 25 30 30 35 35 40 40 45 45 50];
Pedal_In=[A(1) A(1) A(2) A(2) A(3) A(3) A(4) A(4) A(5) A(5) A(6) A(6) A(7) A(7) A(8) A(8) A(9) A(9) A(10) A(10)]';

%Pedal_In=[30 30 36 36 42 42 39 39 47 47 53 53 56 56 48 48 52 52 21 21];
%Eng_In=[981 981 981 981 981 981 981 981 981 981 981 981 981 981 981 981
%981 981 981 981]
Eng_In=[A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11) A(11)]';
Pedal=timeseries(Pedal_In,tim_in,Name='Pedal');
Eng=timeseries(Eng_In,tim_in,Name='Eng');
ds=Simulink.SimulationData.Dataset;
ds{1}=Pedal;
ds{2}=Eng;

signalbuilder('AbstractFuelControl_M2/Model 1/Signal Builder','set',1,ds);
signalbuilder('AbstractFuelControl_M2/Model 1/Signal Builder','activegroup',1);

sim('AbstractFuelControl_M2');
AF_out=AF.Data;
Time=AF.Time;
Ped_A=Ped_Ang_In.Data;
Eng_A=Eng_Spd.Data;
%AF_ref=AF_Ref.Data;
%mode=Controller.Data;
end