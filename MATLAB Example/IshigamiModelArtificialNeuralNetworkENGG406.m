%% TutorialIshigami function: Model Definition and ANN
% <html>
% <h3 style="color:#317ECC">Copyright 2006-2014: <b> COSSAN working group</b></h3>
% Author: <b>Bright-Oparaji</b> <br> 
% <i>Institute for Risk and Uncertainty, University of Liverpool, UK</i>
% <br>COSSAN web site: <a href="http://www.cossan.co.uk">http://www.cossan.co.uk</a>
% <br><br>
% <span style="color:gray"> This file is part of <span style="color:orange">openCOSSAN</span>.  The open source general purpose matlab toolbox
% for numerical analysis, risk and uncertainty quantification (<a
% href="http://www.cossan.co.uk">http://www.cossan.co.uk</a>).
% <br>
% <span style="color:orange">openCOSSAN</span> is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License.
% <span style="color:orange">openCOSSAN</span> is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details. 
%  You should have received a copy of the GNU General Public License
%  along with openCOSSAN. If not, see <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses/"</a>.
% </span></html>

clear
clc
Xrv1 = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrv2 = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrv3 = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrvset = RandomVariableSet('Cmembers',{'Xrv1','Xrv2','Xrv3'},'CXrv',{Xrv1, Xrv2, Xrv3});

% Add two parameters
parameterA = Parameter('value', 7);
parameterB = Parameter('value', 0.1);

% Define Input object
Xinput = Input('CXmembers',{Xrvset, parameterA, parameterB},...
    'CSmembers',{'Xrvset','parameterA','parameterB'});

Xmio = Mio('Sfile','ishigami.m',...
    'Spath',pwd,...
    'Cinputnames',{'Xrv1' 'Xrv2' 'Xrv3' 'parameterA' 'parameterB'}, ...
    'Coutputnames',{'out'},'Liostructure',true);
% Add the MIO object to an Evaluator object
Xevaluator = Evaluator('CXmembers',{Xmio},'CSmembers',{'Xmio'});

%% Preparation of the Physical Model
% Define the Physical Model
Xmod = Model('Xinput', Xinput, 'Xevaluator', Xevaluator);
%% Define sampler used in the analysis
Xls = LatinHypercubeSampling('Nsamples',1e3,'Lintermediateresults',false);
%% Apply LHS algorithm to model 
Xout = Xls.apply(Xmod);
%% Prepare data for training ANN
Minput=Xout.getValues('CSnames',Xmod.Xinput.CnamesRandomVariable);
Voutput=Xout.getValues('Sname','out');
NcalibrationSamples=floor(Xout.Nsamples*0.8);
MinputTrain = Minput(1:NcalibrationSamples,:);
VoutputTrain = Voutput(1:NcalibrationSamples);
MinputTest = Minput(NcalibrationSamples+1:end,:);
VoutputTest = Voutput(NcalibrationSamples+1:end);
%% Train ANN
ANN = feedforwardnet(10);
[ANN,tr] = train(ANN,Minput',Voutput');
%% Predict with ANN
out = ANN(MinputTest');
out = out';
figure
scatter(out,VoutputTest);
rsquare(out, VoutputTest);
xlabel('ANN Output')
ylabel('Target')
%% Perform reliability analysis on top of neural network
Nsamples = 1000;
Xrv1_r = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrv2_r = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrv3_r = RandomVariable('Sdistribution','uniform','lowerbound',-pi,'upperbound',pi);
Xrvset_r = RandomVariableSet('Cmembers',{'Xrv1_r','Xrv2_r','Xrv3_r'},'CXrv',{Xrv1_r, Xrv2_r, Xrv3_r});
Xsamples = Xrvset_r.sample('Nsamples', Nsamples);
newOut = ANN(transpose(Xsamples.MsamplesPhysicalSpace));
failureProbability = sum(newOut<=0)/Nsamples;
%%



