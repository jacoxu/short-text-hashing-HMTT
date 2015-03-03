function main_HMTT(dataset, tagRatio, sufName, selected_feature,lamda)
disp('version: 2014/12/01-19:41, HMTT-demo')
%% configure
option_main.lamda = lamda;
option_main.tagRatio = tagRatio;
option_main.a_weight = 1;
option_main.b_weight = 0.1;
option_main.sufName = sufName;
option_main.incOri = false;
clear tagRatio sufName lamda;
%% Step1
prepare_dataset(dataset);
%% Step2 - Step3: Algorithm 1.
% Step2
samplingSize = 100;
knn4Weight = 10;
randIdxGen(dataset, samplingSize, knn4Weight);
% Step3
T_granularityTopic = [010, 030, 050, 070, 090, 120, 150];
M_choosedNum = 3;
if ~isempty(selected_feature)
   option_main.selected_feature = selected_feature;
else
   option_main.selected_feature = chooseTopicSets(dataset, T_granularityTopic, M_choosedNum);
end
clear selected_feature;
%% Step4
preCompute_dateset(dataset,option_main);
%% Step5
% Algorithem 2.
alg_list={'HMTT-Fea'};
option_main.fusionModel = 'FeaLevel';
doExperiment(dataset,alg_list,option_main);
% Algorithem 3.
alg_list={'HMTT-Dec'};
option_main.fusionModel = 'DecLevel';
doExperiment(dataset,alg_list,option_main);
%% partially baseLine methods
alg_list={'LSI', 'LSH'};
doExperiment(dataset,alg_list,option_main);
%% StepFinal
alg_showlist={['HMTT-Fea-',option_main.sufName], ['HMTT-Dec-',option_main.sufName],'LSI', 'LSH'};
show_results(dataset,alg_showlist)
end