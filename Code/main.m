clc; clear;
warning off;
rand('state',0)
randn('state',0)
dataset = 'SearchSnippets'; 
tagRatio = 0.4;
lamda = 1;
tagRatioName = 'Tag040';
% you can change this value to select topic models by yourself
selected_feature = [];
main_HMTT(dataset, tagRatio, tagRatioName, selected_feature,lamda);