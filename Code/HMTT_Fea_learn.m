function [model, B] = HMTT_Fea_learn(train_data, parameters, maxbits,selected_feature,T_granularityTopic)
options = parameters;

options.k = 27;
options.bNormalized =0;
options.WeightMode ='Cosine';
options.Metric = 'Cosine';
options.NeighborMode = 'Supervised';
options.tagRatio = parameters.tagRatio; 
options.a_weight = parameters.a_weight; 
options.b_weight = parameters.b_weight;
options.gnd = parameters.gndTrain;
options.kernel = parameters.kernel;

if(~parameters.incOri)
    selected_feature(selected_feature==0) =[];
end
views=length(selected_feature);
[isornot,Id]=ismember(selected_feature,T_granularityTopic);
Id = Id(isornot);
u_hat = parameters.u_weight(Id)/min(parameters.u_weight(Id));
% u_hat = ones(1,views);
clear isornot Id;
X =[];
if(~parameters.incOri)
    for i=1:views
    if(selected_feature(i)==0)
        commandLine = ['X=[X;',num2str(parameters.lamda),'*[train_data.instance_00',int2str(selected_feature(i)),']];'];
    else
        if(selected_feature(i)<100)
            commandLine = ['X=[X;',num2str(u_hat(i)),'*[train_data.instance_0',int2str(selected_feature(i)),']];'];
        else
            commandLine = ['X=[X;',num2str(u_hat(i)),'*[train_data.instance_',int2str(selected_feature(i)),']];'];
        end
    end
    eval(commandLine);
    end
else
    for i=1:views
    if(selected_feature(i)==0)
        commandLine = ['X=[X;',num2str(parameters.lamda),'*[train_data.instance_00',int2str(selected_feature(i)),']];'];
    else
        if(selected_feature(i)<100)
            commandLine = ['X=[X;',num2str(u_hat(i-1)),'*[train_data.instance_0',int2str(selected_feature(i)),']];'];
        else
            commandLine = ['X=[X;',num2str(u_hat(i-1)),'*[train_data.instance_',int2str(selected_feature(i)),']];'];
        end
    end
    eval(commandLine);
    end 
end

X = X';
X = sparse(X);
Y = LapEig(X,options,maxbits);

%B = double(Y>0)
Z = repmat(median(Y),size(Y,1),1);
B = double(Y>Z);

model = cell(1,maxbits);
if strcmp(options.kernel,'Linear')
    for p = 1:maxbits
        model(p) = {train(B(:,p),X,'-q')};
    end
end
if strcmp(options.kernel,'Gaussian')
    for p = 1:maxbits
        model(p) = {svmtrain(B(:,p),X,['-q -c 1 -g ',int2str(length(X))])};
    end
end
end
