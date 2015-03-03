function preCompute_dateset(dataset,option_main)
disp('Step.4-preCompute_dateset: Hello, here is preCompute_dateset function!')
if ~exist(['../Data/',dataset,'/',dataset,'_Step4_preComputed-',option_main.sufName,'.mat'], 'file')
    %%
    load(['../Data/',dataset,'/',dataset,'_Step3_choosedTopicSet']); 
    load(['../Data/',dataset,'/',dataset],'topic*');
    option = option_main;
    sufName = option_main.sufName;
    selected_feature = [0,O_multiTopic];
    for j=1:length(feaTrain(:,1))
        train_data(j).label = gndTrain(j);
        for i=1:length(T_granularityTopic)+1
           if(i==1)
               train_data(j).instance_000 = feaTrain(j,:)'; 
           else
               if(T_granularityTopic(i-1)<100)
                   commandLine = ['train_data(j).instance_0',int2str(T_granularityTopic(i-1)),' = topic0',int2str(T_granularityTopic(i-1)),'_train_theta(j,:)'';'];
                   eval(commandLine);
               else
                   commandLine = ['train_data(j).instance_',int2str(T_granularityTopic(i-1)),' = topic',int2str(T_granularityTopic(i-1)),'_train_theta(j,:)'';'];
                   eval(commandLine);
               end
           end
        end
    end
    for j=1:length(feaTest(:,1))
        test_data(j).label = gndTest(j);
        for i=1:length(T_granularityTopic)+1
           if(i==1)
               test_data(j).instance_000 = feaTest(j,:)'; 
           else
               if(T_granularityTopic(i-1)<100)
                   commandLine = ['test_data(j).instance_0',int2str(T_granularityTopic(i-1)),' = topic0',int2str(T_granularityTopic(i-1)),'_test_theta(j,:)'';'];
                   eval(commandLine);
               else
                   commandLine = ['test_data(j).instance_',int2str(T_granularityTopic(i-1)),' = topic',int2str(T_granularityTopic(i-1)),'_test_theta(j,:)'';'];
                   eval(commandLine);
               end
           end
        end    
    end
    %% construct confidence matrix S by Eq.2
    option.k = 25;
    option.bNormalized =0;
    option.WeightMode ='Cosine';
    option.Metric = 'Cosine';
    option.NeighborMode = 'Supervised';
    option.gnd = gndTrain;
    for i=1:length(T_granularityTopic)+1
        if(i==1)
            disp('preCompute_dateset: Start to construct the original feature similarity matrix!')
            Similarity_Matrix{i} = constructS(feaTrain,option);
        else
            if(T_granularityTopic(i-1)<100)
                disp(['preCompute_dateset: Start to construct the topic0',int2str(T_granularityTopic(i-1)),' similarity matrix!'])
                commandLine = ['Similarity_Matrix{i} = constructS(topic0',int2str(T_granularityTopic(i-1)),'_train_theta,option);'];
                eval(commandLine);
            else
                disp(['preCompute_dateset: Start to construct the topic',int2str(T_granularityTopic(i-1)),' similarity matrix!'])
                commandLine = ['Similarity_Matrix{i} = constructS(topic',int2str(T_granularityTopic(i-1)),'_train_theta,option);'];
                eval(commandLine);
            end
        end
    end
    cateTrainTest = (repmat(gndTrain,1,length(gndTest)) == repmat(gndTest,1,length(gndTrain))');
    clear M_choosedNum O_multiTopic commandLine gnd_subTrain i j knn option randIndex4Weight subTrain_knn *_theta option_main;
    save(['../Data/',dataset,'/',dataset,'_Step4_preComputed-',sufName])
end
clear;
end