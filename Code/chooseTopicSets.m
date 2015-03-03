function [O_multiTopic] = chooseTopicSets(dataset, T_granularityTopic, M_choosedNum)
disp('Step.3-chooseTopicSets: Hello, here is chooseTopicSet function!')
if ~exist(['../Data/',dataset,'/',dataset,'_Step3_choosedTopicSet.mat'], 'file')
    load(['../Data/',dataset,'/',dataset,'_Step2_randIdxGen']);
    load(['../Data/',dataset,'/',dataset],'topic*');
    % Initialize weight vector u
    u_weight = zeros(length(T_granularityTopic),1);
    for i=1:length(gnd_subTrain)
        for j = 1:length(T_granularityTopic)
            if(T_granularityTopic(j) < 100)
                topic_theta = ['topic0',int2str(T_granularityTopic(j)),'_train_theta'];
            else
                topic_theta = ['topic',int2str(T_granularityTopic(j)),'_train_theta'];
            end
            dist_nnMinus = 0;
            dist_nnPlus = 0;
            for k = 1: knn
                commandLine = ['dist_nnMinus = dist_nnMinus + kldiv(',topic_theta,'(randIndex4Weight(i),:),',topic_theta,'(subTrain_knn(i).nnMinusIdx(k),:),''sym'');'];
                eval(commandLine);
                commandLine = ['dist_nnPlus = dist_nnPlus + kldiv(',topic_theta,'(randIndex4Weight(i),:),',topic_theta,'(subTrain_knn(i).nnPlusIdx(k),:),''sym'');'];
                eval(commandLine);
            end
            u_weight(j) = u_weight(j) + dist_nnMinus/knn - dist_nnPlus/knn;
        end
    end
    clear k topic_theta dist_nnMinus dist_nnPlus; 
    O_multiTopic = [];
    T_multiTopic = T_granularityTopic;
    maxScoreIdx = find(u_weight == max(u_weight));
    num = 1;
    %find the optimial topic sets
    O_multiTopic(num) = T_multiTopic(maxScoreIdx);
    disp(['chooseTopicSets: topic',int2str(T_multiTopic(maxScoreIdx)),' is selected'])
    T_multiTopic(maxScoreIdx) =[];
    while (length(O_multiTopic) < M_choosedNum)
        [~,part1Idx] = ismember(T_multiTopic,T_granularityTopic);
        part1Score = u_weight(part1Idx);
        num = num +1;
        maxScoreIdx = find(part1Score == max(part1Score));
        O_multiTopic(num) = T_multiTopic(maxScoreIdx);
        disp(['chooseTopicSets: topic',int2str(T_multiTopic(maxScoreIdx)),' is selected'])
        T_multiTopic(maxScoreIdx) =[];
    end
    clear T_multiTopic maxScoreIdx num i j p q ScoresT commandLine part1Idx part1Score part2Score distTiTj *_phi topic*;
    save(['../Data/',dataset,'/',dataset,'_Step3_choosedTopicSet'])
else
    load(['../Data/',dataset,'/',dataset,'_Step3_choosedTopicSet'])
end
end