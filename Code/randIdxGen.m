function [randIndex4Weight, gnd_subTrain, subTrain_knn] = randIdxGen(dataset, num, knn)
disp('Step.2-randIdxGen: Hello, here is randIdxGen function!')
if ~exist(['../Data/',dataset,'/',dataset,'_Step2_randIdxGen.mat'], 'file')
    load(['../Data/',dataset,'/',dataset,'_Step1_prepared'])
    %normalize TF-IDF
    feaTrain_Unit = normalize(feaTrain);
    %find the total number of classes
    classNum = max(gndTrain);
    %random sample a sub-set X^
    start = 1;
    offset =0;
    for i=1:classNum
        %find the dataset with common label
        curTrainIdx = find(gndTrain==i);
        %randomly shuffles dataset
        randIndex = randperm(length(curTrainIdx));
        %and pick the top X^
        offset = offset + num;
        if(i == 1)
            randIndex4Weight = curTrainIdx(randIndex(1:num));
        else
            randIndex4Weight = [randIndex4Weight; curTrainIdx(randIndex(1:num))];
        end
        gnd_subTrain(start:offset) = i;
        start = start + num;
    end
    clear classNum start offset i test randIndex curTrainIdx;
    %find knn+ and knn- as the step 3 in Algorithm 1
    for i=1:length(gnd_subTrain)
        curClassIdx = randIndex4Weight(gnd_subTrain==gnd_subTrain(i));
        %delete itself
        curClassIdx(curClassIdx == randIndex4Weight(i)) = [];
        dcurClassIdx = randIndex4Weight(gnd_subTrain~=gnd_subTrain(i));
        %measure distance with cosine function
        %compute the distance of the texts sharing any common tags
        dist_nnPlus = feaTrain_Unit(randIndex4Weight(i),:)*feaTrain_Unit(curClassIdx,:)';
        [~, nnPlus_idx] = sort(-dist_nnPlus,2);
        nnPlus_idx = nnPlus_idx(:,1:knn);
        subTrain_knn(i).nnPlusIdx = curClassIdx(nnPlus_idx);
        %compute the distance of the texts not sharing any common tags
        dist_nnMinus = feaTrain_Unit(randIndex4Weight(i),:)*feaTrain_Unit(dcurClassIdx,:)';
        [~, nnMinus_idx] = sort(-dist_nnMinus,2);
        nnMinus_idx = nnMinus_idx(:,1:knn);
        subTrain_knn(i).nnMinusIdx = dcurClassIdx(nnMinus_idx);
    end
    clear curClassIdx dcurClassIdx dist_nnPlus dist_nnMinus dump i nnMinus_idx nnPlus_idx num feaTrain_Unit;
    save(['../Data/',dataset,'/',dataset,'_Step2_randIdxGen'])
end
clear;
end