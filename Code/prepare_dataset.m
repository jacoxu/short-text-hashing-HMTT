function prepare_dataset(dataset)
disp('Step.1-prepare_dataset: Hello, here is prepare_dataset function!')
%%
if ~exist(['../Data/',dataset,'/',dataset,'_Step1_prepared.mat'], 'file')
    load(['../Data/',dataset,'/',dataset]);
    if (strncmpi(dataset,'20Newsgroups',12))
        %Compute TF-IDF
        [feaTrain,feaTest] = tf_idf(fea(trainIdx,:),fea(testIdx,:));
        feaTrain = feaTrain .* ((1./sum(feaTrain,2))*ones(1,length(feaTrain(1,:))));
        feaTest = feaTest .* ((1./sum(feaTest,2))*ones(1,length(feaTest(1,:))));
    end
    if (strcmp(dataset,'SearchSnippets'))
        %Compute TF-IDF
        [feaTrain,feaTest] = tf_idf(fea(trainIdx,:),fea(testIdx,:));
        feaTrain = feaTrain .* ((1./sum(feaTrain,2))*ones(1,length(feaTrain(1,:))));
        feaTest = feaTest .* ((1./sum(feaTest,2))*ones(1,length(feaTest(1,:))));
    end
    gndTrain = uint8(gnd(trainIdx));
    gndTest  = uint8(gnd(testIdx));
    clear fea gnd trainIdx testIdx topic*;
    save(['../Data/',dataset,'/',dataset,'_Step1_prepared'])
end
clear;
end