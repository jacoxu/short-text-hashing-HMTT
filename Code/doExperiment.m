function doExperiment(dataset, alg_list, option_main)
% doExperiment: Learning to Hash
disp('Step.5-doExperiment: Hello, here is doExperiment function!')
%%
if isfield(option_main,'fusionModel')
    fusionModel = option_main.fusionModel;
end

load(['../Data/',dataset,'/',dataset,'_Step4_preComputed-',option_main.sufName]);
if isfield(option_main,'fusionModel')
    option_main.fusionModel=fusionModel;
end
%% debug area for multi-granularity and single-granularity
selected_feature = option_main.selected_feature;
%% 
codeLen = 4:4:64;
%Hamming radius Ϊ 0-3
hammRadius = 0:3;
maxbits = codeLen(end);

%% start to learn hash code and train hash function 
for g = 1:length(alg_list)
    if (strncmpi('HMTT',alg_list{g},4))
        disp(['Step.5-doExperiment: Start to learn hash code by HMTT-relevant algrithm ',alg_list{g},'!'])
        if ~exist(['../Data/',dataset,'/',dataset,'_result_',alg_list{g},'-',option_main.sufName,'.mat'], 'file')
            parameters = option_main;
            parameters.max_iter=10;
            parameters.tol=0.000001;
            %please tune parameters, recommended value: C1=100, C2=10 for SearchSnippets,and C1=100 C2=1 for 20Newsgroups.
            parameters.C1=100;
            parameters.C2=10;
            parameters.u_weight = u_weight;
            parameters.gndTrain = gndTrain;
            if (strcmp(fusionModel,'DecLevel'))
                disp(['Step.5-doExperiment: Start to learn hash code by algrithm ',alg_list{g},' with decison level fusion!'])
                parameters.code_length = maxbits;
                [train_hash_maxbit, train_media, W, alpha, index_corpse]=HMTT_learn(train_data, Similarity_Matrix, selected_feature,T_granularityTopic, parameters);
                test_hash_maxbit = HMTT_compress(test_data, train_media, W, alpha, selected_feature,T_granularityTopic, index_corpse, parameters);
                train_hash{g} = train_hash_maxbit';
                test_hash{g} = test_hash_maxbit';
                clear parameters f_learn f_compress cbTrain_list cbTest_list train_media W;
            elseif(strcmp(fusionModel,'FeaLevel'))
                disp(['Step.5-doExperiment: Start to learn hash code by algrithm ',alg_list{g},' with feature level fusion!'])
                parameters.kernel = 'Linear';
                [model, train_hash_maxbit] = HMTT_Fea_learn(train_data, parameters, maxbits,selected_feature,T_granularityTopic);
                test_hash_maxbit = HMTT_Fea_compress(test_data, model, parameters, maxbits,selected_feature,T_granularityTopic);
                train_hash{g} = train_hash_maxbit;
                test_hash{g} = test_hash_maxbit;
                clear f_learn f_compress metric kernel k model train_hash_maxbit test_hash_maxbit;           
            else
                error(['HMTT_learn: you input a wrong fusionModel ',parameters.fusionModel])
            end
        end
   %% conduct BaseLine
    else
        disp(['Step.5-doExperiment: Start to learn hash code by baseline algrithm ',alg_list{g},'!'])
        if ~exist(['../Data/',dataset,'/',dataset,'_result_',alg_list{g},'.mat'], 'file')
            f_learn = eval(['@baseline_',alg_list{g},'_learn']);
            f_compress = eval(['@baseline_',alg_list{g},'_compress']);
            metric = 'Cosine';
            kernel = 'Linear';
            k = 25;
            [model, train_hash_maxbit] = f_learn(feaTrain, gndTrain, k, metric, kernel, maxbits);
            test_hash_maxbit = f_compress(feaTest, model, kernel, maxbits);
            train_hash{g} = train_hash_maxbit;
            test_hash{g} = test_hash_maxbit;
            clear f_learn f_compress metric kernel k model train_hash_maxbit test_hash_maxbit;
        end
    end
end
clear feaTrain feaTest *_theta;
for g = 1:length(alg_list)
    if (strncmpi('HMTT',alg_list{g},4))
        file_name_str=['../Data/',dataset,'/',dataset,'_result_',alg_list{g},'-',option_main.sufName,'.mat'];
    else
        file_name_str=['../Data/',dataset,'/',dataset,'_result_',alg_list{g},'.mat'];
    end
    if ~exist(file_name_str, 'file')
        m = length(codeLen);
        n = length(hammRadius);
        cateP = zeros(m,n);
        cateR = zeros(m,n);
        %% 
        for i = 1:m
            nbits = codeLen(i);
            disp(['Step.5-doExperiment: Compact hash code to ',int2str(nbits),'!'])
            cbTrain = compactbit(train_hash{g}(:,1:nbits));
            cbTest  = compactbit(test_hash{g}(:,1:nbits)); 
            hammTrainTest  = hammingDist(cbTest,cbTrain)';
            for j = 1:n 
                Ret = (hammTrainTest <= hammRadius(j)+0.00001);
                [cateP(i,j), cateR(i,j)] = evaluate_macro(cateTrainTest, Ret);
            end
        end
       %%
        hammRadius_test_list = {0:32,0:16,0:8};
        for hmk = 1:length(hammRadius_test_list)
            n_test = length(hammRadius_test_list{hmk});
            nbits_test = n_test-1;
            commandLine = ['cateP_test',int2str(nbits_test),' = zeros(n_test);'];
            eval(commandLine);
            commandLine = ['cateR_test',int2str(nbits_test),' = zeros(n_test);'];
            eval(commandLine);
            disp(['Step.5-doExperiment: Start to evalute ',int2str(nbits_test),' hash code for ',alg_list{g},'!'])
            cbTrain_test = compactbit(train_hash{g}(:,1:nbits_test));
            cbTest_test  = compactbit(test_hash{g}(:,1:nbits_test)); 
            hammTrainTest_test  = hammingDist(cbTest_test,cbTrain_test)';
            for j = 1:n_test 
                Ret = (hammTrainTest_test <= hammRadius_test_list{hmk}(j)+0.00001);
                commandLine = ['[cateP_test',int2str(nbits_test),'(j), cateR_test',int2str(nbits_test),'(j)] = evaluate_macro(cateTrainTest, Ret);'];
                eval(commandLine);
            end
        end
        %%
        clear i nbits cbTrain cbTest hammTrainTest j Ret 
        clear hammRadius_test_list hmk n_test nbits_test commandLine cbTrain_test cbTest_test hammTrainTest_test topic*;
        if (strncmpi('HMTT',alg_list{g},4))
            save(['../Data/',dataset,'/',dataset,'_result_',alg_list{g},'-',option_main.sufName]);
        else
            save(['../Data/',dataset,'/',dataset,'_result_',alg_list{g}]);
        end
    end
end
%% 
for g_f = 1:length(alg_list)
    if (strncmpi('HMTT',alg_list{g_f},4))
        load(['../Data/',dataset,'/',dataset,'_result_',alg_list{g_f},'-',option_main.sufName]);
    else
        load(['../Data/',dataset,'/',dataset,'_result_',alg_list{g_f}]);
    end
    clear g train_data test_data Similarity_Matrix;
    if (strncmpi('HMTT',alg_list{g_f},4))
        save(['../Data/',dataset,'/',dataset,'_result_',alg_list{g_f},'-',option_main.sufName]);
    else
        save(['../Data/',dataset,'/',dataset,'_result_',alg_list{g_f}]);
    end
end
clear;
end
