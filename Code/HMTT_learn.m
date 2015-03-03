function [train_hash, train_media, W_temp, alpha, index_corpse]=HMTT_learn(train_data, Similarity_Matrix, selected_feature,T_granularityTopic, parameters)
if(~parameters.incOri)
    selected_feature(selected_feature==0) =[];
end
views=length(selected_feature);
n_docs = length(train_data);
if (strcmp(parameters.fusionModel,'DecLevel'))
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['all_instance{i}=[train_data.instance_00',int2str(selected_feature(i)),'];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['all_instance{i}=[train_data.instance_0',int2str(selected_feature(i)),'];'];
            else
                commandLine = ['all_instance{i}=[train_data.instance_',int2str(selected_feature(i)),'];'];
            end
        end
        eval(commandLine);
    end
elseif(strcmp(parameters.fusionModel,'FeaLevel'))
    [isornot,Id]=ismember(selected_feature,T_granularityTopic);
    Id = Id(isornot);
    u_hat = parameters.u_weight(Id)/min(parameters.u_weight(Id));
    clear isornot Id;
    instance_subtract =[];
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['instance_subtract=[instance_subtract;',num2str(parameters.lamda),'*[train_data.instance_00',int2str(selected_feature(i)),']];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['instance_subtract=[instance_subtract;',num2str(u_hat(i)),'*[train_data.instance_0',int2str(selected_feature(i)),']];'];
            else
                commandLine = ['instance_subtract=[instance_subtract;',num2str(u_hat(i)),'*[train_data.instance_',int2str(selected_feature(i)),']];'];
            end
        end
        eval(commandLine);
    end
else
    error(['HMTT_learn: you input a wrong fusionModel ',parameters.fusionModel])
end
%%
if (strcmp(parameters.fusionModel,'DecLevel'))
    %Calculate the Laplacian matrix
    L_corpse=[];
    for i=1:views
        D_temp=sum(Similarity_Matrix{i});
        ww=find(D_temp==0);
        D_temp(ww)=1;
        L_corpse=[L_corpse,{(diag(D_temp.^(-1/2))*(diag(D_temp)-Similarity_Matrix{i})*diag(D_temp.^(-1/2)))}];
    end
    clear Similarity_Matrix D_temp ww;
elseif(strcmp(parameters.fusionModel,'FeaLevel'))
     disp('HMTT_learn: Start to construct the multi-feature fusion similarity matrix!')
        option.k = 25;
        option.bNormalized =0;
        option.WeightMode ='Cosine';
        option.Metric = 'Cosine';
        option.NeighborMode = 'Supervised';
        option.tagRatio = parameters.tagRatio; 
        option.a_weight = parameters.a_weight; 
        option.b_weight = parameters.b_weight; 
        option.gnd = parameters.gndTrain;
        Similarity_Matrix = constructS(instance_subtract',option);
        clear option;
        D_temp=sum(Similarity_Matrix);
        ww=find(D_temp==0);
        D_temp(ww)=1;
        L_this=diag(D_temp.^(-1/2))*(diag(D_temp)-Similarity_Matrix)*diag(D_temp.^(-1/2));
else
    error(['HMTT_learn: you input a wrong fusionModel ',parameters.fusionModel])
end

tic; 
%%
index_corpse=1;
if (strcmp(parameters.fusionModel,'DecLevel'))
    for i=1:views
        dimension=size(all_instance{i},1);
        index_corpse=[index_corpse,index_corpse(end)+dimension-1, index_corpse(end)+dimension];
    end
elseif(strcmp(parameters.fusionModel,'FeaLevel'))
    dimension=size(instance_subtract,1);
    index_corpse=[1,dimension, dimension+1];
else
    error(['HMTT_learn: you input a wrong fusionModel ',parameters.fusionModel])
end
index_corpse=index_corpse(1:end-1);
clear dimension;
total_dimension=index_corpse(end);
alpha=ones(views,1);
alpha=alpha/sum(alpha);
if (strcmp(parameters.fusionModel,'DecLevel'))
    for i=1:views
        temp_instance = all_instance{i};
        temp_instance_2=sparse(zeros(total_dimension, n_docs));
        temp_instance_2(index_corpse((2*(i-1)+1)):index_corpse((2*i)), :) = temp_instance;
        all_instance(i)={temp_instance_2};
    end
    clear temp_instance temp_instance_2;
    W_temp = zeros(total_dimension,parameters.code_length);
    for i=1:parameters.max_iter 
        disp(['HMTT_learn: iteration: ***',int2str(i),' for decision fustion training']);
        alpha_old=alpha;
        L_this=sparse(zeros(n_docs,n_docs));
        for ii=1:views
            L_this=L_this+parameters.C1*0.5*L_corpse{ii};
        end
          instance_subtract=sparse(zeros(total_dimension, length(train_data)));
        for jj=1:views
            temp_instance = all_instance{jj};
            instance_subtract = instance_subtract+alpha(jj)*temp_instance;
        end
        if (total_dimension < 5000)
            instance_subtract = full(instance_subtract);
            L_this = full(L_this);
        end
        clear temp_instance;
        Q=(parameters.C2*instance_subtract*instance_subtract'+eye(total_dimension))\(parameters.C2*instance_subtract);
        QTX=Q'*instance_subtract;
        second_term=parameters.C2*(sparse(eye(n_docs))-QTX-QTX'+QTX*QTX')+Q'*Q;
        L_decompose= L_this + second_term;
        L_decompose=(L_decompose+L_decompose')/2;
        clear QTX
        options.disp=0;
        [eigVecs,eigVals]=eigs(L_decompose, parameters.code_length,  'sa', options);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        eigVals=abs(diag(eigVals));
        [~,bb]=sort(eigVals,'ascend');
        eigVecs=(eigVecs(:,bb));
        Y=eigVecs(:,1:(parameters.code_length));
        W_temp=Q*Y;
        Y=Y';
        %%%%%%%%%%%%%get output
        output=[];
        for ii=1:views
            output_temp = W_temp'*all_instance{ii};
            output=[output, {output_temp}];
        end %Output = W'*X
        output_Y=[];
        for ii=1:views
            output_Y=[output_Y, reshape(output{ii}, size(output{ii}, 1)*size(output{ii},2), 1)];
        end
       %%%%%%%%%%%%%%%%%%%%%%%%% 
        constant2=-2* parameters.C2*reshape(Y, size(Y,1)*size(Y,2),1)'*output_Y;
        constant2=constant2';
        H=parameters.C2*output_Y'*output_Y;
        %optimize alpha via quadprog
        alpha = quadprog(2*H,  constant2, [], [], ones(1, views), 1, zeros(views,1), ones(views,1),alpha_old);
        if sum((alpha-alpha_old).^2)<parameters.tol
            break;
        end
    end
    alpha=alpha_old; 
elseif(strcmp(parameters.fusionModel,'FeaLevel'))
    disp('HMTT_learn: the X~ has obtained by the first sub step!')
    W_temp = ones(total_dimension,parameters.code_length);
    for i=1:parameters.max_iter
        %calculate \sum \lambda_s X L X
        disp(['HMTT_learn: iteration: ***',int2str(i),' for feature fustion training']);
        W_old = W_temp;
        L_this=parameters.C1*0.5*L_this;
        if (total_dimension < 5000)
            instance_subtract = full(instance_subtract);
            L_this = full(L_this);
        end
        Q=(parameters.C2*instance_subtract*instance_subtract'+eye(total_dimension))\(parameters.C2*instance_subtract);
        QTX=Q'*instance_subtract;
        second_term=parameters.C2*(sparse(eye(n_docs))-QTX-QTX'+QTX*QTX')+Q'*Q;
        L_decompose= L_this + second_term;
        L_decompose=(L_decompose+L_decompose')/2;
        clear QTX
        options.disp=0;
        [eigVecs,eigVals]=eigs(L_decompose, parameters.code_length,  'sa', options);
        eigVals=abs(diag(eigVals));
        [~,bb]=sort(eigVals,'ascend');
        eigVecs=(eigVecs(:,bb));
        Y=eigVecs(:,1:(parameters.code_length));
        W_temp=Q*Y;
        Y=Y'; 
        if norm(W_temp-W_old,2)<parameters.tol
            break;
        end
    end
    clear W_old;
else
    error(['HMTT_learn: you input a wrong fusionModel ',parameters.fusionModel])
end

%learn W for test examples
%% W_temp is the classifier. 
train_hash=[];

if length(unique(Y(1,:)))~=2
    train_media=[];
    for i=1:size(Y,1)
        temp=Y(i,:);
    train_hash = [train_hash;(temp>median(temp))];
    train_media = [train_media; median(temp)];
    end
else
    train_hash=Y;
end
W_temp=Q*train_hash';%W=Q*Y
clear Q;
train_time=toc;
disp(['HMTT_learn: cost time ',int2str(train_time)]);
end