function test_hash=HMTT_compress(test_data, ~, W, alpha, selected_feature,T_granularityTopic, index_corpse,parameters)
if(~parameters.incOri)
    selected_feature(selected_feature==0) =[];
end
views=length(selected_feature);
%% 
if (strcmp(parameters.fusionModel,'DecLevel'))
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['all_instance_test{i}=[test_data.instance_00',int2str(selected_feature(i)),'];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['all_instance_test{i}=[test_data.instance_0',int2str(selected_feature(i)),'];'];
            else
                commandLine = ['all_instance_test{i}=[test_data.instance_',int2str(selected_feature(i)),'];'];
            end
        end
        eval(commandLine);
    end
    %%
    for i=1:views
        temp_instance = all_instance_test{i};
        temp_instance_2=zeros(index_corpse(end), size(temp_instance,2));
        temp_instance_2(index_corpse((2*(i-1)+1)):index_corpse((2*i)), :) = temp_instance;
        all_instance_test(i)={sparse(temp_instance_2)};
    end
    output_test=[];
    tic
    for i=1:views
       output_test=[output_test, {W'*all_instance_test{i}}];
    end
    output_test_final=zeros(parameters.code_length,length(test_data));
    for i=1:views
         output_test_final = output_test_final+alpha(i)*output_test{i};
    end
    clear output_test all_instance_test;
elseif(strcmp(parameters.fusionModel,'FeaLevel'))
    [isornot,Id]=ismember(selected_feature,T_granularityTopic);
    Id = Id(isornot);
    u_hat = parameters.u_weight(Id)/min(parameters.u_weight(Id));
    clear isornot Id;
    instance_subtract_test =[];
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['instance_subtract_test=[instance_subtract_test;',num2str(parameters.lamda),'*[test_data.instance_00',int2str(selected_feature(i)),']];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['instance_subtract_test=[instance_subtract_test;',num2str(u_hat(i)),'*[test_data.instance_0',int2str(selected_feature(i)),']];'];
            else
                commandLine = ['instance_subtract_test=[instance_subtract_test;',num2str(u_hat(i)),'*[test_data.instance_',int2str(selected_feature(i)),']];'];
            end
        end
        eval(commandLine);
    end
    output_test_final = W'*instance_subtract_test;
else
    error(['SSHMT_compress: you input a wrong fusionModel ',parameters.fusionModel])
end

%% get the output for the test set
test_hash = [];
for i=1:size(output_test_final,1)
    temp=output_test_final(i,:);
    test_hash = [test_hash;(temp>0.5)];
end
test_time =toc;
disp(['HMTT_compress: cost time ',int2str(test_time)]);
end