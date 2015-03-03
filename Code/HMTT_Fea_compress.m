function B = HMTT_Fea_compress(test_data, model, parameters, maxbits,selected_feature,T_granularityTopic)
if(~parameters.incOri)
    selected_feature(selected_feature==0) =[];
end
views=length(selected_feature);
[isornot,Id]=ismember(selected_feature,T_granularityTopic);
Id = Id(isornot);
u_hat = parameters.u_weight(Id)/min(parameters.u_weight(Id));
clear isornot Id;
X =[];
if(~parameters.incOri)
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['X=[X;',num2str(parameters.lamda),'*[test_data.instance_00',int2str(selected_feature(i)),']];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['X=[X;',num2str(u_hat(i)),'*[test_data.instance_0',int2str(selected_feature(i)),']];'];
            else
                commandLine = ['X=[X;',num2str(u_hat(i)),'*[test_data.instance_',int2str(selected_feature(i)),']];'];
            end
        end
        eval(commandLine);
    end
else
    for i=1:views
        if(selected_feature(i)==0)
            commandLine = ['X=[X;',num2str(parameters.lamda),'*[test_data.instance_00',int2str(selected_feature(i)),']];'];
        else
            if(selected_feature(i)<100)
                commandLine = ['X=[X;',num2str(u_hat(i-1)),'*[test_data.instance_0',int2str(selected_feature(i)),']];'];
            else
                commandLine = ['X=[X;',num2str(u_hat(i-1)),'*[test_data.instance_',int2str(selected_feature(i)),']];'];
            end
        end
        eval(commandLine);
    end
end
X = X';
X = sparse(X);
Nsamples = size(X,1);

U = zeros(Nsamples,maxbits);
y = zeros(Nsamples,1);

if strcmp(parameters.kernel,'Linear')
    for p = 1:maxbits
        U(:,p) = predict(y,X,model{p});
    end
end
if strcmp(parameters.kernel,'Gaussian')
    for p = 1:maxbits
        U(:,p) = svmpredict(y,X,model{p});
    end
end

B = (U>0.5);

end
