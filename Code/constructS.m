function [W, elapse] = constructS(fea,options)
if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end
%=================================================
if ~isfield(options,'Metric')
    options.Metric = 'Cosine';
end

switch lower(options.Metric)
    case {lower('Euclidean')}
    case {lower('Cosine')}
        if ~isfield(options,'bNormalized')
            options.bNormalized = 0;
        end
    otherwise
        error('Metric does not exist!');
end

%=================================================
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

switch lower(options.NeighborMode)
    case {lower('KNN')}  %For simplicity, we include the data point itself in the kNN
        if ~isfield(options,'k')
            options.k = 5;
        end
    case {lower('Supervised')}
        if ~isfield(options,'bLDA')
            options.bLDA = 0;
        end
        if options.bLDA
            options.bSelfConnected = 1;
        end
        if ~isfield(options,'k')
            options.k = 0;
        end
        if ~isfield(options,'gnd')
            error('Label(gnd) should be provided under ''Supervised'' NeighborMode!');
        end
        if ~isempty(fea) && length(options.gnd) ~= size(fea,1)
            error('gnd doesn''t match with fea!');
        end
    otherwise
        error('NeighborMode does not exist!');
end

%=================================================

if ~isfield(options,'WeightMode')
    options.WeightMode = 'Binary';
end

bBinary = 0;
switch lower(options.WeightMode)
    case {lower('Binary')}
        bBinary = 1; 
    case {lower('HeatKernel')}
        if ~strcmpi(options.Metric,'Euclidean')
            warning('''HeatKernel'' WeightMode should be used under ''Euclidean'' Metric!');
            options.Metric = 'Euclidean';
        end
        if ~isfield(options,'t')
            options.t = 1;
        end
    case {lower('Cosine')}
        if ~strcmpi(options.Metric,'Cosine')
            warning('''Cosine'' WeightMode should be used under ''Cosine'' Metric!');
            options.Metric = 'Cosine';
        end
        if ~isfield(options,'bNormalized')
            options.bNormalized = 0;
        end
    otherwise
        error('WeightMode does not exist!');
end

%=================================================

if ~isfield(options,'bSelfConnected')
    options.bSelfConnected = 1;
end

%=================================================
tmp_T = cputime;

if isfield(options,'gnd') 
    nSmp = length(options.gnd);
else
    nSmp = size(fea,1);
end
maxM = 62500000; %500M
BlockSize = floor(maxM/(nSmp*3));

if strcmpi(options.NeighborMode,'Supervised')
    Label = unique(options.gnd);
    nLabel = length(Label);
    if options.bLDA
        G = zeros(nSmp,nSmp);
        for idx=1:nLabel
            classIdx = options.gnd==Label(idx);
            G(classIdx,classIdx) = 1/sum(classIdx);
        end
        W = sparse(G);
        elapse = cputime - tmp_T;
        return;
    end

    switch lower(options.WeightMode)
        case {lower('Binary')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D dump;
                    idx = idx(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = 1;
                    idNow = idNow+nSmpClass;
                    clear idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
                G = max(G,G');
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = 1;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(G);
        case {lower('HeatKernel')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                    dump = exp(-dump/(2*options.t^2));
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    D = exp(-D/(2*options.t^2));
                    G(classIdx,classIdx) = D;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            W = sparse(max(G,G'));
        case {lower('Cosine')} 
            if ~options.bNormalized
                [nSmp, nFea] = size(fea);
                if issparse(fea)
                    fea2 = fea';
                    feaNorm = sum(fea2.^2,1).^.5;
                    for i = 1:nSmp
                        fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
                    end
                    fea = fea2';
                    clear fea2;
                else
                    feaNorm = sum(fea.^2,2).^.5;
                    for i = 1:nSmp
                        fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
                    end
                end

            end

            if options.k > 0
                if ~isfield(options,'tagRatio')
                    G = zeros(nSmp*(options.k+1),3);
                    idNow = 0;
                    for i=1:nLabel
                        classIdx = find(options.gnd==Label(i));
                        D = fea(classIdx,:)*fea(classIdx,:)';
                        [dump idx] = sort(-D,2); % sort each row
                        clear D;
                        idx = idx(:,1:options.k+1);
                        dump = -dump(:,1:options.k+1);

                        nSmpClass = length(classIdx)*(options.k+1);
                        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                        G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                        G(idNow+1:nSmpClass+idNow,3) = dump(:);
                        idNow = idNow+nSmpClass;
                        clear dump idx
                    end
                    G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
                else
                    G = zeros(nSmp*(options.k+1),3);
                    idNow = 0;
                    idClass =0;
                    for i=1:nLabel
                        classIdx = find(options.gnd==Label(i));
                        D = zeros(length(classIdx),nSmp);
                        cutIdx = fix(length(classIdx)*options.tagRatio);
                        if (cutIdx<0||cutIdx>length(classIdx))
                            error('constructS: you input a illogical tagRatio');
                        else
                            if((cutIdx~=0)&&(cutIdx ~= length(classIdx)))
                                aIdx = classIdx(1:cutIdx);
                                bIdx = [classIdx((cutIdx+1):length(classIdx));find(options.gnd~=Label(i))];
                                D(aIdx-idClass,aIdx) = options.a_weight*(fea(aIdx,:)*fea(aIdx,:)');
                                D(aIdx-idClass,bIdx) = options.b_weight*(fea(aIdx,:)*fea(bIdx,:)');
                                aIdx = classIdx((cutIdx+1):length(classIdx));
                                bIdx = 1:nSmp;
                                D(aIdx-idClass,bIdx) = options.b_weight*(fea(aIdx,:)*fea(bIdx,:)');
                            elseif(cutIdx==0)
                                aIdx = classIdx((cutIdx+1):length(classIdx));
                                bIdx = 1:nSmp;
                                D(aIdx-idClass,bIdx) = options.b_weight*(fea(aIdx,:)*fea(bIdx,:)');
                            elseif(cutIdx == length(classIdx))
                                aIdx = classIdx(1:cutIdx);
                                bIdx = [classIdx((cutIdx+1):length(classIdx));find(options.gnd~=Label(i))];
                                D(aIdx-idClass,aIdx) = options.a_weight*(fea(aIdx,:)*fea(aIdx,:)'); 
                                D(aIdx-idClass,bIdx) = options.b_weight*(fea(aIdx,:)*fea(bIdx,:)');
                            end
                        end
                        [dump idx] = sort(-D,2); 
                        clear D;
                        idx = idx(:,1:options.k+1);
                        dump = -dump(:,1:options.k+1);
                        nSmpClass = length(classIdx)*(options.k+1);
                        G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                        G(idNow+1:nSmpClass+idNow,2) = idx(:);
                        G(idNow+1:nSmpClass+idNow,3) = dump(:);
                        idNow = idNow+nSmpClass;
                        clear dump idx
                        idClass = idClass + length(classIdx);
                    end
                    G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp); 
                end
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = fea(classIdx,:)*fea(classIdx,:)';
                end
            end

            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end

            W = sparse(max(G,G'));
        otherwise
            error('WeightMode does not exist!');
    end
    elapse = cputime - tmp_T;
    return;
end


if strcmpi(options.NeighborMode,'KNN') && (options.k > 0)
    if strcmpi(options.Metric,'Euclidean')
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = EuDist2(fea(smpIdx,:),fea,0);
                dist = full(dist);
                [dump idx] = sort(dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = dump(:,1:options.k+1);
                if ~bBinary
                    dump = exp(-dump/(2*options.t^2));
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = 1;
                end
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = EuDist2(fea(smpIdx,:),fea,0);
                dist = full(dist);
                [dump idx] = sort(dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = dump(:,1:options.k+1);
                if ~bBinary
                    dump = exp(-dump/(2*options.t^2));
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = 1;
                end
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    else
        if ~options.bNormalized
            [nSmp, nFea] = size(fea);
            if issparse(fea)
                fea2 = fea';
                clear fea;
                for i = 1:nSmp
                    fea2(:,i) = fea2(:,i) ./ max(1e-10,sum(fea2(:,i).^2,1).^.5);
                end
                fea = fea2';
                clear fea2;
            else
                feaNorm = sum(fea.^2,2).^.5;
                for i = 1:nSmp
                    fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
                end
            end
        end
        
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                [dump idx] = sort(-dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = -dump(:,1:options.k+1);

                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                [dump idx] = sort(-dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = -dump(:,1:options.k+1);

                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    end
    
    if strcmpi(options.WeightMode,'Binary')
        W(find(W)) = 1;
    end
    
    if isfield(options,'bSemiSupervised') && options.bSemiSupervised
        tmpgnd = options.gnd(options.semiSplit);
        
        Label = unique(tmpgnd);
        nLabel = length(Label);
        G = zeros(sum(options.semiSplit),sum(options.semiSplit));
        for idx=1:nLabel
            classIdx = tmpgnd==Label(idx);
            G(classIdx,classIdx) = 1;
        end
        Wsup = sparse(G);
        if ~isfield(options,'SameCategoryWeight')
            options.SameCategoryWeight = 1;
        end
        W(options.semiSplit,options.semiSplit) = (Wsup>0)*options.SameCategoryWeight;
    end
    
    if ~options.bSelfConnected
        for i=1:size(W,1)
            W(i,i) = 0;
        end
    end

    W = max(W,W');
    
    elapse = cputime - tmp_T;
    return;
end

% Complete Graph
if strcmpi(options.Metric,'Euclidean')
    W = EuDist2(fea,[],0);
    W = exp(-W/(2*options.t^2));
else
    if ~options.bNormalized
        [nSmp, nFea] = size(fea);
        if issparse(fea)
            fea2 = fea';
            feaNorm = sum(fea2.^2,1).^.5;
            for i = 1:nSmp
                fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
            end
            fea = fea2';
            clear fea2;
        else
            feaNorm = sum(fea.^2,2).^.5;
            for i = 1:nSmp
                fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
            end
        end
    end
    
    W = full(fea*fea');
end

if ~options.bSelfConnected
    for i=1:size(W,1)
        W(i,i) = 0;
    end
end
W = max(W,W');
elapse = cputime - tmp_T;


