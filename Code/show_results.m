function show_results(dataset,alg_showlist,h)
disp('Step.Final-show_results: Hello, here is show_results function!')
%%
if nargin < 3
    h = 1;
end
j = h+1;  % hamming ball radius <= h
colours = 'rbkgcmbk';  
symbols = 'x^os*.dv'; 
linetypes = {'-','-','-','-','-','-','-','-'};
figureSize = [300,200,405,330];
lineWidth = 2;
alg_num = length(alg_showlist);
cateP_list = cell([1,alg_num]);
cateR_list = cell([1,alg_num]);
cateP_test32_list = cell([1,alg_num]);
cateR_test32_list = cell([1,alg_num]);
cateP_test16_list = cell([1,alg_num]);
cateR_test16_list = cell([1,alg_num]);
cateP_test8_list = cell([1,alg_num]);
cateR_test8_list = cell([1,alg_num]);

for g = 1:alg_num
    load(['../Data/',dataset,'/',dataset,'_result_',alg_showlist{g}]);
    clear trueP trueR trueF1;
    cateP_list{g} = cateP;
    cateR_list{g} = cateR;
    clear cateP cateR;
    cateP_test32_list{g} = cateP_test32;
    cateR_test32_list{g} = cateR_test32;
    cateP_test16_list{g} = cateP_test16;
    cateR_test16_list{g} = cateR_test16;
    cateP_test8_list{g} = cateP_test8;
    cateR_test8_list{g} = cateR_test8;
end

%% cate Precision-Recall curve _test32
figure('position',figureSize)
for g = 1:alg_num
    if(g==1)
        figH = plot(cateR_test32_list{g}(:,1), cateP_test32_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
        hold on;
        continue;
    end
    plot(cateR_test32_list{g}(:,1), cateP_test32_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
    hold on;
end
title([dataset,'-32 bits']);
xlabel('Recall');
ylabel('Precision');
axis([0 1 0 1]);
legend(alg_showlist, 'location', 'NorthEast');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0.25,2.5,4,3]);
uistack(figH,'top')
saveas(gcf,['../fig/catePR_test32_',dataset,'.fig']);

%% cate Precision-Recall curve _test16
figure('position',figureSize)
for g = 1:alg_num
    if(g==1)
        figH = plot(cateR_test16_list{g}(:,1), cateP_test16_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
        hold on;
        continue;
    end
    plot(cateR_test16_list{g}(:,1), cateP_test16_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
    hold on;
end
title([dataset,'-16 bits']);
xlabel('Recall');
ylabel('Precision');
axis([0 1 0 1]);
legend(alg_showlist, 'location', 'NorthEast');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0.25,2.5,4,3]);
uistack(figH,'top')
saveas(gcf,['../fig/catePR_test16_',dataset,'.fig']);
%% cate Precision-Recall curve _test8
figure('position',figureSize)
for g = 1:alg_num
    if(g==1)
        figH =  plot(cateR_test8_list{g}(:,1), cateP_test8_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
        hold on;
        continue;
    end
    plot(cateR_test8_list{g}(:,1), cateP_test8_list{g}(:,1), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
    hold on;
end
title([dataset,'-8 bits']);
xlabel('Recall');
ylabel('Precision');
axis([0 1 0 1]);
legend(alg_showlist, 'location', 'NorthEast');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperPosition', [0.25,2.5,4,3]);
set(gcf, 'PaperSize', [8.267716 11.692913]);
uistack(figH,'top')
saveas(gcf,['../fig/catePR_test8_',dataset,'.fig']);
%%
j=4; %fix to [0,1,2,3].
%% cate Precision-Recall curve
figure('position',figureSize)
for g = 1:alg_num
    if(g==1)
        figR = plot(cateR_list{g}(:,j), cateP_list{g}(:,j), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
        hold on;
        continue;
    end
    if(g==2)
        figB = plot(cateR_list{g}(:,j), cateP_list{g}(:,j), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
        hold on;
        continue;
    end
    plot(cateR_list{g}(:,j), cateP_list{g}(:,j), [colours(g),symbols(g),char(linetypes(g))],'linewidth',lineWidth);
    hold on;
end
title('SearchSnippets - (4:4:64 bits)');
xlabel('Recall');
ylabel('Precision');
axis([0 1 0 1]);
legend(alg_showlist, 'location', 'NorthEast');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0.25,2.5,4,3]);
uistack(figB,'top')
uistack(figR,'top')
saveas(gcf,['../fig/catePR_',dataset,'.fig']);
end
