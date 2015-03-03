function precision = evaluate_labels(train_hash, test_hash, train_data, test_data, parameters)

[neighbors] = kNearestNeighbors(train_hash', test_hash', parameters.neighboors); 
    
train_labelset= [train_data.label];
test_labelset = [test_data.label];
    precision_corpse=[];
for i=1:size(neighbors,1)
    train_label_temp = train_labelset(neighbors(i,:));
    true_label_temp = test_data(i).label;
    ww=find(train_label_temp == true_label_temp);
    precision_corpse=[precision_corpse, (length(ww)+1)/(length(train_label_temp)+1)];
    
end
    precision = mean(precision_corpse);
