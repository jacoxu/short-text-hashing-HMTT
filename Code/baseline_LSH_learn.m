function [model, B] = baseline_LSH_learn(X,~,~,~,~, maxbits)
hs = normrnd(0, 1, size(X, 2), maxbits);
model.hs = hs;
Ym = X * model.hs;
B = (Ym > 0);
end
