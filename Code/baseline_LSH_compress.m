function B = baseline_LSH_compress(X, model,~,~)
Ym = X * model.hs;
B = (Ym > 0);
end
