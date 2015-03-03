function B = baseline_LSI_compress(X, model, ~, ~)

Nsamples = size(X,1);

Y = X * model.V * model.invS;
Z = repmat(model.medU,Nsamples,1);
B = (Y>Z);

end
