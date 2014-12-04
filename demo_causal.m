%
% sample code for calculating causality measure using Partial CCA

T=400;% the length of time series
r=0.1;% noise variance
method='gspcca';% estimation method. You can change to 'pcca'

[X,Y,Ys,Yp]=maketimeseries(T,r);

if(strcmp(method,'gspcca'))
    [p_W,p_Z,p_tau,p_alpha,L_s] = infer_gs(Ys,Yp,10^4,5,10);
    [lambda,dim] = corrs(Ys,Yp,p_W,p_tau,p_alpha);
elseif(strcmp(method,'pcca'))
    lambda = partialccafortime(X,Y);
else 
    fprintf('wrong method name');
    return;
end

measure= -sum(log(1-lambda.^2))/2;
fprintf('causality measure: %f\n',measure);
