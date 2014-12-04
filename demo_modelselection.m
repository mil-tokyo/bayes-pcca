%
% sample code for parameter estimation using Partial CCA

d1=50; %dimension of Y1. 5 for low-dimensional data.
d2=50; %dimension of Y2. 4 for low-dimensional data.
dx=5; %dimension of third variable. 3 for low-dimensional data.
dz=5; %dimension of latent variable. 2 for low-dimensional data.
dd=10; %dimension of latent variable used in the parameter estimation. 5 for low-dimensional data.
N=400; %the number of samples
iter=10^5; %the number of iteration for each parameter estimation
repeat=10; % the number of parameter estimation
method='gspcca';% estimation method, can be changed to 'bpcca', 'bic', 'cv'

[Ws,Ys,X]=makepccadata(d1,d2,dx,dz,N);
Wx1=Ws{1}.Wx;
Wx2=Ws{2}.Wx;
W_norm = trace([Wx1;Wx2]'*[Wx1;Wx2]);

if(strcmp(method,'gspcca'))
    [p_W,p_Z,p_tau,p_alpha,L_s] = infer_gs(Ys,X,iter,dd,repeat);
    dW = [p_W{1}.mean(:,1:dx);p_W{2}.mean(:,1:dx)] - [Wx1;Wx2];
    errorW = trace(dW' * dW)/W_norm;
    dim=0;
    for j = 1:(size(p_W{1}.mean,2)-dx)
        if(p_alpha{1}.means(dx+j) < 50 && p_alpha{2}.means(dx+j) < 50)
            dim = dim + 1;
        end
    end
elseif(strcmp(method,'bpcca'))
    [p_W,p_Z,p_phai,p_alpha,L_s] = infer_b(Ys,X,iter,dd,repeat);
    dW = [p_W{1}.mean(:,1:dx);p_W{2}.mean(:,1:dx)] - [Wx1;Wx2];
    errorW = trace(dW' * dW)/W_norm;
    dim=0;
    for j = 1:(size(p_W{1}.mean,2)-dx)
        if(p_alpha{1}.means(dx+j) < 50 && p_alpha{2}.means(dx+j) < 50)
            dim = dim + 1;
        end
    end
elseif(strcmp(method,'bic'))
    [Ws,Psi,p_Z,Lnp,dim] = infer_bic(Ys,X,iter);
    dW = [Ws{1}(:,1:dx);Ws{2}(:,1:dx)] - [Wx1;Wx2];
    errorW = trace(dW' * dW)/W_norm;
elseif(strcmp(method,'cv'))
    [Ws,Lnp,dim] = infer_cv(Ys,X,iter);
    dW = [Ws{1};Ws{2}] - [Wx1;Wx2];
    errorW=trace(dW' * dW)/W_norm;
else 
    fprintf('wrong method name\n');
    return;
end
fprintf('W error: %f, dimension of latent variable: %d\n',errorW,dim);
