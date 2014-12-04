function [Ws,Psi,p_Z,Lnp,dimz]=infer_bic(Ys,X,iter)
    %
    % calculate Probabilistic PCCA using EM algorithm
    % Use Bayesian Information Criterion (BIC) as model selection
    %
    %   Input
    %       Ys ... 2 * 1 cell array of dims(i) * datanum matrix
    %           data to calculate projection
    %       X ... dim * datanum matrix
    %           data to eliminate the effect
    %       iter ... the number for iteration
    %
    %   Output
    %       Ws ... 2 * 1 cell array of dims(i) * (dimx + dimz) matrix
    %           Projection Matrix 
    %               W = [Wx Wz]
    %       Psi ... 2 * 1 cell array of dims(i) * dims(i) matrix
    %           elementwise covariance matrix
    %       p_Z ... struct of mean ... dimz * N cov ... dimz * dimz
    %           Probability Distribution for latent variable
    %       Lnp ... vector of log likelihood for each iteration
    %       dimz ... estimated dimension of latent variable
    [Ws,Psi,p_Z,Lnp] = empcca(Ys,X,1,iter); % estimate model parameters for dimz=1
    d1=size(Ys{1},1);d2=size(Ys{2},1);
    N=size(Ys{1},2);
    dimz = 1;
    for j = 2:min(d1,d2) % estimate model parameters for each dimz
        [Ws_new,Psi_new,p_Z_new,Lnp_new] = empcca(Ys,X,j,iter);
        if(Lnp_new(size(Lnp_new)) - j*(d1+d2)*log(N)/2 > ... % compare BIC
                Lnp(size(Lnp)) - dimz*(d1+d2)*log(N)/2)
            Ws=Ws_new;
            Psi=Psi_new;
            p_Z=p_Z_new;
            Lnp=Lnp_new;
            dimz = j;
        end
    end
end
