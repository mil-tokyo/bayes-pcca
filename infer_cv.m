function [Ws,Lnp,dimz]=infer_cv(Ys,X,iter)
    %
    % calculate Probabilistic PCCA using EM algorithm
    % Use 5-fold cross vaidation (CV) as model selection
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
    %       Lnp ... sum of log likelihood for test data
    %       dimz ... estimated dimension of latent variable
    d1=size(Ys{1},1);d2=size(Ys{2},1);dims=[d1,d2];dx=size(X,1);
    N=size(Ys{1},2);
    Lnps = zeros(min(d1,d2),1);
    Wxs = cell(min(d1,d2),1);
    for j = 1:min(d1,d2)
        Wxs{j} = cell(2,1);
        for m = 1:2
            Wxs{j}{m}=zeros(dims(m),dx);
        end
    end
    for j = 1:min(d1,d2)
        for k = 1:5
            indT = N/5*(k-1)+1:N/5*k; % for test
            indV = setdiff(1:N,indT); % for train
            TYs=cell(2,1);
            VYs=cell(2,1);
            for m = 1:2
                TYs{m} = Ys{m}(:,indT);
                VYs{m} = Ys{m}(:,indV);
            end
            TX = X(:,indT);
            VX = X(:,indV);
            [Ws,Psi,p_Z,Lnp] = empcca(VYs,VX,j,iter); % estimate parameter for train data
            for m = 1:2
                Wx = Ws{m}(:,1:dx);
                Wz = Ws{m}(:,dx+1:dx+j);
                C = Psi{m} + Wz * Wz';
                Lnps(j) = Lnps(j) + (-0.5) * trace(inv(C) * (TYs{m} - Wx * TX) * (TYs{m}-Wx * TX)') ... % add log likelihood for test data
                    -0.5 * log(det(C)) * N/5 - dims(m)/2 * log(2*pi) * N/5;
                Wxs{j}{m}=Wxs{j}{m}+Wx/5;
            end
        end
    end
    [B,IX]=sort(Lnps,'descend');
    dimz=IX(1);
    Ws=Wxs{dimz};
    Lnp=Lnps(dimz)
end
