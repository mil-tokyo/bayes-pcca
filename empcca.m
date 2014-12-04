function [Ws,Psi, p_Z,Lnp] =  empcca(Ys,X,dimz,iter)
    %
    % calculate Probabilistic PCCA using EM algorithm
    %
    %   Input
    %       Ys ... 2 * 1 cell array of dims(i) * datanum matrix
    %           data to calculate projection
    %       X ... dim * datanum matrix
    %           data to eliminate the effect
    %       dimz ... the dimension of latent variable
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

    %preprocess data
    X = bsxfun(@minus,X,sum(X,2)/size(X,2));
    for i = 1:2
        T{i} = bsxfun(@minus,Ys{i},sum(Ys{i},2)/size(Ys{i},2));
    end

    %initialize
    dims = zeros(2,1); %dimensions for input data
    for i = 1:2
        dims(i) = size(Ys{i},1);
    end
    dimx = size(X,1); %dimension for X
    dimw = dimx + dimz; %dimension for W
    N = size(X,2); % the number of data

    Ws = cell(2,1);
    for i = 1:2
        Ws{i} = mvnrnd(zeros(dims(i),1),eye(dims(i)),dimw)'; %randomly initialize W
    end

    Psi = cell(2,1);
    for i = 1:2
        Psi{i} = wishrnd(eye(dims(i)),dims(i)); %randomly initialize Psi
    end

    p_Z = struct('mean',zeros(dimz,N),'cov',eye(dimz));

    %iteration
    for it = 1:iter
        %update p_Z
        p_Z.cov = eye(dimz);
        for i = 1:2
            p_Z.cov = p_Z.cov + Ws{i}(:,dimx+1:dimw)' * inv(Psi{i}) * Ws{i}(:,dimx+1:dimw);
        end
        p_Z.cov = inv(p_Z.cov);

        p_Z.mean = zeros(dimz,N);
        for i = 1:2
            p_Z.mean = p_Z.mean + Ws{i}(:,dimx+1:dimw)' * inv(Psi{i}) * (Ys{i} - Ws{i}(:,1:dimx) * X);
        end
        p_Z.mean = p_Z.cov * p_Z.mean;

        %update Ws,Psi
        for i = 1:2
            Ws{i} = Ys{i} * [X;p_Z.mean]' * inv([X * X', X * p_Z.mean';p_Z.mean * X',p_Z.mean * p_Z.mean' + N * p_Z.cov]);
            Psi{i} = (Ys{i} * Ys{i}' - Ws{i} * [X;p_Z.mean] * Ys{i}')/N;
        end

        %calculate Lnp
        Lp = 0;
        for i = 1:2
            Wx = Ws{i}(:,1:dimx);
            Wz = Ws{i}(:,dimx+1:dimw);
            C = Psi{i} + Wz * Wz';
            Lp = Lp + (-0.5) * trace(inv(C) * (Ys{i} - Wx * X) * (Ys{i}-Wx * X)') ...
                -0.5 * log(det(C)) * N - dims(i)/2 * log(2*pi) * N;
        end
        if(it==1)
            Lnp = Lp;
        else
            Lnp = [Lnp;Lp];
        end
        if(it>1 && abs((Lnp(it)-Lnp(it-1))/Lnp(it)) < 10^-4) break;end;
    end
end


