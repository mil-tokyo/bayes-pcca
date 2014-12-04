function [p_W,p_Z,p_tau,p_alpha,L_s] = gspcca(T,X,iter,dimz)
    %
    % calculate bayesian pcca with isotropic noise using variational bayes proposed in section 4.2
    %
    %	input
    %	T ... 2 * 1 cell array of dim(i) * datanum matrix
    %		data to calculate correlation
    %	X ... dim * datanum matrix
    %		data to eliminate the effect
    %	iter ... the number to iteration
    %	dimz ... the dimension of latent variable
    %
    %	output
    %	p_W ... 2 * 1 cell array of struct
    %	    posterior distribution of weight vector
    %	p_Z ... struct
    %	    posterior distribution of latent variable
    %	p_tau ... 2 * 1 cell array of struct
    %	    posterior distribution of parameter for covariance
    %	p_alpha ... 2 * 1 cell array of struct
    %	    posterior distribution of parameter for variance of weight vector
    %	L_s ... vector of variational lower bounds

    %preprocess data
    X = bsxfun(@minus,X,sum(X,2)/size(X,2));
    for i = 1:2
        T{i} = bsxfun(@minus,T{i},sum(T{i},2)/size(T{i},2));
    end
    datavar = zeros(2,1); % datawise variance
    for i = 1:2
        datavar = sum(T{i}.^2)/size(X,2);
    end
    %initialize
    dims = zeros(2,1); %dimension for each input
    dims(1) = size(T{1},1);
    dims(2) = size(T{2},1);
    dimx = size(X,1); %dimension for partial
    dimm = dimx + dimz;	%the number of row of W
    N = size(X,2);	% the number of data

    a0 = 10^-14;	%parameter for prior for alpha,tau
    b0 = 10^-14;

    p_W = cell(2,1);
    for i = 1:2
        p_W{i} = struct('mean',zeros(dims(i),dimm),'cov',eye(dimm),'WtW',zeros(dimm,dimm));
        p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + dims(i) * p_W{i}.cov;
    end

    p_Z = struct('mean',zeros(dimz,N),'cov',eye(dimz),'ZZt',zeros(dimz,dimz),'XZXZt',zeros(dimm,dimm));
    %randomize Zmean
    p_Z.mean = mvnrnd(zeros(dimz,1),eye(dimz),N)';
    p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
    p_Z.XZXZt = [X*X',X*p_Z.mean';p_Z.mean*X',p_Z.ZZt];

    p_tau = cell(2,1);
    for i = 1:2
        p_tau{i} = struct('a',a0*10^3,'b',b0,'mean',10^3);
    end

    %I refer to CCAGFA for initialization.
    p_alpha = cell(2,1);
    for i = 1:2
        p_alpha{i} = struct('a',a0,'bs',b0 * ones(dimm,1),'means',dimm*dims(i)/(datavar(i)-1/p_tau{i}.mean) * ones(dimm,1));
    end

    %option for optimize R
    options = optimoptions('fminunc','Algorithm','quasi-newton','GradObj','on','Display','off');

    %iteration
    for it = 1:iter
        %update W
        for i = 1:2
            p_W{i}.cov = inv(diag(p_alpha{i}.means) + p_tau{i}.mean * p_Z.XZXZt);
            p_W{i}.mean = T{i} * [X;p_Z.mean]' * p_W{i}.cov * p_tau{i}.mean;
            p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + dims(i) * p_W{i}.cov;
        end

        %update Z
        p_Z.cov = eye(dimz);
        for i = 1:2
            p_Z.cov = p_Z.cov + p_tau{i}.mean * p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm);
        end
        p_Z.cov = inv(p_Z.cov);

        p_Z.mean = zeros(dimz,N);
        for i = 1:2
            p_Z.mean = p_Z.mean + p_tau{i}.mean * (p_W{i}.mean(:,dimx+1:dimm)' * T{i} - p_W{i}.WtW(dimx+1:dimm,1:dimx) * X);
        end
        p_Z.mean = p_Z.cov * p_Z.mean;
        p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
        p_Z.XZXZt = [X*X',X*p_Z.mean';p_Z.mean*X',p_Z.ZZt];

        %optimize rotation
        if it > 1
            try
                R = fminunc(@minusLR,eye(dimz),options);
            catch err
                R=eye(dimz);
            end
            p_Z.mean = inv(R) * p_Z.mean;
            p_Z.cov = inv(R) * p_Z.cov * inv(R)';
            for i = 1:2
                p_W{i}.mean = p_W{i}.mean * [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R];
                p_W{i}.cov = [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R'] * p_W{i}.cov * [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R];
            end
            p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
            p_Z.XZXZt = [X*X',X*p_Z.mean';p_Z.mean*X',p_Z.ZZt];
            for i = 1:2
                p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + dims(i) * p_W{i}.cov;
            end
        end

        %update alpha
        for i = 1:2
            p_alpha{i}.a = a0 + dims(i)/2;
            p_alpha{i}.bs = b0 + diag(p_W{i}.WtW)/2;
            p_alpha{i}.means = p_alpha{i}.a./p_alpha{i}.bs;
        end

        %update tau
        for i = 1:2
            p_tau{i}.a = a0 + N * dims(i)/2;
            p_tau{i}.b = b0 + (trace(T{i} * T{i}' - 2 * T{i} * [X;p_Z.mean]' * p_W{i}.mean') + trace(p_W{i}.WtW * p_Z.XZXZt))/2;
            p_tau{i}.mean = p_tau{i}.a/p_tau{i}.b;
        end
        disp(Lq());

        %calculate L_s
        if(it == 1)L_s = Lq();
        else L_s = [L_s;Lq()];
        end
        if(it > 1 && abs((L_s(it)-L_s(it-1))/L_s(it)) < 10^-4) break;end;
    end

    %function for calculating variational lowerbound
    function L = Lq()
        L = 0;
        %calc E[log(ptau)] + H(ptau)
        for i = 1:2
            L = L + a0 * (log(b0)-log(p_tau{i}.b)) + (gammaln(p_tau{i}.a)-gammaln(a0)) + (a0 - p_tau{i}.a) * psi(p_tau{i}.a) + p_tau{i}.mean * (p_tau{i}.b - b0);
        end
        %calc E[log(p(W)] + H(pW)
        for i = 1:2
            L = L + (-0.5) * trace(diag(p_alpha{i}.means) * p_W{i}.WtW);
            L = L + dims(i)/2 * sum(psi(p_alpha{i}.a) - log(p_alpha{i}.bs));
            s = svd(p_W{i}.cov);
            L=L+dims(i)/2*sum(log(s))+dims(i)*dimm/2;
            %L = L + dims(i)/2 * log(det(p_W{i}.cov)) + dims(i) * dimm / 2;
        end
        %calc E[log(p(alpha))] + H(palpha)
        for i = 1:2
            for m = 1:dimm
                L = L + a0 * (log(b0)-log(p_alpha{i}.bs(m))) + (gammaln(p_alpha{i}.a)-gammaln(a0)) + (a0 - p_alpha{i}.a) * psi(p_alpha{i}.a) + p_alpha{i}.means(m) * (p_alpha{i}.bs(m)-b0);
            end
        end
        %calc E[logpz] + H(pz)
        L = L + (-0.5) * trace(p_Z.ZZt) + N * 0.5 * log(det(p_Z.cov)) + 0.5 * N * dimz;
        %calc E[logpt]
        for i = 1:2
            L = L + (-0.5) * p_tau{i}.mean * (trace(T{i} * T{i}' - 2 * T{i} * [X;p_Z.mean]' * p_W{i}.mean') + trace(p_W{i}.WtW * p_Z.XZXZt))/2;
            L = L + N * dims(i)/2 * (psi(p_tau{i}.a)-log(p_tau{i}.b));
            L = L - N * dims(i)/2 * log(2*pi);
        end
    end

    %function used to optimize rotation matrix
    function [f,g] = minusLR(R)
        iR=inv(R);
        %[U,S,V] = svd(R);
        f = 0;
        f = f - trace( iR * p_Z.ZZt * iR') / 2;
        f = f + (dims(1) + dims(2) - N) * sum(log(det(R)));
        for i = 1:2
            for j = 1:dimz
                f = f - dims(i)/2 * log(R(:,j)' * p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R(:,j));
            end
        end
        f = -1 * f;

        if nargout > 1
            g = iR' * iR * p_Z.ZZt * iR' + (dims(1)+dims(2) - N) * iR';
            for i = 1:2
                g = g - dims(i) * bsxfun(@rdivide,p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R,diag(R' * p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R)');
            end
            g = -1 * g;
        end
    end
end
