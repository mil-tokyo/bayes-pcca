function [p_W,p_Z,p_phai,p_alpha,L_s] = bpcca(T,X,iter,dimz)
    %
    % calculate bayesian pcca using variational bayes proposed in section 4.1
    %
    %	input
    %	T ...2 * 1 cell array of dim(i) * datanum matrix
    %		data to calculate correlation
    %	X ... dim * datanum matrix
    %		data to eliminate the effect
    %	iter ... the number of iteration
    %	dimz ... the dimension of latent variable
    %
    %	output
    %	p_W ... 2 * 1 cell array of struct
    %	    posterior distribution of Weight vector
    %	p_Z ... struct
    %	    posterior distribution of latent variable
    %	p_phai ... 2 * 1 cell array of struct
    %	    posterior distribution of covariance
    %	p_alpha ... 2 * 1 cell array of struct
    %	    posterior distribution of parameter for variance of weight vector
    %	L_s ... vector of variational lowerbounds

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
    dims = zeros(2,1);
    dims(1) = size(T{1},1);
    dims(2) = size(T{2},1);
    dimx = size(X,1);
    dimm = dimx + dimz;
    N = size(X,2);

    K0 = cell(2,1);
    nu0 = cell(2,1);
    for i = 1:2
        K0{i}=10^(-14)*eye(dims(i));
        nu0{i} = dims(i);
    end
    a0 = 10^-14;%parameter for prior for alpha,tau
    b0 = 10^-14;

    p_W = cell(2,1);
    for i = 1:2
        p_W{i} = struct('mean',zeros(dims(i),dimm),'cov',zeros(dimm,dimm,dims(i)),'WtW',zeros(dimm,dimm));
        p_W{i}.mean = mvnrnd(zeros(dims(i),1),10^3*eye(dims(i)),dimm)';
        p_W{i}.cov = repmat(diag(gamrnd(1,1,dimm,1)),[1,1,dims(i)]);
        p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + sum(p_W{i}.cov,3);
    end

    p_Z = struct('mean',zeros(dimz,N),'cov',eye(dimz),'ZZt',zeros(dimz,dimz),'XZXZt',zeros(dimm,dimm));
    p_Z.mean = mvnrnd(zeros(dimz,1),eye(dimz),N)';
    p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
    p_Z.XZXZt = [X*X',X*p_Z.mean';p_Z.mean*X',p_Z.ZZt];

    p_phai = cell(2,1);
    for i = 1:2
        p_phai{i} = struct('nu',nu0{i},'K',K0{i},'mean',10^3 * eye(dims(i)));
    end

    %I refer to CCAGFA for initialization.
    p_alpha = cell(2,1);
    for i = 1:2
        p_alpha{i} = struct('a',a0,'bs',ones(dimm,1) * b0,'means',dimm*dims(i)/(datavar(i)-10^-3) * ones(dimm,1));
    end

    %option for optimize R
    options = optimoptions('fminunc','Algorithm','quasi-newton','GradObj','on','Display','off');

    %iteration
    for it = 1:iter

        %update W
        for i = 1:2
            for j = 1:dims(i)
                p_W{i}.cov(:,:,j) = inv(diag(p_alpha{i}.means) + p_Z.XZXZt * p_phai{i}.mean(j,j));
            end
        end

        for i = 1:2
            for j = 1:dims(i)
                p_W{i}.mean(j,:) = p_phai{i}.mean(j,:) * T{i} * [X;p_Z.mean]' - (p_phai{i}.mean(j,:)*p_W{i}.mean - p_phai{i}.mean(j,j) * p_W{i}.mean(j,:)) * p_Z.XZXZt;
                p_W{i}.mean(j,:) = p_W{i}.mean(j,:) * p_W{i}.cov(:,:,j);
            end
        end

        for i = 1:2
            p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + sum(p_W{i}.cov,3);
        end

        %update Z
        p_Z.cov=eye(dimz);
        for i = 1:2
            p_Z.cov = p_Z.cov + p_W{i}.mean(:,dimx+1:dimm)' * p_phai{i}.mean * p_W{i}.mean(:,dimx+1:dimm);
            for j = 1:dims(i)
                p_Z.cov = p_Z.cov + p_W{i}.cov(dimx+1:dimm,dimx+1:dimm,j) * p_phai{i}.mean(j,j);
            end
        end
        p_Z.cov = inv(p_Z.cov);

        p_Z.mean = zeros(dimz,N);
        for i = 1:2
            p_Z.mean = p_Z.mean + p_W{i}.mean(:,dimx+1:dimm)' * p_phai{i}.mean * (T{i} - p_W{i}.mean(:,1:dimx) * X);
            for j = 1:dims(i)
                p_Z.mean = p_Z.mean - p_W{i}.cov(dimx+1:dimm,1:dimx,j) * p_phai{i}.mean(j,j) * X;
            end
        end
        p_Z.mean = p_Z.cov * p_Z.mean;
        p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
        p_Z.XZXZt = [X*X',X * p_Z.mean';p_Z.mean * X',p_Z.ZZt];


        %optimize rotation
        if it > 1
            try
                R = fminunc(@minusLR,eye(dimz),options);
            catch err
                R = eye(dimz);
            end
            p_Z.mean = inv(R) * p_Z.mean;
            p_Z.cov = inv(R) * p_Z.cov * inv(R)';
            for i = 1:2
                p_W{i}.mean = p_W{i}.mean * [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R];
                for j = 1:dims(i)
                    p_W{i}.cov(:,:,j) = [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R'] * p_W{i}.cov(:,:,j) * [eye(dimx) zeros(dimx,dimz);zeros(dimz,dimx) R];
                end
            end
            p_Z.ZZt = p_Z.mean * p_Z.mean' + N * p_Z.cov;
            p_Z.XZXZt = [X*X',X*p_Z.mean';p_Z.mean*X',p_Z.ZZt];
            for i = 1:2
                p_W{i}.WtW = p_W{i}.mean' * p_W{i}.mean + sum(p_W{i}.cov,3);
            end
        end

        %update alpha
        for i = 1:2
            p_alpha{i}.a = a0 + dims(i)/2;
            for j = 1:dimm
                p_alpha{i}.bs(j) = b0 + (p_W{i}.WtW(j,j))/2;
            end
            p_alpha{i}.means = p_alpha{i}.a./p_alpha{i}.bs;
        end

        %update phai
        for i = 1:2
            p_phai{i}.nu = nu0{i} + N;
        end

        for i = 1:2
            p_phai{i}.K = K0{i} + T{i} * T{i}' - p_W{i}.mean * [X;p_Z.mean] * T{i}' - T{i} * [X;p_Z.mean]' * p_W{i}.mean' + p_W{i}.mean * p_Z.XZXZt * p_W{i}.mean';
            for j = 1:dims(i)
                p_phai{i}.K(j,j) = p_phai{i}.K(j,j) + trace(p_W{i}.cov(:,:,j) * p_Z.XZXZt);
            end
            p_phai{i}.mean = p_phai{i}.nu * inv(p_phai{i}.K);
        end

        disp(Lq());
        if(it == 1)L_s = Lq();
        else L_s = [L_s;Lq()];
        end
        if(it > 1 && abs((L_s(it)-L_s(it-1))/L_s(it)) < 10^-4) break;end;
        %p_Z.mean = p_Z.mean + 10^-5 * chol(p_Z.cov)' * mvnrnd(zeros(dimz,1),eye(dimz),N)';
    end

    %function for calculating variational lowerbound
function L = Lq()
    L = 0;

    %E[logpW] + H[qW]
    for i = 1:2
        L = L + (-0.5) * trace(diag(p_alpha{i}.means) * p_W{i}.WtW);
        L = L + dims(i)/2 * sum(psi(p_alpha{i}.a) - log(p_alpha{i}.bs));
        for j = 1:dims(i)
            s = svd(p_W{i}.cov(:,:,j));
            L = L + 0.5 * sum(log(s));
            %L = L + 0.5 * log(det(p_W{i}.cov(:,:,j)));
        end
        L = L + dims(i) * dimm / 2;
    end

    %E[logpalpha] + H[qalpha]
    for i = 1:2
        for m = 1:dimm
            L = L + a0 * (log(b0) - log(p_alpha{i}.bs(m))) + (gammaln(p_alpha{i}.a)-gammaln(a0)) + (a0 - p_alpha{i}.a) * psi(p_alpha{i}.a) + p_alpha{i}.means(m) * (p_alpha{i}.bs(m)-b0);
        end
    end

    %E[logpZ] + H[qZ]
    L = L + (-0.5) * trace(p_Z.ZZt) + N * 0.5 * log(det(p_Z.cov)) + 0.5 * N * dimz;

    %E[logpphai] + H[qphai]
    for i = 1:2
        L = L - 0.5 * trace(K0{i} * p_phai{i}.mean);
        for j = 1:dims(i)
            L = L + (nu0{i} - p_phai{i}.nu)/2 * psi((p_phai{i}.nu+1-j)/2);
        end
        L = L - (nu0{i} - p_phai{i}.nu)/2 * log(det(p_phai{i}.K));
        L = L + (nu0{i} * log(K0{i}(1,1))*dims(i)-p_phai{i}.nu*log(det(p_phai{i}.K)))/2;
        for j = 1:dims(i)
            L = L -( gammaln((nu0{i}+1-j)/2) - gammaln((p_phai{i}.nu+1-j)/2));
        end
    end

    %logpt
    for i = 1:2
        tmp = T{i} * T{i}' - p_W{i}.mean * [X;p_Z.mean] * T{i}' - T{i} * [X;p_Z.mean]' * p_W{i}.mean' + p_W{i}.mean * p_Z.XZXZt * p_W{i}.mean';
        for j = 1:dims(i)
            tmp(j,j) = tmp(j,j) + trace(p_W{i}.cov(:,:,j) * p_Z.XZXZt);
        end
        L = L - 0.5 * trace(p_phai{i}.mean * tmp);
        for j = 1:dims(i)
            L = L + N * 0.5 * psi((p_phai{i}.nu+1-j)/2);
        end
        L = L + N * dims(i) * log(2);
        L = L - N * log(det(p_phai{i}.K));
        L = L - N * dims(i)/2 * log(2*pi);
    end
end

%function used to optimize rotation matrix
function [f,g] = minusLR(R)
    iR=inv(R);
    f = 0;
    f = f - trace(iR * p_Z.ZZt * iR')/2;
    f = f + (dims(1) + dims(2) - N) * log(det(R));
    for i = 1:2
        for j = 1:dimz
            f = f - dims(i)/2 * log(R(:,j)' * p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R(:,j));
        end
    end
    f = -1 * f;

    if nargout > 1
        g = iR'*iR*p_Z.ZZt*iR' + (dims(1)+dims(2)-N)*iR';
        for i = 1:2
            g = g - dims(i) * bsxfun(@rdivide,p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R,diag(R' * p_W{i}.WtW(dimx+1:dimm,dimx+1:dimm) * R)');
        end
        g = -1 * g;
    end
end

end

