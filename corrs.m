function [r,dim] = corrs(Ys,X,p_W,p_tau,p_alpha)
    %
    % calculate correlation and the number of active dimensions
    %
    %   input
    %   Ys ... 2 * 1 cell arrya of dim(i) * datanum matrix
    %       data to calculate correlation
    %   X ... dim * datanum matrix
    %       data to eliminate the effect
    %   p_W ... 2 * 1 cell array of struct
    %       posterior distribution of weight vector
    %   p_tau ... 2 * 1 cell array of struct
    %       posterior distribution of parameter for covariance
    %   p_alpha ... 2 * 1 cell array of struct
    %       posterior distribution of parameter for variance of weight vector
    %
    %   output
    %   r ... vector of correlations for each dimension
    %   dim ... the number of active dimensions

    X = bsxfun(@minus,X,sum(X,2)/size(X,2));
    for i = 1:2
        Ys{i} = bsxfun(@minus,Ys{i},sum(Ys{i},2)/size(Ys{i},2));
    end
    dx = size(X,1);
    dz = size(p_W{1}.mean,2)-dx;
    dim=dz;

    Z1 = inv(eye(dz) + p_tau{1}.mean * p_W{1}.mean(:,dx+1:dx+dz)' * p_W{1}.mean(:,dx+1:dx+dz)) * p_tau{1}.mean * p_W{1}.mean(:,dx+1:dx+dz)' * (Ys{1} - p_W{1}.mean(:,1:dx) * X);
    Z2 = inv(eye(dz) + p_tau{2}.mean * p_W{2}.mean(:,dx+1:dx+dz)' * p_W{2}.mean(:,dx+1:dx+dz)) * p_tau{2}.mean * p_W{2}.mean(:,dx+1:dx+dz)' * (Ys{2} - p_W{2}.mean(:,1:dx) * X);
    r = diag(corr(Z1',Z2'));
    for j = 1:dz
        if(~(p_alpha{1}.means(dx+j) < 50 && p_alpha{2}.means(dx+j) < 50))
            r(j)=0;
            dim=dim-1;
        end
    end
end
