function [p_W,p_Z,p_phai,p_alpha,L_s]=infer_b(Ys,X,iter,dd,repeat)
    %
    % calculate bayesian pcca using variational bayes proposed in section 4.1
    % repeat with random initialization and choose the best one.
    %
    %	input
    %	Ys ...2 * 1 cell array of dim(i) * datanum matrix
    %		data to calculate correlation
    %	X ... dim * datanum matrix
    %		data to eliminate the effect
    %	iter ... the number of iteration
    %	dd ... the dimension of latent variable
    %	repeat ... the number of estimation with random initialization
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

    [p_W,p_Z,p_phai,p_alpha,L_s] = bpcca(Ys,X,iter,dd); % first estimation
    for j = 2:repeat
        [p_W_new,p_Z_new,p_phai_new,p_alpha_new,L_s_new] = bpcca(Ys,X,iter,dd); % estimation
        if(L_s_new(size(L_s_new)) > L_s(size(L_s))) %compare variational lowerbounds
            p_W = p_W_new;
            p_Z = p_Z_new;
            p_phai = p_phai_new;
            p_alpha = p_alpha_new;
            L_s = L_s_new;
        end
    end
end
