function [X,Y,Ys,Yp] = maketimeseries(T,r)
    %
    % make time series data such that
    % x_t = 0.5 * x_{t-1} + e_xt
    % y_t = 0.5 * y_{t-1} + W * x_{t-1} + e_yt
    %
    %	input
    %	T ... the length of time series
    %	r ... noise variance
    %
    %	output
    %	X ... [x_1 x_2 ... x_T]
    %	Y ... [y_1 y_2 ... y_T]
    %	Ys ... input for partial CCA
    %	Yp ... third variable for partial CCA


    W=[mvnrnd(zeros(2,1), 0.5*eye(2),20) zeros(20,18)];

    exs = mvnrnd(zeros(20,1),1*eye(20),T)';
    eys = mvnrnd(zeros(20,1),1*eye(20),T)';
    A=0.5;
    B=0.5;
    X = exs;
    Y = eys;
    for t = 2:T
        X(:,t)=A*X(:,t-1)+exs(:,t);
        Y(:,t)=B*Y(:,t-1)+W*X(:,t-1)+eys(:,t);
    end
    Y=repmat(Y,[2,1]) + mvnrnd(zeros(40,1),eye(40)*r,T)';

    Xp = X(:,1:T-1);
    Yt = Y(:,2:T);
    Yp = Y(:,1:T-1);
    Ys=cell(2,1);
    Ys{1} = Yt;
    Ys{2} = Xp;
end
