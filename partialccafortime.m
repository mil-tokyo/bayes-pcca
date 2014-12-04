function lambda = partialccafortime(X,Y)
    %
    % calculate causality measure from  time series X to Y using Partial CCA
    %
    %   input
    %   X ... dimx * T matrix
    %       time series
    %   Y ... dimy * T matrix
    %       time series
    %
    %   output
    %   lambda ... vector of canonical correlations 

    T = size(X,2);
    dx = size(X,1);
    dy = size(Y,1);

    Yt = Y(:,2:T)';
    Yp = Y(:,1:T-1)';
    Xp = X(:,1:T-1)';

    C = cov([Yt Yp Xp]);
    yt=1:dy;
    yp=dy+1:2*dy;
    xp=2*dy+1:2*dy+dx;

    Yt = Yt - Yp * pinv(C(yp,yp)) * C(yp,yt);
    Xp = Xp - Yp * pinv(C(yp,yp)) * C(yp,xp);
    [A,B,lambda]=canoncorr(Yt,Xp);
end
