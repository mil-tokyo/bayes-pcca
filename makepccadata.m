function [Ws,Ys,X]= makepccadata(d1,d2,dx,dz,N)
    %
    % make pcca data from probabilistic generative model
    %
    %   input
    %   d1 ... dimension for Y1
    %   d2 ... dimension for Y2
    %   dx ... dimension for third variable
    %   dz ... dimension for latent variable
    %   N ... the number of samples
    %
    %   output
    %   Ws ... 2 * 1 cell array of struct
    %       projection matrix
    %   Ys ... 2 * 1 cell array of di * N matrix
    %       input data for Partial CCA
    %   X ... dx * N matrix
    %       third variable

    Wx1 = mvnrnd(zeros(d1,1),eye(d1),dx)' ;
    Wx2 = mvnrnd(zeros(d2,1),eye(d2),dx)' ;

    Wz1 = mvnrnd(zeros(d1,1),eye(d1),dz)' ;
    Wz2 = mvnrnd(zeros(d2,1),eye(d2),dz)' ;

    W1 = mvnrnd(zeros(d1,1),eye(d1),floor(dz/2)) ;
    V1 = W1' * W1 + eye(d1);
    W2 = mvnrnd(zeros(d2,1),eye(d2),floor(dz/2)) ;
    V2 = W2' * W2 + eye(d2);


    X = mvnrnd(zeros(dx,1),eye(dx),N)';
    Z = mvnrnd(zeros(dz,1),eye(dz),N)';

    Ys=cell(2,1);
    Ys{1} = Wx1 * X + Wz1 * Z + mvnrnd(zeros(d1,1),V1,N)';
    Ys{2} = Wx2 * X + Wz2 * Z + mvnrnd(zeros(d2,1),V2,N)';

    Ws=cell(2,1);
    Ws{1}=struct('Wx',Wx1,'Wz',Wz1);
    Ws{2}=struct('Wx',Wx2,'Wz',Wz2);

end
