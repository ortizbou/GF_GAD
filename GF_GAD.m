%%%% Graph Filtering for Graph Anomaly Detection (GF-GAD)

function [AUC1,AUC2,D,U,H]=GF_GAD(A,f,GT)
%  Input: A: Adjacency matrix NxN
%         f: Graph signal matrix Nxp
%         GT: Ground truth labels (anomalouos nodes 0, normal nodes 1) 
%  Output: H: Filter response
%          AUC: Area under the ROC (performance metric)
%  Author: Meiby Ortiz-Bouza
%  Address: Michigan State University, ECE
%  email: ortizbou@msu.edu

%%% Parameters
T=3;  
rho=1;
alpha=0.5;


[~,N]=size(A);
[~,p]=size(f);  


%% Learn filter
An=normadj(A);    % normalized Adjacency
Ln = eye(N)-An;   % normalized Laplacian
[U,d]=eig(full(Ln));
D=diag(d);


anomal=find(GT==0);   % truth anomalous nodes

%%%  t-th shifted input signal as S(t) := U'*D^t*U'*F
for t=1:T
zt{t}=U*d^(t-1)*U'*f;
end

for i=1:N
    for t=1:T
    zn(t,:,i)=zt{t}(i,:);
    end
end



%% Initializations
mu1=rand(N,p);
V=mu1/rho;
h=rand(T,1);
h=h/norm(h);
H=0;
for t=1:T
    Hnew=H+h(t)*diag(D.^(t-1));
    H=Hnew;  
end

thr=alpha/rho;
for n=1:40
    %% ADMM (Z,h,V)
    %%% B^(k+1) update using h^k and V^k
    X=(eye(N)-U*H*U')*f-V;
    B=wthresh(X,'s',thr);
    %%% h^(k+1) update using B^(k+1) and V^k
    E=B-f+V;
    count1=0;
    count2=0;
    SZ=0;
    for t=1:p
    for k=1:N
        for l=1:N
            if (Ln(k,l)~=0)==1
            count1=count1+1;
            ZN1(:,:,count1)=zn(:,t,k)*Ln(k,l)*zn(:,t,l)';
            end
        end
        count1=0;
        SZnew=SZ+sum(ZN1,3);
        SZ=SZnew;
        clear ZN1
        count2=count2+1;
        ZN2(:,:,count2)=zn(:,t,k)*zn(:,t,k)';
        b(:,:,count2)=zn(:,t,k)*E(k,t);
    end
    end
    Y=2*SZ+rho*sum(ZN2,3);
    h_new=-inv(Y)*rho*sum(b,3);
    h_new=h_new/norm(h_new);

    H=0; %% C filter for next iteration
    for t=1:T
        Hnew=H+h_new(t)*diag(D.^(t-1));
        H=Hnew;  
    end

    %%% V^(k+1) update using V^k, Z^(k+1), and c^(k+1)
    V_new=V+rho*(B-(eye(N)-U*H*U')*f);
    if norm(abs(h_new)-abs(h))<10^-3
        break
    end
    h=h_new;
    V=V_new;
end
clear b ZN2


f_tilde=U*H*U'*f;


%%% Anomaly scoring ||f_i-ftilde_i||^2 (reconstruction error)

scores=vecnorm(f'-f_tilde');
mdl = fitglm(scores,double(~GT),'Distribution','binomial','Link','logit');
scores = mdl.Fitted.Probability;
[X1,Y2,thres1,AUC1] = perfcurve(GT,scores,0);


%%% Anomaly scoring based on smoothness

for i=1:N
    s=A(i,:).*vecnorm((f-f(i,:))');
    e0(i)=sum(s);
end

for i=1:N
    s=A(i,:).*vecnorm((f_tilde-f_tilde(i,:))');
    en(i)=sum(s);
end

scores=e0-en;
mdl = fitglm(scores,double(~GT),'Distribution','binomial','Link','logit');
scores = mdl.Fitted.Probability;
[X2,Y2,thres2,AUC2] = perfcurve(GT,scores,0);

end
