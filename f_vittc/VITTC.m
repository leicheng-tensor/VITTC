function [A_completed,Gcore,Lambda,Tau,rse,rank_estimated,Power] = VITTC(A_raw,A_observed,Mask,initmethod,num_of_iter,threshold,varargin)
% ------------------------------------------------------
% Variational Inference for the Tensor Train completion based on the model
% proposed in the following paper
% 
% 1. Xu, L., Cheng, L., Wong, N., & Wu, Y. C. (2020). Learning tensor train representation with automatic rank determination from incomplete noisy data. arXiv preprint arXiv:2010.06564.
% 2. Xu, L., Cheng, L., Wong, N., & Wu, Y. C. (2021, December). Overfitting Avoidance in Tensor Train Factorization and Completion: Prior Analysis and Inference. In 2021 IEEE International Conference on Data Mining (ICDM) (pp. 1439-1444). IEEE.
% 
% ------------------Input------------------
% A_raw
%       -The noisy, incomplete tensor data
% A_observed
%       -The original tensor data
% Mask
%       -Indicating tensor, 1 -> entry observed, 0 -> not observed
% initialmethod
%       -'svdinit' -> use Gaussian random variables to fill in empty entries
%       -'randinit'-> all entries are initialized by Gaussian variables
% num_of_iter
%       -Max iteration for the VI update
% threshold
%       -Slices with average power threshold-times larger than the minimum average slice power will be discarded
% optional: maxrank
%       -max rank for the TT ranks
% ------------------Output------------------
% A_completed
%       -Estimated tensor data
% Gcore
%       -Estimated TT cores
% Lambda
%       -The mean of the Gamma varaibles
% Tau
%       -The estimated noise power
% rse
%       -rse between A_observed & A_completed
% rank_estimated
%       -The estimated rank
% Power
%       -The average power for each TT rank
% 
% ------------------------------------------------------
% XU Le, 2020
% ------------------------------------------------------

Size_A = size(A_observed);
indnorm = 10^(ndims(A_observed)-1)/max( abs(A_observed(:)) );
A_observed = A_observed.*indnorm;
A_Ctemp = 0;

% Initialization
[Lambda,Tau,Gcore] = VITTC_initialize_indpd(A_observed,Mask,initmethod,varargin{:});

% VI iteration
for i = 1:num_of_iter
    [Gcore,V_final,W_final] = update_gcore_indpd_C_var(A_observed,Mask,Gcore,Lambda,Tau);
    Lambda = update_lambda_indpd_C_var(A_observed,Gcore,Lambda,Tau);
    Tau = update_tau_indpd_C_var(A_observed,Mask,Gcore,Lambda,Tau,V_final,W_final);
    [Gcore,Lambda] = rank_reduce_relative_indpd_C_var(Gcore,Lambda,threshold);
    
    A_completed = tt2full(Gcore,Size_A);
    re = sumsqr(A_completed(:)-A_Ctemp(:))/numel(A_completed);
    if re < 1e-9
        break
    end
    A_Ctemp = A_completed;
end

% recover the tensor, and get the rse from the raw data
A_completed = tt2full(Gcore,Size_A);
A_completed = A_completed./indnorm;
rse = sumsqr(A_completed-A_raw)/sumsqr(A_raw);

% mean power of corresponding slices w.r.t. TT ranks
ndims_A  = ndims(A_observed);
Power_L = cell(1,ndims_A);
Power_H = cell(1,ndims_A);
for order = 1:ndims_A
    Power_L{order} = zeros(size(Lambda.mean{order}));
    Power_H{order} = zeros(size(Lambda.mean{order+1}));
    meansqr = Gcore.mean{order}(:)'*Gcore.mean{order}(:)/numel(Gcore.mean{order});
    for r = 1:length(Power_L{order})
        gcoreslice = Gcore.mean{order}(r,:,:);
        Power_L{order}(r) = gcoreslice(:)'*gcoreslice(:)/meansqr/numel(gcoreslice);
    end
    for r = 1:length(Power_H{order})
        gcoreslice = Gcore.mean{order}(:,r,:);
        Power_H{order}(r) = gcoreslice(:)'*gcoreslice(:)/meansqr/numel(gcoreslice);
    end
end
Power = cell(1,ndims_A); Power{1} = 1;
for order = 2:ndims_A
    Power{order} = Power_L{order}+Power_H{order-1};
%     figure; bar(Power{order});
end

%% the final guessed rank
rank_estimated = ones(ndims(A_observed)+1,1);
for order = 1:ndims(A_observed)
    rank_estimated(order) = size(Gcore.mean{order},1);
end

end