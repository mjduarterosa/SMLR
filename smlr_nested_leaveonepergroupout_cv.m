function [Pall,P,lams_used,acc_fold,wsmlr_used,log_params1,log_params2,opt] = smlr_nested_leaveonepergroupout_cv(X,Y,opt)

% -------------------------------------------------------------------------
% Sparse Multinomial Logistic Regression - LOSGO Nested CV
%
% M. J. Rosa, Centre for Neuroimaging Sciences, King's College London
% Based on toolbox by A. Marquand
% -------------------------------------------------------------------------

[N,M]    = size(Y);
N        = N/M;
Nscans_s = 1;

% lam2 -> inf : UST
% lam2 -> 0   : LASSO
% lam1 -> 0   : L2 only

log_params1 = [-5; -4; -3; -2; -1; 0; 1; 2; 3; 4; 5];
% log_params1 = [-5; -4.5; -4; -3.5; -3; -2.5; -2; -1.5; -1; -0.5; 0; 0.5; 1; 1.5; 2];
% log_params2 = [-5; -4.5; -4; -3.5; -3; -2.5; -2; -1.5; -1; -0.5; 0; 0.5; 1; 1.5; 2];
log_params2 = [-5; -4; -3; -2; -1; 0; 1; 2; 3; 4; 5];

prange1     = 10.^(log_params1);
prange2     = 10.^(log_params2);

try opt.estimate_all_w; catch, opt.estimate_all_w = true; end
% try opt.estimate_all_w; catch, opt.estimate_all_w = false; end
try opt.verbose;        catch, opt.verbose = true; end
try opt.tol;            catch, opt.tol = 0.001; end

% Outer LOO-CV loop
%%%%%%%%%%%%%%%%%%%
P    = zeros(N,M);
optf = opt; optf.tol = 0.025;

% First loop
parfor stest = 1:N
    
    % Get train and test data
    te     = 0:N:N*(M-1);
    te     = stest+te;
    rm     = 1:N*M;
    rm(te) = []; 
       
    keyboard
    % Standardize
    Xz = (X - repmat(mean(X(rm,:)),size(X,1),1)) ./ repmat(std(X(rm,:)),size(X,1),1);
    Xz(isnan(Xz)) = 0;
    
    % Nested LOO-CV loop
    %%%%%%%%%%%%%%%%%%%%
    correct_valid = zeros(length(prange1),length(prange2),M);
    
    for svalid = 1:N - 1
        disp(['.............................................>>>>Test fold:', num2str(stest)]);
        disp(['>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Validation fold:', num2str(svalid)]);
        
        % Get train and test 
        va          = 0:N-1:N*(M-1);
        va          = svalid + va;
        va          = rm(va);
        tr          = 1:N*M;
        tr([te va]) = [];

        % loop over all parameter settings
        nrange1 = length(prange1);
        nrange2 = length(prange2);
        for i = 1:nrange1
            for j = 1:nrange2
                
                % Display results
                disp(sprintf('Parameters: %d out of %d || %d out of %d',i,nrange1,j,nrange2));
                
                lam1 = prange1(i);
                lam2 = prange2(j);
                
                % train
                w_smlr = smlr(Xz(tr,:),Y(tr,:),lam1,lam2,optf);
               
                % test
                p = smlr_multinomial_p(Xz(va,:)*w_smlr, M);
                
                % compute accuracy
                for m = 1:M
                    srange = (m-1)*Nscans_s+1:m*Nscans_s;
                    c = 0;
                    for n = 1:length(srange)
                        [~, maxi] = max(p(srange(n),:));
                        if maxi == m
                            c = c+1; 
                        end
                    end
                    correct_valid(i,j,m) = correct_valid(i,j,m)+c;
                end      
            end
        end
    end
    
    % End nested loop
    %%%%%%%%%%%%%%%%%
    
    % compute accuracy measures
    valid_acc_allclasses = correct_valid ./ (Nscans_s * (N-1));
    valid_acc = mean(valid_acc_allclasses,3);
    
    % now choose best parameter setting
    [m1 ind1] = max(valid_acc);
    [m2 ind2] = max(max(valid_acc));
    opti      = ind1(ind2);
    optj      = ind2;
    lam1      = prange1(opti);
    lam2      = prange2(optj);
    
    % Store lams
    lams_used(stest,:)  = [lam1 lam2];
    acc_fold(:,:,stest) = valid_acc;
    
    % train
    tr = rm;
    w_smlr= smlr(Xz(tr,:),Y(tr,:),lam1,lam2,opt);
    
    % Store weights
    wsmlr_used(stest,:,:) = w_smlr';
    
    % test
    p = smlr_multinomial_p(Xz(te,:)*w_smlr, M);
    
    Pall(:,:,stest) = p;
    % get final P
    for pi = 1:M
        P(stest,pi) = p(pi,pi);
    end
 
end

