%------------------------------------
% KMeans
%------------------------------------

clear

%% Kmeans algorithm

load flood_data_Notdead.mat;
% X has the data
% Y is the class 1-Flood 0-Not flood or 1-Deads 0-not Deads
                 
[num_patrones, num_variables] = size(X);
p_train = 0.7;

num_patrones_train = round(p_train*num_patrones);

ind_permuta = randperm(num_patrones);

inds_train = ind_permuta(1:num_patrones_train);
inds_test = ind_permuta(num_patrones_train+1:end);

X_train = X(inds_train, :);
Y_train = Y(inds_train);

X_test= X(inds_test, :);
Y_test = Y(inds_test);

k=3;

Y_test_asig = fClassify_kNN(X_train, Y_train, X_test, k);

% confusion matrix plot
plotconfusion(Y_test, Y_test_asig);

[f,c]=size(Y_test_asig);

TP = 0;
FP = 0;
TN = 0;
FN = 0;

for i=1: c
    if(Y_test_asig(1,i)==Y_test(1,i) && Y_test(1,i)==0)
        TN = TN+1;
    end
    
    if(Y_test_asig(1,i)==Y_test(1,i) && Y_test(1,i)==1)
        TP = TP+1;
    end
    
    if(Y_test_asig(1,i)~=Y_test(1,i) && Y_test(1,i)==0)
        FN = FN+1;
    end
    
    if(Y_test_asig(1,i)~=Y_test(1,i) && Y_test(1,i)==1)
        FP = FP+1;
    end
end

error = (FP+FN)/(TP+TN+FP+FN);

fprintf('\n******\Global error = %1.4f%% (classification)\n', error*100);
% False Positive Rate
   
FPR = FP/(FP+TN);
fprintf('\n******\nFalse Positive Rate = %1.4f%% (classification)\n', FPR*100);

% False Negative Rate
FNR = FN/(TP+FN);
fprintf('\n******\nFalse Negative Rate = %1.4f%% (classification)\n', FNR*100);

% Precision
precision = TP/(TP+FP);
fprintf('\n******\nPrecision = %1.4f%% (classification)\n', precision*100);

% Recall
recall = sum(Y_test_asig==1 & Y_test==1)/sum(Y_test==1);
fprintf('\n******\nRecall = %1.4f%% (classification)\n', recall*100);

