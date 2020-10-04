function Y_assign = fClassify_kNN(X_train, Y_train, X_test, k)
% This function implements the kNN classification algorithm with the
% eucludean distance
%
% INPUT
%   - X_train: Matrix (n_train x n), where n_Train is the number of 
%   training elements and n is the number of features (the length of the 
%   feature vector)
%   - Y_train: The classes of the elements in the training set. It is a
%   vector of length n_train with the number of the class.
%   - X_test: matrix (n_test x n), where n_test is the number of elements 
%   in the test set and n is the number of features (the length of the 
%   feature vector).
%   - k: Number of nearest neighbours to consider in order to make an
%   assignation
%
% OUTPUT
%   A vector with length n_test, with the classess assigned by the algorithm 
%   to the elements in the training set.
%

    numElemTest = size(X_test, 1);
    numElemTrain = size(X_train, 1);

    Y_assign = zeros(1, numElemTest);
    
    for i=1:numElemTest
        
        x_test_i = X_test(i,:);
          
        [filas,columnas]=size(x_test_i);
        distanciasEuclideas = zeros(1,numElemTrain);
        
        for j=1:numElemTrain
            x_train_j = X_train(j,:);
            
            sumatorioDist = 0;
            
            for ii=1:columnas
             sumatorioDist = sumatorioDist + (x_test_i(1,ii) - x_train_j(1,ii))^2;
            end
            distanciasEuclideas(j) = sqrt(sumatorioDist);
        end

        [distanciasEuclideasOrdenadas, indices] = sort(distanciasEuclideas, 'ascend');
        
         countClass1 = 0;
         countClass0 = 0;
         for j=1:k 
             if(Y_train(indices(1,j)) == 1 )
                 countClass1 = countClass1 +1;
             else
                 countClass0 = countClass0 +1;
             end
         end
            if(countClass1>=countClass0)
                Y_assign(1,i) = 1;
            else
                Y_assign(1,i) = 0;
            end
    end

end

