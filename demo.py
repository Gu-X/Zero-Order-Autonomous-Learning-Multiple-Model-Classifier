import scipy.io
import os
import sklearn.metrics
from ALMMo0_System import ALMMo0classifier_testing
from ALMMo0_System import ALMMo0classifier_learning
## read the dataset
dirpath = os.getcwd()
dr =dirpath+'\\exampledata.mat'
mat_contents = scipy.io.loadmat(dr)
tradata=mat_contents['Data_Train']
tralabel=mat_contents['Label_Train']
tesdata=mat_contents['Data_Test']
teslabel=mat_contents['Label_Test']
##
SystemParameters=ALMMo0classifier_learning(tradata,tralabel) # Train the ALMMo-0 classifier
# SystemParameters is the trained classifier
TestLabel=ALMMo0classifier_testing(tesdata,SystemParameters) # Conduct classification with the pretrained classifier
print(sklearn.metrics.confusion_matrix(teslabel,TestLabel))# Get the confusion matrix
