

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import statsmodels.api as sm
from scipy import stats




## LDA 
train = pd.read_csv('/Users/cherry/Downloads/quality_train.csv')
test = pd.read_csv('/Users/cherry/Downloads/quality_test.csv')

train = pd.DataFrame(train)
test = pd.DataFrame(test)

X_train= train[['popular','genre']]
y_train= train['category']
X_test= train[['popular','genre']]
y_test= train['category']

lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)


pred=model.predict(X_test)
print(np.unique(pred, return_counts=True))

print("------LDA confusion matrix------")
print(confusion_matrix(pred, y_test))


print("-----LDA classification report------")
print(classification_report(y_test, pred, digits=3))



##multiple class logistic regression 
logisticregression= LogisticRegression(C=1, multi_class='ovr',
                                 max_iter=100).fit(X_train, y_train)

pred_multi = logisticregression.predict(X_test)
print("------LOGISTIC confusion matrix------")
print(confusion_matrix(pred_multi, y_test))
print("-----LOGISTIC classification report------")
print(classification_report(y_test, pred_multi, digits=3))

training_accuracy = []
test_accuracy = []
print("-----logistic regresion coefficients------")
print(logisticregression.coef_, logisticregression.intercept_)


params = np.append(logisticregression.intercept_,logisticregression.coef_)


newX = pd.DataFrame({"Constant":np.ones(len(X_train))}).join(pd.DataFrame(X_train))
MSE = (sum((y_train-pred_multi)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
print('--95% confidence interval--"')
print(myDF3)


# try c values from 0.001 to 100:
c_settings = np.arange(0.001, 100, 1)
for i in c_settings:
    # build the model
    clf = LogisticRegression(C=i, multi_class='auto', max_iter=1000)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(c_settings, training_accuracy, label="training accuracy")
plt.plot(c_settings, test_accuracy, label="test accuracy")
plt.legend()




##QDA
qda = QuadraticDiscriminantAnalysis()
model2 = qda.fit(X_train, y_train)

pred2=model2.predict(X_test)
print(np.unique(pred2, return_counts=True))
print("------QDA confusion matrix------")
print(confusion_matrix(pred2, y_test))
print("-----QDA classification report------")
print(classification_report(y_test, pred2, digits=3))

##knn


neighbors = [1,5,10,50]
train_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
      
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)

  
# Generate plot
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("------KNN confusion matrix------")
print(confusion_matrix(y_test, y_pred))
print("-----KNN classification report------")
print(classification_report(y_test, y_pred))

















