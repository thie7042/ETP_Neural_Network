import tensorflow as tf

import sklearn
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.linear_model import Lasso

from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.metrics import r2_score
from sklearn import metrics
import scipy.stats
from sklearn.linear_model import ElasticNet
import scipy.stats as stats
import pylab
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNetCV

from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

style.use("ggplot")

# data = pd.read_csv("student-mat.csv", sep=";")

# data = pd.read_csv("Workingdata.csv")
data = pd.read_csv("Masterdata.csv")
# Choosing the attributes which we want



#data = data[["STR_A_Bot_02m","TMP_C_Bot_02m"]]

data = data[["STR_A_Bot_02m","TMP_C_Bot_02m","TMP_C_Bot_12m","TMP_C_Bot_17m"]]
print(data.head())

#G3 is a label i.e what we are looking to predict
# predict = "G3"
# predict = "STR_A_Bot_09m"
predict = "STR_A_Bot_02m"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
#keeping track of best score


###########
from sklearn import metrics
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV









###cut here
best = 0;
for _ in range(1):
    #Taking all attributes and labels and splitting them up into training and test arrays - 20% of data is a test sample
    #x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)
    #This is the training process. it is being skipped now that the model has been created

    # lasso = Lasso(alpha=0.0001)
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)


    elastic.fit(x_train,y_train)

    #accuracy of model
    acc = elastic.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        #store model
        with open("studentmodelelast.pickle", "wb") as f:
            pickle.dump(elastic,f)


print('Best accuracy: ', best)




####

######


##cut here####'''

pickle_in = open("studentmodelelast.pickle", "rb")
elastic = pickle.load(pickle_in)

print("Coefficient: " , elastic.coef_)
print("Intercept: " , elastic.intercept_)

# Using the test data for predictions
predictions = elastic.predict(x_test)
train_predictions = elastic.predict(x_train)


for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


plt.plot(predictions)
plt.plot(y_test)
plt.show()

# p ="TMP_C_Bot_02m"
# style.use("ggplot")
# plt.scatter(data[p],data["STR_A_Bot_02m"])
# plt.show()


plt.scatter(y_test, predictions, s=1)
plt.xlabel("True Values (Strain µε)")
plt.ylabel("Predictions (Strain µε)")
plt.axis('equal')
plt.axis('square')
# plt.xlim([150,plt.xlim()[1]])
# plt.ylim([150,plt.ylim()[1]])




plt.plot([170, 232], [170, 232], color='black', linewidth=0.5)

plt.xlim(170,232)
plt.ylim(170,232)



plt.show()




######################################
'''data = pd.read_csv("Masterdata.csv")
second_predictions = elastic.predict(X)
MAsecond_predictions = pd.Series(second_predictions).rolling(window=100).mean()
MAy = pd.Series(y).rolling(window=100).mean()
plt.plot(MAsecond_predictions,  c='b', label='Prediction',linewidth=1)
plt.plot(second_predictions,  c='b',linewidth=0.01)

plt.plot(MAy,  c='r', label='True values',linewidth=1)
plt.plot(y,  c='r',linewidth=0.01)


plt.legend()
plt.show()'''

# ####################################################


####



'''plt.title("Strain Prediction")
plt.plot(y, label = 'Test Data',linewidth=1.0)
plt.plot(second_predictions,  label = 'Test Prediction',linewidth=1.0)
plt.xlim(0,104000)
plt.legend()
plt.xlabel("Timestamp (seconds)")
plt.ylabel("Strain (µε)")
plt.show()'''
####


#y_hat_train = linear.predict(y_train )

###


print('_________________________________')
#print("The R^2 score on the train set is: \t{:0.3f}".format(r2_score(train_labels,train_predictions)))
#print(r2_score(train_labels,train_predictions))
print("The R^2 score on the test set is: \t{:0.3f}".format(r2_score(y_test, predictions)))
print((r2_score(y_test, predictions)))


#######
print('____________________')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#####
print('____________________')
print("The R^2 score on the train set is: \t{:0.3f}".format(r2_score(y_train,  train_predictions)))
print(r2_score(y_train,  train_predictions))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, train_predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, train_predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, train_predictions)))
print('____________________')
#####


#
# data = pd.read_csv("Testingdata.csv")
# data = data[["STR_A_Bot_02m","TMP_C_Bot_02m"]]
# X = np.array(data.drop([predict],1))
# y = np.array(data[predict])
# second_predictions = linear.predict(X)
#
#
#
# MAsecond_predictions = pd.Series(second_predictions).rolling(window=100).mean()
# MAy = pd.Series(y).rolling(window=100).mean()
#
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Horizontally stacked subplots')
# ax1.plot(MAsecond_predictions,  c='b', label='Prediction',linewidth=1)
# ax1.plot(second_predictions,  c='b',linewidth=0.01)
# ax1.plot(MAy,  c='r', label='True values',linewidth=1)
# ax1.plot(y,  c='r',linewidth=0.01)
#
# ax2.scatter(y, second_predictions)
# plt.show()
#
#
# plt.plot(MAsecond_predictions,  c='b', label='Prediction',linewidth=1)
# plt.plot(second_predictions,  c='b',linewidth=0.01)
# plt.ylim((150, 230))
# plt.plot(MAy,  c='r', label='True values',linewidth=1)
# plt.plot(y,  c='r',linewidth=0.01)
#
#
# plt.legend()
# plt.show()




###################################







fig, ax = plt.subplots(figsize=(6, 4))

print("errors:")
error = pd.Series(predictions - y_test)



print(error)

std = np.std(error, ddof=1)
mean = np.mean(error)




domain = np.linspace(np.min(error),np.max(error))
plt.plot(domain, scipy.stats.norm.pdf(domain,mean,std), label = '$\mathcal{N}$' + f'$( \mu \\approx {round(mean)}, \sigma \\approx {round(std)} )$' )
error.plot(kind="hist", density=True, alpha=0.65, bins=25)  # change density to true, because KDE uses density
# Plot KDE
error.plot(kind="kde", color="black", label = 'Kernel density estimate')
plt.title("Normal Fit & KDE")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.legend()
plt.show()



stats.probplot(error, dist="norm", plot=pylab)
pylab.show()



####Testing , Cut here



print("__________________________TESTING___________________________")



cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
ratios = arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# fit model
model.fit(x_train,y_train)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
print('l1_ratio_: %f' % model.l1_ratio_)
##CUT HERE


#####






nsample = 100
rng = np.random.default_rng()
ax = plt.subplot(111)
x = stats.norm.rvs(loc=0, scale=1, size=nsample, random_state=rng)
res = stats.probplot(error, plot=plt)

ax.get_lines()[0].set_marker('.')
ax.get_lines()[0].set_markeredgecolor('black')
ax.get_lines()[0].set_markerfacecolor('none')
ax.get_lines()[0].set_markersize(12.0)
ax.get_lines()[0].set_markeredgewidth(0.4)



ax.get_lines()[1].set_linewidth(1.0)
plt.title("Normal Q-Q Plot")
plt.show()



