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

import scipy.stats as stats
import pylab

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso


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
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=1)
    #This is the training process. it is being skipped now that the model has been created

    lasso = Lasso(alpha=0.001)

    #lasso = Lasso(alpha=0.001)
   # lasso = Lasso(alpha=0.059)

    lasso.fit(x_train,y_train)

    #accuracy of model
    acc = lasso.score(x_test,y_test)
    print(acc)

    #if acc > best:
        #best = acc
        #store model
        #with open("studentmodellasso.pickle", "wb") as f:
            #pickle.dump(lasso,f)


    # store model
    with open("studentmodellasso.pickle", "wb") as f:
        pickle.dump(lasso, f)


print('Best accuracy: ', best)




####

######


##cut here####'''


'''pickle_in = open("studentmodellasso.pickle", "rb")
lasso = pickle.load(pickle_in)'''

print("Coefficient: " , lasso.coef_)
print("Intercept: " , lasso.intercept_)

# Using the test data for predictions
predictions = lasso.predict(x_test)
train_predictions = lasso.predict(x_train)


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
second_predictions = lasso.predict(X)
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


# prepare a range of alpha values to test
list = []

for i in range(1000):
    list.append(i/1000)
print(list)




alphas = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.11, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129, 0.13, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139, 0.14, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149, 0.15, 0.151, 0.152, 0.153, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.16, 0.161, 0.162, 0.163, 0.164, 0.165, 0.166, 0.167, 0.168, 0.169, 0.17, 0.171, 0.172, 0.173, 0.174, 0.175, 0.176, 0.177, 0.178, 0.179, 0.18, 0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.19, 0.191, 0.192, 0.193, 0.194, 0.195, 0.196, 0.197, 0.198, 0.199, 0.2, 0.201, 0.202, 0.203, 0.204, 0.205, 0.206, 0.207, 0.208, 0.209, 0.21, 0.211, 0.212, 0.213, 0.214, 0.215, 0.216, 0.217, 0.218, 0.219, 0.22, 0.221, 0.222, 0.223, 0.224, 0.225, 0.226, 0.227, 0.228, 0.229, 0.23, 0.231, 0.232, 0.233, 0.234, 0.235, 0.236, 0.237, 0.238, 0.239, 0.24, 0.241, 0.242, 0.243, 0.244, 0.245, 0.246, 0.247, 0.248, 0.249, 0.25, 0.251, 0.252, 0.253, 0.254, 0.255, 0.256, 0.257, 0.258, 0.259, 0.26, 0.261, 0.262, 0.263, 0.264, 0.265, 0.266, 0.267, 0.268, 0.269, 0.27, 0.271, 0.272, 0.273, 0.274, 0.275, 0.276, 0.277, 0.278, 0.279, 0.28, 0.281, 0.282, 0.283, 0.284, 0.285, 0.286, 0.287, 0.288, 0.289, 0.29, 0.291, 0.292, 0.293, 0.294, 0.295, 0.296, 0.297, 0.298, 0.299, 0.3, 0.301, 0.302, 0.303, 0.304, 0.305, 0.306, 0.307, 0.308, 0.309, 0.31, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317, 0.318, 0.319, 0.32, 0.321, 0.322, 0.323, 0.324, 0.325, 0.326, 0.327, 0.328, 0.329, 0.33, 0.331, 0.332, 0.333, 0.334, 0.335, 0.336, 0.337, 0.338, 0.339, 0.34, 0.341, 0.342, 0.343, 0.344, 0.345, 0.346, 0.347, 0.348, 0.349, 0.35, 0.351, 0.352, 0.353, 0.354, 0.355, 0.356, 0.357, 0.358, 0.359, 0.36, 0.361, 0.362, 0.363, 0.364, 0.365, 0.366, 0.367, 0.368, 0.369, 0.37, 0.371, 0.372, 0.373, 0.374, 0.375, 0.376, 0.377, 0.378, 0.379, 0.38, 0.381, 0.382, 0.383, 0.384, 0.385, 0.386, 0.387, 0.388, 0.389, 0.39, 0.391, 0.392, 0.393, 0.394, 0.395, 0.396, 0.397, 0.398, 0.399, 0.4, 0.401, 0.402, 0.403, 0.404, 0.405, 0.406, 0.407, 0.408, 0.409, 0.41, 0.411, 0.412, 0.413, 0.414, 0.415, 0.416, 0.417, 0.418, 0.419, 0.42, 0.421, 0.422, 0.423, 0.424, 0.425, 0.426, 0.427, 0.428, 0.429, 0.43, 0.431, 0.432, 0.433, 0.434, 0.435, 0.436, 0.437, 0.438, 0.439, 0.44, 0.441, 0.442, 0.443, 0.444, 0.445, 0.446, 0.447, 0.448, 0.449, 0.45, 0.451, 0.452, 0.453, 0.454, 0.455, 0.456, 0.457, 0.458, 0.459, 0.46, 0.461, 0.462, 0.463, 0.464, 0.465, 0.466, 0.467, 0.468, 0.469, 0.47, 0.471, 0.472, 0.473, 0.474, 0.475, 0.476, 0.477, 0.478, 0.479, 0.48, 0.481, 0.482, 0.483, 0.484, 0.485, 0.486, 0.487, 0.488, 0.489, 0.49, 0.491, 0.492, 0.493, 0.494, 0.495, 0.496, 0.497, 0.498, 0.499, 0.5, 0.501, 0.502, 0.503, 0.504, 0.505, 0.506, 0.507, 0.508, 0.509, 0.51, 0.511, 0.512, 0.513, 0.514, 0.515, 0.516, 0.517, 0.518, 0.519, 0.52, 0.521, 0.522, 0.523, 0.524, 0.525, 0.526, 0.527, 0.528, 0.529, 0.53, 0.531, 0.532, 0.533, 0.534, 0.535, 0.536, 0.537, 0.538, 0.539, 0.54, 0.541, 0.542, 0.543, 0.544, 0.545, 0.546, 0.547, 0.548, 0.549, 0.55, 0.551, 0.552, 0.553, 0.554, 0.555, 0.556, 0.557, 0.558, 0.559, 0.56, 0.561, 0.562, 0.563, 0.564, 0.565, 0.566, 0.567, 0.568, 0.569, 0.57, 0.571, 0.572, 0.573, 0.574, 0.575, 0.576, 0.577, 0.578, 0.579, 0.58, 0.581, 0.582, 0.583, 0.584, 0.585, 0.586, 0.587, 0.588, 0.589, 0.59, 0.591, 0.592, 0.593, 0.594, 0.595, 0.596, 0.597, 0.598, 0.599, 0.6, 0.601, 0.602, 0.603, 0.604, 0.605, 0.606, 0.607, 0.608, 0.609, 0.61, 0.611, 0.612, 0.613, 0.614, 0.615, 0.616, 0.617, 0.618, 0.619, 0.62, 0.621, 0.622, 0.623, 0.624, 0.625, 0.626, 0.627, 0.628, 0.629, 0.63, 0.631, 0.632, 0.633, 0.634, 0.635, 0.636, 0.637, 0.638, 0.639, 0.64, 0.641, 0.642, 0.643, 0.644, 0.645, 0.646, 0.647, 0.648, 0.649, 0.65, 0.651, 0.652, 0.653, 0.654, 0.655, 0.656, 0.657, 0.658, 0.659, 0.66, 0.661, 0.662, 0.663, 0.664, 0.665, 0.666, 0.667, 0.668, 0.669, 0.67, 0.671, 0.672, 0.673, 0.674, 0.675, 0.676, 0.677, 0.678, 0.679, 0.68, 0.681, 0.682, 0.683, 0.684, 0.685, 0.686, 0.687, 0.688, 0.689, 0.69, 0.691, 0.692, 0.693, 0.694, 0.695, 0.696, 0.697, 0.698, 0.699, 0.7, 0.701, 0.702, 0.703, 0.704, 0.705, 0.706, 0.707, 0.708, 0.709, 0.71, 0.711, 0.712, 0.713, 0.714, 0.715, 0.716, 0.717, 0.718, 0.719, 0.72, 0.721, 0.722, 0.723, 0.724, 0.725, 0.726, 0.727, 0.728, 0.729, 0.73, 0.731, 0.732, 0.733, 0.734, 0.735, 0.736, 0.737, 0.738, 0.739, 0.74, 0.741, 0.742, 0.743, 0.744, 0.745, 0.746, 0.747, 0.748, 0.749, 0.75, 0.751, 0.752, 0.753, 0.754, 0.755, 0.756, 0.757, 0.758, 0.759, 0.76, 0.761, 0.762, 0.763, 0.764, 0.765, 0.766, 0.767, 0.768, 0.769, 0.77, 0.771, 0.772, 0.773, 0.774, 0.775, 0.776, 0.777, 0.778, 0.779, 0.78, 0.781, 0.782, 0.783, 0.784, 0.785, 0.786, 0.787, 0.788, 0.789, 0.79, 0.791, 0.792, 0.793, 0.794, 0.795, 0.796, 0.797, 0.798, 0.799, 0.8, 0.801, 0.802, 0.803, 0.804, 0.805, 0.806, 0.807, 0.808, 0.809, 0.81, 0.811, 0.812, 0.813, 0.814, 0.815, 0.816, 0.817, 0.818, 0.819, 0.82, 0.821, 0.822, 0.823, 0.824, 0.825, 0.826, 0.827, 0.828, 0.829, 0.83, 0.831, 0.832, 0.833, 0.834, 0.835, 0.836, 0.837, 0.838, 0.839, 0.84, 0.841, 0.842, 0.843, 0.844, 0.845, 0.846, 0.847, 0.848, 0.849, 0.85, 0.851, 0.852, 0.853, 0.854, 0.855, 0.856, 0.857, 0.858, 0.859, 0.86, 0.861, 0.862, 0.863, 0.864, 0.865, 0.866, 0.867, 0.868, 0.869, 0.87, 0.871, 0.872, 0.873, 0.874, 0.875, 0.876, 0.877, 0.878, 0.879, 0.88, 0.881, 0.882, 0.883, 0.884, 0.885, 0.886, 0.887, 0.888, 0.889, 0.89, 0.891, 0.892, 0.893, 0.894, 0.895, 0.896, 0.897, 0.898, 0.899, 0.9, 0.901, 0.902, 0.903, 0.904, 0.905, 0.906, 0.907, 0.908, 0.909, 0.91, 0.911, 0.912, 0.913, 0.914, 0.915, 0.916, 0.917, 0.918, 0.919, 0.92, 0.921, 0.922, 0.923, 0.924, 0.925, 0.926, 0.927, 0.928, 0.929, 0.93, 0.931, 0.932, 0.933, 0.934, 0.935, 0.936, 0.937, 0.938, 0.939, 0.94, 0.941, 0.942, 0.943, 0.944, 0.945, 0.946, 0.947, 0.948, 0.949, 0.95, 0.951, 0.952, 0.953, 0.954, 0.955, 0.956, 0.957, 0.958, 0.959, 0.96, 0.961, 0.962, 0.963, 0.964, 0.965, 0.966, 0.967, 0.968, 0.969, 0.97, 0.971, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977, 0.978, 0.979, 0.98, 0.981, 0.982, 0.983, 0.984, 0.985, 0.986, 0.987, 0.988, 0.989, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999])

'''# create and fit a ridge regression model, testing each alpha


bestone = 0
bestestimator = 1000
for i in range(1):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2,random_state=0)


    model = Lasso()

    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(x_train,y_train)
    print(grid)
    # summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.alpha)

    if (grid.best_score_ > bestone):
        bestone = grid.best_score_
        bestestimator =grid.best_estimator_.alpha


print("Final results: ")
print(bestone)
print(bestestimator)
######Cuthere'''


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



