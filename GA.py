from __future__ import absolute_import, division, print_function
import seaborn as sns
import pathlib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import joblib
import scipy.stats
from matplotlib import style
from sklearn.metrics import r2_score
import pylab
import scipy.stats as stats
#
import numpy as np
from scipy.stats import norm
#

####Use Masterdata for current best



style.use("ggplot")
print(tf.__version__)


data = pd.read_csv("Masterdata.csv")

data = data[["STR_A_Bot_02m","TMP_C_Bot_02m","TMP_C_Bot_12m","TMP_C_Bot_17m"]]


train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)





train_stats = train_dataset.describe()
train_stats.pop('STR_A_Bot_02m')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('STR_A_Bot_02m')
test_labels = test_dataset.pop('STR_A_Bot_02m')


def norm(x):
    return (x - train_stats['mean'])/ train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data =norm(test_dataset)
################################################################################################################








#CUTHERE




from keras.layers import InputLayer, Input

#for i in range(4):
#while analysis == True:
    ###Cut here
def run_once(n_layers, n_neurons, learning_rate):

    # 64 outputs added to the first two layers
    # 1 hidden layer with 64 inputs
    # 1 output layer with 1 output (the predicted strain)


    def build_model(n_layers, n_neurons, learning_rate):

        if n_layers == 1:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))

        elif n_layers == 2:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
        elif n_layers == 3:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
        elif n_layers == 4:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(3,)))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
            model.add(tf.keras.layers.Dense(1))

        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        #print("neurons \/ :")
        #print((i+1)*5)
        # optimizer = tf.keras.optimizers.RMSprop(0.01)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model


    model = build_model(n_layers, n_neurons, learning_rate)

    # the patience parameter is the amount of epochs to check for improvement - Use 30
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    model.summary()


    class PrintDot(keras.callbacks.Callback):
        counter = 0

        def on_epoch_end(self, epoch, logs):
            PrintDot.counter += 1
            print(PrintDot.counter)


    #
    #EPOCHS = 1000
    EPOCHS = 100
    # EPOCHS = 100

    history = model.fit(
        normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[early_stop, PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    print(hist)


    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch



        print('Train Error MSE')
        print(hist['mse'])

        print('Val Error MSE')
        print(hist['val_mse'])
        #####



    #plot_history(history)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} Strain".format(mae))

    if mae < run_once.bestmae:
        print("New best:" )
        model.summary()
        run_once.bestmae = mae
        model.save('saved_model_tune/my_GA_model.h5')
    # mse = fitness to be minimized
    return mse, mae

run_once.bestmae = 10

#mse, bestmse = run_once(2,10,0.02, bestmse)
######################SAVING############################

# Save the entire model as a SavedModel.
# model.save('saved_model_tune/my_model.h5')

#####Cut here##

import pygad
import numpy





number_of_genes = 3

#num_generations=20
#sol_per_pop=50

num_generations=40
sol_per_pop=20


gene_space = [{'low': 1, 'high': 4, "step": 1},
              {'low': 1, 'high': 50, "step": 1},
              {'low': 0, 'high': 0.1, "step": 0.001}]



recordingAll = []
recAllFit = []



def on_start(ga):
    print("This is the start")


    init =  ga_instance.initial_population





def on_generation(ga):
    print("New Generation")

    print("Generation", ga.generations_completed)
    #print(ga.population)


    print('Gens complete:' + str(ga.generations_completed) )




        ##############################################################################

on_generation.counting = 0



def fitness_func(solution, solution_idx):

    capture = [solution[0], solution[1], solution[2]]

    mse, fitness = run_once(solution[0], solution[1], solution[2])




    print("MAE: " , fitness)

    fitness = 100000/ (fitness*100 + 1)

    print(fitness)

    recordingAll.append(capture)
    recAllFit.append(fitness)

    return fitness





def parent_selection_func(fitness, num_parents, ga_instance):

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()

    parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

    for parent_num in range(num_parents):
        parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]



ga_instance = pygad.GA(num_generations= num_generations,
                       sol_per_pop= sol_per_pop,
                       num_parents_mating=2,
                       num_genes= number_of_genes,
                       fitness_func=fitness_func,
                       gene_type=[int, int, float],
                       mutation_num_genes=1,
                       gene_space = gene_space,
                       on_generation=on_generation,
                       parent_selection_type=parent_selection_func,
                       on_start= on_start)


#print("Initial Population")
#print(ga_instance.initial_population)

ga_instance.run()



#print("Final Population")
#print(ga_instance.population)



print("______________________________")
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))



ga_instance.plot_fitness()


##################Graphinh









#######################LOADING########################


print('#######################################################################')

model = tf.keras.models.load_model('saved_model_tune/my_GA_model.h5')

print("Best model parameters: ")
model.summary()
######################################################

"""train_predictions = model.predict(normed_train_data).flatten()
# line above is testing

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} Strain".format(mae))

test_predictions = model.predict(normed_test_data).flatten()
style.use("ggplot")

plt.scatter(test_labels, test_predictions, s=1)
plt.xlabel("True Values (Strain µε)")
plt.ylabel("Predictions (Strain µε)")
plt.title("Actual Strain vs Predicted Strain")
plt.plot([170, 232], [170, 232], color='black', linewidth=0.5)

print(str(r2_score(test_labels, test_predictions)))

plt.axis("equal")
plt.axis('square')
plt.xlim(170,232)
plt.ylim(170,232)



plt.show()

plt.title("Strain Prediction")
plt.plot(test_labels.values.flatten(), label = 'Test Data',linewidth=1.0)
plt.plot(test_predictions,  label = 'Test Prediction',linewidth=1.0)
plt.xlim(0,20500)
plt.legend()
plt.xlabel("Timestamp (seconds)")
plt.ylabel("Strain (µε)")
plt.show()

print('Length: ', len(test_labels))
print('Length: ', len(test_predictions))

print(test_labels)
print(test_predictions)


print(mae)
print(mse)
print(loss)

# Frequency distribution
error = test_predictions - test_labels

plt.hist(error, bins = 25, color="#56B4E9", alpha=0.65,)
plt.plot(kind="kde")
plt.xlabel("Prediction Error [Strain µε]")
_= plt.ylabel("Count")
plt.title("Frequency Distribution Histogram")

plt.show()

# Testing results for normality
##############################

fig, ax = plt.subplots(figsize=(6, 4))

# Plot
# Plot histogram
print(type(error))

std = np.std(error, ddof=1)
mean = np.mean(error)
printmean = str(mean)
print("This is the mean")
print(printmean)

domain = np.linspace(np.min(error),np.max(error))
plt.plot(domain, scipy.stats.norm.pdf(domain,mean,std), label = '$\mathcal{N}$' + f'$( \mu \\approx {round(mean)}, \sigma \\approx {round(std)} )$' )
error.plot(kind="hist", density=True, alpha=0.65, bins=25)  # change density to true, because KDE uses density
# Plot KDE
error.plot(kind="kde", color="black", label = 'Kernel density estimate')
plt.title("Normal Fit & KDE")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.legend()


# Overall
# ax.grid(False)
# ax.set_title("Avocado Prices in U.S. Markets", size=17, pad=10)

#
#


plt.show()
print("Error length: ", len(error))

###########################
print('_________________________________')
print("The R^2 score on the train set is: \t{:0.3f}".format(r2_score(train_labels,train_predictions)))
print(r2_score(train_labels,train_predictions))
print("The R^2 score on the test set is: \t{:0.3f}".format(r2_score(test_labels,test_predictions)))
print(r2_score(test_labels,test_predictions))
print("Testing set Mean Abs Error: {:5.2f} Strain".format(mae))



stats.probplot(error, dist="norm", plot=pylab)
pylab.show()



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
plt.show()"""




######################TESTING BELOW HERE######################'''
