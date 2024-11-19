


import numpy as np
import matplotlib.pyplot as plt





!pip install keras





import os
os.environ['KERAS_BACKEND'] = 'jax'





import keras
from keras.models import Sequential
from keras.layers import Dense


Dense?





def generate_data(x_min=0, x_max=5, data_size=400):
    x = np.linspace(x_min, x_max, num=data_size)
    x = x.reshape(data_size, 1)
    ϵ = 0.2 * np.random.randn(*x.shape)
    y = x**0.5 + np.sin(x) + ϵ
    x, y = [z.astype('float32') for z in (x, y)]
    return x, y





x, y = generate_data()





fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()





x_validate, y_validate = generate_data()





def build_regression_model(model):
    model.add(Dense(units=1))
    model.compile(optimizer=keras.optimizers.SGD(), 
                  loss='mean_squared_error')
    return model





def build_nn_model(model, k=10, activation_function='tanh'):
    # Construct network
    model.add(Dense(units=k, activation=activation_function))
    model.add(Dense(units=k, activation=activation_function))
    model.add(Dense(units=k, activation=activation_function))
    model.add(Dense(1))
    # Embed training configurations
    model.compile(optimizer=keras.optimizers.SGD(), 
                  loss='mean_squared_error')
    return model





def plot_loss_history(training_history, ax):
    ax.plot(training_history.epoch, 
            np.array(training_history.history['loss']), 
            label='training loss')
    ax.plot(training_history.epoch, 
            np.array(training_history.history['val_loss']),
            label='validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Mean squared error)')
    ax.legend()





model = Sequential()
regression_model = build_regression_model(model)





training_history = regression_model.fit(
    x, y, batch_size=x.shape[0], verbose=0,
    epochs=4000, validation_data=(x_validate, y_validate))





fig, ax = plt.subplots()
plot_loss_history(training_history, ax)
plt.show()





print("Testing loss on the validation set.")
regression_model.evaluate(x_validate, y_validate, verbose=2)





y_predict = regression_model.predict(x_validate, verbose=2)





def plot_results(x, y, y_predict, ax):
    ax.scatter(x, y)
    ax.plot(x, y_predict, label="fitted model", color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')





fig, ax = plt.subplots()
plot_results(x_validate, y_validate, y_predict, ax)
plt.show()





model = Sequential()
nn_model = build_nn_model(model)


training_history = nn_model.fit(
    x, y, batch_size=x.shape[0], verbose=0,
    epochs=4000, validation_data=(x_validate, y_validate))


fig, ax = plt.subplots()
plot_loss_history(training_history, ax)
plt.show()





print("Testing loss on the validation set.")
nn_model.evaluate(x_validate, y_validate, verbose=2)





y_predict = nn_model.predict(x_validate, verbose=2)


def plot_results(x, y, y_predict, ax):
    ax.scatter(x, y)
    ax.plot(x, y_predict, label="fitted model", color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


fig, ax = plt.subplots()
plot_results(x_validate, y_validate, y_predict, ax)
plt.show()
