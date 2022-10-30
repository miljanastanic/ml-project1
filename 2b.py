# TODO popuniti kodom za problem 2b
#zadatak 2b
print('zadatak2b')
import pandas as pd
#import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

podaci = pd.read_csv("/content/2a.xls",sep = '\t', usecols=[1, 2, 3, 6], encoding="utf-8")
print(podaci)

def pravljenjeMatrica(podaci, stepen):
    lista = []
    for step in range(1, stepen + 1):
        lista.append(np.power(podaci, step))
    return np.column_stack(lista)

nb_features_min = 1
nb_features_max = 6

data = dict()
podaciNp = podaci.to_numpy()
data['x'] = podaciNp[:, 0]
data['y'] = podaciNp[:, 1]

#nasumicno mesanje podataka
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalizacija (obratiti pažnju na axis=0). preuzeto sa materijala
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])
# Cuvanje domena i kodomena ulaznih podataka
min_x = min(data['x'])
max_x = max(data['x'])
min_y = min(data['y'])
max_y = max(data['y'])

data_loss = []

nb_epochs = 100

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)

for nb_features in range(nb_features_min, nb_features_max+1):
    data_x = data['x'][:]
    data_y = data['y'][:]

    data_loss.append([])

    tf.reset_default_graph()

    # Kreiranje feature matrice
    data_x = pravljenjeMatrica(data_x, nb_features)

    # Ulazni podaci
    X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    Y = tf.placeholder(shape=(None), dtype=tf.float32)
    Y_col = tf.reshape(Y, (-1, 1))

    # Koeficijenti polinoma
    w = tf.Variable(tf.zeros(nb_features))
    bias = tf.Variable(0.0)

    # Matrica parametara
    w_col = tf.reshape(w, (nb_features, -1))

    # Hipoteza
    hyp = tf.add(tf.matmul(X, w_col), bias)

    # Funkcija greske
    mse = tf.reduce_mean(tf.square(hyp - Y_col))

    # Optimizacija
    opt_op = tf.train.AdamOptimizer().minimize(mse)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Trening
        for epoch in range(nb_epochs):
            epoch_loss = 0
            for sample in range(nb_samples):
                feed = {X: data_x[sample].reshape((1, nb_features)),Y: data_y[sample]}
                _, curr_loss = sess.run([opt_op, mse], feed_dict=feed)
                epoch_loss += curr_loss
            epoch_loss /= nb_samples
            data_loss[nb_features-1].append(epoch_loss)
            if epoch % 10 == 0:
                print('{}/{} : {:.5f}'.format(epoch+1, nb_epochs, epoch_loss))
 
        # Racunanje finalnih vrednosti parametara i funkcije greske
        w_val = sess.run(w)
        bias_val = sess.run(bias)
        loss_val = sess.run(mse, feed_dict={X: data_x, Y: data_y})
        print('w = {}, bias = {:.5f}, loss {:.5f}'.format(w_val, bias_val, loss_val))

        # Grafik koji prikazuje funkciju koja opisuje ulazne podatke
        xs = pravljenjeMatrica(np.linspace(min_x, max_x, 100), nb_features)
        hyp_val = sess.run(hyp, feed_dict={X: xs})
        stage = nb_features / nb_features_max
        plt.plot(xs[:,0].tolist(), hyp_val.tolist(), color=(1-stage, stage, 0))

# Grafik koji prikazuje ulazne podatke
plt.scatter(data['x'], data['y'])
plt.xlim([min_x, max_x])
plt.ylim([min_y, max_y])

# Grafik koji prikazuje zavisnost funkcije troska od stepena polinoma
plt.subplot(1, 2, 2)
for nb_features in range(nb_features_min, nb_features_max+1):
    stage = nb_features / nb_features_max
    plt.plot(range(1, nb_epochs+1), data_loss[nb_features-1], color=(1-stage, stage, 0))
plt.show()