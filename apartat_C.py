from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
#%matplotlib notebook
from matplotlib import pyplot as plt
import scipy.stats


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = load_dataset('day.csv')
data = dataset.values

x = data[:, :16]
y = data[:, 15]

############

# print("Dimensionalitat de la BBDD:", dataset.shape)
# print("Dimensionalitat de les entrades X", x.shape)
# print("Dimensionalitat de l'atribut Y", y.shape)

############

# print("Per comptar el nombre de valors no existents:")
# print(dataset.isnull().sum())

############

# print("Per visualitzar les primeres 5 mostres de la BBDD:")
# print(dataset.head())

############

# print("Per veure estadístiques dels atributs numèrics de la BBDD:")
# print(dataset.describe())

############

# mostrem atribut 0
# plt.figure()

############

# ax = plt.scatter(x[:,15], y)
# plt.show()

############

plt.figure()
plt.title("Histograma de l'atribut 0")
plt.xlabel("Attribute Value")
plt.ylabel("Count")
hist = plt.hist(x[:,15], bins=11, range=[np.min(x[:,15]), np.max(x[:,15])], histtype="bar", rwidth=0.8)
plt.show()

############


import seaborn as sns

# # Mirem la correlació entre els atributs d'entrada per entendre millor les dades
# correlacio = dataset.corr()

# plt.figure()

# ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
# plt.show()

############

# # Mirem la relació entre atributs utilitzant la funció pairplot
# relacio = sns.pairplot(dataset)
# plt.show()

############

# Quin és el tipus de cada atribut?

# Quins atributs tenen una distribució Guassiana?

# Quin és l'atribut objectiu? Per què?

############
############

