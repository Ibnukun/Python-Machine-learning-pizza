import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


modelnb = GaussianNB()
pizzaDataset = pd.read_csv('Pizza.csv',names = ['Brand', 'mois', 'prot', 'fat', 'ash','sodium','carb','cal'], header=0, sep=",")
x = pizzaDataset.drop(["Brand"], axis=1)
x.head(300)
y = pizzaDataset["Brand"]
y.head(300)
x_test = pizzaDataset.drop(["Brand"], axis=1)
x_test.head(300)
y_uji = pizzaDataset["Brand"]
y_uji.head(300)
nbtrain = modelnb.fit(x, y)
Y_predict = nbtrain.predict(x_test)
accuracy= accuracy_score(y_uji, Y_predict)
print("Akurasi Naive Bayes : ",accuracy)