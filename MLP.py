import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPClassifier

File_path = 'C:/Learning/data/'
File_name = 'Iris.xlsx'

df = pd.read_excel(File_path+File_name)
x = df.iloc[:, 1:4]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = MLPClassifier(hidden_layer_sizes=(2,2), 
activation = 'relu',solver='adam',max_iter=10000)
model.fit(x_train,y_train)

predict = model.predict([[20,50,90]])

print('Accuracy_Training: ',model.score((x_train), y_train), '\n')
print('Accuracy_Test: ',model.score((x_test), y_test), '\n')

Label = y_train.unique()
y_pred = model.predict(x_test)
Confu = confusion_matrix(y_test ,y_pred)
CM_view = ConfusionMatrixDisplay(confusion_matrix = Confu,display_labels = Label)
CM_view.plot()
plot.show()

