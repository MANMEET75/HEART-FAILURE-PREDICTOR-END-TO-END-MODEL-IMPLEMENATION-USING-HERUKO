import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv("heart_failure_clinical_records_dataset.csv")
dataset.drop(["creatinine_phosphokinase","diabetes","platelets","anaemia","serum_sodium","sex","smoking"],axis=1,inplace=True)

X = dataset.iloc[:, :-1].values  # independent variables   
y = dataset.iloc[:, -1].values   # dependent variable

print(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# saving model to the disk
pickle.dump(classifier,open("model.pkl","wb"))

# loading model to compare the results
model=pickle.load(open("model.pkl","rb"))
print(classifier.predict([[35,4,0,1.8,250]])>0.5) # it means the heart of that particular will not fail over here

# from sklearn.metrics import r2_score
# r2_score(y_test,y_pred) # it means our decision tree algoruthm is best for this data as per this result