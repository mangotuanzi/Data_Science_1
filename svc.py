import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,train_test_split

y=np.load('label.npy')


x=np.load('feature.npy')
#Gvals = [0.01,0.02,0.025,0.03,0.04]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
#for G in Gvals:,
model1 = SVC(kernel='rbf', class_weight='balanced',C=8,gamma=0.0005)
model1.fit(x_train,y_train)
score=model1.score(x_test,y_test)
print ("rbf AUC: %f" %(score))
model2 = SVC(kernel='linear', class_weight='balanced',C=8)
model2.fit(x_train,y_train)
score2=model2.score(x_test,y_test)
print ("linear AUC: %f" %(score2))
model3 = SVC(kernel='poly', class_weight='balanced',C=8)
model3.fit(x_train,y_train)
score3=model3.score(x_test,y_test)
print ("poly AUC: %f" %(score3))
model4 = SVC(kernel='poly', class_weight='balanced',C=8)
model4.fit(x_train,y_train)
score4=model4.score(x_test,y_test)
print ("sigmoid AUC: %f" %(score4))
##find the best c and gamma

'''
score_hist = []
Cvals = np.logspace(-5, 15, 11, base=2)
# cvals = np.logspace(-5, 15, 11, base=2)
  gammavals= np.logspace(-9, 3, 13, base=2)
for C in Cvals:
    model.C = C
    
    score = cross_val_score(model,x,y,cv=3,n_jobs=-1)
    print('fns1')
    score_hist.append((score,C))
    print ("C: %f Mean AUC: %f" %(C, score))
bestC = sorted(score_hist)[-1][1]
print ("Best C value: %f" % (bestC))
'''
