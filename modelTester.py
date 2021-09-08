import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

#el 0,er 1,sl 2,sr 3,sl2 4,sr2 5,kl 6,lr 7,nt 10,nbl 11,nbr 12,nf 13,ttr 14,ttl 15,tb 16,tf 17
joint = 17

def fixposition(newX):
    dif = [0, 0]
    if set(newX[8]) == set([0, 0]):
        if set(newX[9]) == set([0, 0]) or set(newX[12]) == set([0, 0]):
            return newX
        else:
            dif = [0.5, 0.5]-[(newX[9][0]+newX[12][0])/2,
                              (newX[9][1]+newX[12][1])/2]
    if set(dif) == set([0, 0]):
        dif = [0.5, 0.5]-newX[8]
    output = newX+dif
    return output


#Xdata=[]
#Ydata=[] #0 healthy 1 all
trainNames = ["data/xtrain1.npy", "data/xtrain2.npy", "data/xtrain3.npy", "data/xtrain4.npy",
              "data/xtrain5.npy", "data/xtrain6.npy", "data/xtrain7.npy", "data/xtrain8.npy",
              "data/xtrain9.npy", "data/xtrain10.npy", "data/xtrain11.npy", "data/xtrain12.npy",
              ]
traincNames = ["data/xntrain0.npy", "data/xntrain1.npy", "data/xntrain2.npy", "data/xntrain3.npy",
               "data/xntrain4.npy", "data/xntrain5.npy", "data/xntrain6.npy", "data/xntrain7.npy",
               "data/xntrain8.npy", "data/xntrain9.npy", "data/xntrain10.npy",

               ]
xdata = np.array([np.load(fname) for fname in trainNames])
Xdata = xdata[0]
for i in range(xdata.shape[0]-1):
    Xdata = np.vstack((Xdata, xdata[i+1]))


xcdata = np.array([np.load(fname) for fname in traincNames])
Xcdata = xcdata[0]
for i in range(xcdata.shape[0]-1):
    Xcdata = np.vstack((Xcdata, xcdata[i+1]))

Xcdata = Xcdata[:, :, 0:2]


Xdata = np.vstack((Xdata, Xcdata))

testNames = ["data/xtest1.npy", "data/xtest2.npy", "data/xtest3.npy"]
xtest = np.array([np.load(fname) for fname in testNames])
Xtest = xtest[0]
for i in range(xtest.shape[0]-1):
    Xtest = np.vstack((Xtest, xtest[i+1]))



Ytrain = pd.read_csv("data/Ytrain.csv", header=None)
Ytrain = Ytrain.to_numpy()
Yctrain = pd.read_csv("data/Yctrain.csv", header=None)
Yctrain = Yctrain.to_numpy()
Ytrain = np.vstack((Ytrain, Yctrain))
Ytest = pd.read_csv("data/Ytest.csv", header=None)
Ytest = Ytest.to_numpy()

Ytrain = Ytrain[:, joint]
Ytest = Ytest[:, joint]
Ytrain = Ytrain.reshape(Ytrain.shape[0], 1)
Ytest = Ytest.reshape(Ytest.shape[0], 1)
# print(Ytrain.shape)
# print(Ytest.shape)
# print(Xdata.shape)
# print(Xtest.shape)


Xdata = Xdata/6000
Ytrain = Ytrain/180
Xtest = Xtest/6000
Ytest = Ytest/180





def mover(points, xnumber, ynumber):
    a = points[:, 0]
    if np.min(a[np.nonzero(a)]) < abs(xnumber) and xnumber < 0:
        return points
    a = points[:, 1]
    if np.min(a[np.nonzero(a)]) < abs(ynumber) and ynumber < 0:
        return points
    for i in range(points.shape[0]):
        if points[i, 0] != 0 and points[i, 1] != 0:
            points[i, 0] = points[i, 0]+xnumber
            points[i, 1] = points[i, 1]+ynumber
    return points


def jointConnector(points):
    skeleton = []
    skeleton.append([points[9], points[10]])
    skeleton.append([points[10], points[11]])
    skeleton.append([points[8], points[12]])
    skeleton.append([points[12], points[13]])
    skeleton.append([points[13], points[14]])
    skeleton.append([points[15], points[16]])
    skeleton.append([points[17], points[18]])
    skeleton.append([points[17], points[2]])
    skeleton.append([points[18], points[5]])
    skeleton.append([points[1], points[8]])
    skeleton.append([points[8], points[9]])
    skeleton.append([points[9], points[12]])
    skeleton.append([points[10], points[13]])
    skeleton.append([points[2], points[4]])
    skeleton.append([points[2], points[9]])
    skeleton.append([points[5], points[12]])
    skeleton.append([points[5], points[7]])
    skeleton.append([points[1], points[2]])
    skeleton.append([points[2], points[3]])
    skeleton.append([points[3], points[4]])
    skeleton.append([points[1], points[5]])
    skeleton.append([points[5], points[6]])
    skeleton.append([points[6], points[7]])

    return np.array(skeleton)


x_train, y_train = shuffle(Xdata, Ytrain, random_state=13)



for i in range(x_train.shape[0]):
    newX = x_train[i]*0.60
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*0.70
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*0.80
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*0.90
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*0.95
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*1.05
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*1.10
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*1.20
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*1.30
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)
    newX = x_train[i]*1.40
    x_train = np.append(x_train, newX.reshape(1, 25, 2), axis=0)
    y_train = np.append(y_train, y_train[i].reshape(1, 1), axis=0)

# print(y_train.shape)
# print(x_train.shape)
print("zoom is done")
for i in range(x_train.shape[0]):
    newX = fixposition(x_train[i])
    x_train[i] = newX
for i in range(Xtest.shape[0]):
    newX = fixposition(Xtest[i])
    Xtest[i] = newX
print(x_train.shape)


print("preproccessing mover is done")
xtrc = []
for i in range(x_train.shape[0]):
    newX = jointConnector(x_train[i])
    xtrc.append(newX)
x_trainc = np.array(xtrc)

xtec = []
for i in range(Xtest.shape[0]):
    newX = jointConnector(Xtest[i])
    xtec.append(newX)
Xtestc = np.array(xtec)
print("joint connector is done")
x_train = x_trainc
Xtest = Xtestc


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = keras.models.load_model(
    './gg_model'+str(joint)+'.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error})

print(model.summary())




predictions = model.predict(x_train)
predictions = predictions*180
y_train = y_train*180
print("############################")
print("########## Result"+str(joint)+" ##########")
print("RMSE train:")
print(mean_squared_error(y_train, predictions,squared=False))
print("MAE train:")
print(mean_absolute_error(y_train, predictions))
##
##
predictions = model.predict(Xtest)
predictions = predictions*180
Ytest = Ytest*180
print("RMSE test:")
print(mean_squared_error(Ytest, predictions,squared=False))
print("MAE test:")
print(mean_absolute_error(Ytest, predictions))


