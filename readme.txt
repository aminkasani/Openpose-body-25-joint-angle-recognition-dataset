
Predict joint angle of body parts based on sequence pattern recognition tester source and dataset:

if you want to use this dataset, you have to cite:
https://ieeexplore.ieee.org/document/9721801

to test models you have to put all csv and npy files to data folder and select joint number in modelTester.py line 10
xtrain[1-12].npy contain real-world image OpenPose output and
Ytrain.csv is its label.
xntrain[0-10].npy contain 3d human model image OpenPose output and
Yctrain.csv is its label.
xtest[1-3].npy contain real-world test image OpenPose output and
Ytest.csv is its label.
