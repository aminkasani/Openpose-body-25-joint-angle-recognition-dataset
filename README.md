# Predict joint angle of body parts based on sequence pattern recognition tester source and dataset:


if you want to use this dataset, you have to cite:
https://ieeexplore.ieee.org/document/9721801


to test models you have to put all csv and npy files to data folder and select joint number in modelTester.py line 10<br/>
A dataset created by output of OpenPose 25-body model and joint angle label<br/>
xtrain[1-12].npy contain real-world image output and<br/>
Ytrain.csv is its label.<br/>
xntrain[0-10].npy contain 3d human model image output and<br/>
Yctrain.csv is its label.<br/>
xtest[1-3].npy contain real-world test image output and<br/>
Ytest.csv is its label.<br/>
