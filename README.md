# 511Project
Final DNN Error Resiliency project for ECE-511

Files:

The "FullFramework.py" file is designed to be placed in the ApproxTrain folder. Running this script generates a C++ multiplyer with customizable variance, (as well as lut_gen.sh and lut_gen.cc) and causes ApproxTrain to re-generate the lookup tables of all multipliers so that the new multiplier can be utilized. It then runs the generated multiplyer on a very simple CNN with two 2D layers and two dense layers. 

All of the functionality of the "FullFramework.py" file is broken down into other constituent files, in case it is undesirable to run all portions of this file. NNGenerator.py generates the aforementioned CNN and tests it with the LUT of the generated multiplyer (in a file called "injected_error.py"). ErrorGen.py should be run in the lut folder, and generates the c++ multiplyer, and attempts to run the lookup table generator. However, this generation will fail without the updated lut_gen.cc and lut_gen.sh also placed in the lut folder.
