##This file belongs in the lut 
import os
import sys
SelVariance = input("Enter variance (leave blank for default 15):")
try:
    float(SelVariance)
except ValueError:
    SelVariance = "15.0"
outFile = open("./ErrorInject.inl", 'w')
outFile.write("""

//
//Since I don't actually care about real multiplication, I need to define an INL that just adds some error to the normal multiplication. The error should be customizable. 
//My idea is to use the gaussian distribution; this allows for me to use the actually calculated product as the mean, and then a customizable variance.
//This WILL NOT speed up a DNN; this is not designed to speed up a DNN; this is just designed to introduce error.
#include <random>
//



//
float ErrorInject(float Af, float Bf, float variance=""" + SelVariance + """)
{
	float sum = Af * Bf;
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(sum, variance);
	return distribution(generator);

}
//

""")
outFile.flush()
os.system('bash lut_gen.sh')