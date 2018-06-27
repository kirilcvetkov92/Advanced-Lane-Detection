## Import the packages
import numpy as np
from scipy import stats


## Define 2 random distributions
#Sample Size
N = 30
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N)
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)

## Import the packages
import numpy as np
from scipy import stats
t2, p2 = stats.ttest_ind(a,b)
t=str(2*p2)
print("t = " + str(t2))
print("p = " + str(2*p2))