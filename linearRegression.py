# Linear Regression code & example #

# m = sum(x-xmean)(y-ymean)/(x-xmean)**2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [3,4,2,4,5]

x_mean = np.mean(x)
y_mean = np.mean(y)

# make dataframe and zip x and y columns together; then conv to float
df_linear = pd.DataFrame(list(zip(x,y)),columns=['x','y'])
df_linear['x'] = df_linear['x'].astype(float)
df_linear['y'] = df_linear['y'].astype(float)

# subtract the x and y means from the df...
xMinusxMean = df_linear['x']-x_mean
yMinusyMean = df_linear['y']-y_mean
df_linear['xMinusxMean'] = xMinusxMean
df_linear['yMinusyMean'] = yMinusyMean

# xMinusMean times yMinusMean (numer before summed)
xyMeanMulp = df_linear['xMinusxMean']*df_linear['yMinusyMean']
df_linear['xyMeanMulp'] = xyMeanMulp

# sqr of x-xMean (denom before summed)
xMinusxMeanSqrd = (df_linear['x']-x_mean)**2
df_linear['xMinusxMeanSqrd'] = xMinusxMeanSqrd

# calculate m (slope)
m = (np.sum([df_linear['xyMeanMulp']])/np.sum([df_linear['xMinusxMeanSqrd']]))

"""
## With m determined, we can figure out b (y-intercept) by using y-mean as y in and x_mean as x:
## y = mx + b 
## y_mean = m(x_mean) + b
## 3.6 = 0.4(3) + b
## 3.6 = 1.2 + b
## b = 3.6-1.2
## b = 2.4
"""

## b codified
b = (y_mean-(m*(x_mean)))

################### f(x) = mx + b #######################
################### linear prediction ###################

linear_y_prediction = (m*df_linear['x']) + b
df_linear['linear_y_prediction'] = linear_y_prediction


# r-squared
# sum(yp-y_mean)**2 / sum(y-y_mean)**2

r2_numer = round(np.sum((df_linear['linear_y_prediction']-y_mean)**2),2)
r2_denom = round(np.sum((df_linear['y']-y_mean)**2),2)
r_squared = round(r2_numer/r2_denom,2)


####################################################################################

print("\nThe number of observations is:",len(df_linear['x']),
      "\nThe x_mean is:",x_mean,
      "\nThe y_mean is:",y_mean,
      "\nR-Squared value of predictions is:",r_squared)


""" writing to csv
df_linear.to_csv("C:/~df_linear.csv")
"""


plt.plot(df_linear['x'],df_linear['linear_y_prediction'],color="#58b970",label="Regression Line")
plt.scatter(df_linear['x'],df_linear['y'],color="#ef5423",label="Scatter Plot")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

















