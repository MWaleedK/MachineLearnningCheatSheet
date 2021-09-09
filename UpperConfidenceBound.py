import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000 #totalViewers
d=10 #total Ads

ads_selected=[]

numbers_of_selections=[0]*d
sums_of_rewards=[0]*d

total_reward=0

for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if numbers_of_selections[i]>0:
            avgReward=sums_of_rewards[i]/numbers_of_selections[i]
            delta_i=math.sqrt((3/2)(math.log(n))/numbers_of_selections[i])
            upperBound=avgReward+delta_i
        else:
            upperBound=1e400
            if upperBound>max_upper_bound:
                max_upper_bound=upperBound
                ad=i
    ads_selected.append(ad)
    numbers_of_selections[ad]+=1
    reward=dataset.values[n,ad]
    sums_of_rewards[ad]+=reward
    total_reward+=reward

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Frequency')
plt.show()