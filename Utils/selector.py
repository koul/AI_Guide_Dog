#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd


'''
Lookback
'''
class LabelDF(object):
    def __init__(self, input_data, lookback=3, cutoff=1):
        self.df = pd.read_csv(input_data)
        self.lookback = lookback
        self.cutoff = cutoff
        
    '''
    INPUT: List of fields to process
    OUTPUT: Processed DataFrame with Direction

    '''
    def add_direction(self, fields):
        sub_df = self.df[fields]
        windows = sub_df.rolling(self.lookback, closed="right").mean()
                
        ## perform the subtraction
        thresholds = sub_df - windows
        thresholds = thresholds.sum(axis=1)

        directions_positive = (thresholds > self.cutoff).astype(int)
        directions_negative = (thresholds < -1 * self.cutoff).astype(int)
        directions = directions_positive + directions_negative
        
        self.df["directions"] = directions
        print(sum(np.abs(directions)))
        
        return self.df
    
    def underSample_balancer(self):
        dropped_df = self.df.groupby("direction")
        self.df = dropped_df.apply(lambda x: x.sample(dropped_df.size().min())) 
#         display(self.df)
        
    def overSample_balancer(self):
        max_size = self.df['direction'].value_counts().max()
        lst = [self.df]
        for class_index, group in self.df.groupby('direction'):
            lst.append(group.sample(max_size-len(group), replace=True))
        self.df = pd.concat(lst)
        
        display(self.df)
    


# In[63]:


df_class = LabelDF("test_directional_data.csv", lookback=5)
# df_class.add_direction(["Magnetometer - X axis","Magnetometer - Y axis"])
# df_class.underSample_balancer()
df_class.overSample_balancer()


# In[ ]:




