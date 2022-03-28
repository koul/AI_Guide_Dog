import pandas as pd


'''
Duplicates or removes data such that there are equal number of labels in each category.
'''


class Selector(object):

    def __init__(self, input_data):
        self.df = pd.read_csv(input_data)


    def underSample_balancer(self):
        dropped_df = self.df.groupby("direction")
        self.df = dropped_df.apply(lambda x: x.sample(dropped_df.size().min())) 

    def overSample_balancer(self):
        max_size = self.df['direction'].value_counts().max()
        lst = [self.df]
        for class_index, group in self.df.groupby('direction'):
            lst.append(group.sample(max_size-len(group), replace=True))
        self.df = pd.concat(lst)

    





