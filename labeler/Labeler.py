import numpy as np
import pandas as pd


'''
Lookback is defined in seconds
Cutoffs are defined as 
Each second is ~ 30 rows in the CSV that we read (but is not)

WINDOW_SIZE = determines how much to be averaging the data that we have
DELTA_ANGLE: delta = (end - start) % 360
DIRECTION: minimum of either CW or CCW 

'''
class Labeler(object):
    def __init__(self, lookback=90, lookforward=120, window_size=10, cutoffs_hard=[70, 100], cutoffs_slight=None):
        self.lookback = lookback
        self.window_size = window_size
        self.lookforward = lookforward
        
        self.hard_left_cutoffs = [(360-cutoffs_hard[1]), (360-cutoffs_hard[0])]
        self.hard_right_cutoffs = cutoffs_hard

        
    def check_classifier_direction(self, angle):
        if angle < self.hard_right_cutoffs[1] and angle > self.hard_right_cutoffs[0]:
            return 1
        elif angle < self.hard_left_cutoffs[1] and angle > self.hard_left_cutoffs[0]:
            return -1
        return 0

    '''
    There is NaN at the beginning of dataset that is set to 0. Should it be set to 0?
    '''
    def check_regression_direction(self, angle):
        if angle <= self.hard_right_cutoffs[1]: #right
            return angle / self.hard_right_cutoffs[1]
        elif angle >= self.hard_left_cutoffs[0]: # left
            return (angle - 360) / self.hard_right_cutoffs[1]
        elif angle is None:
            return 0
        elif angle > self.hard_right_cutoffs[1] and angle < self.hard_left_cutoffs[1]:
            return 0
        return 0

    '''
    INPUT: List of fields to process
    OUTPUT: Processed DataFrame with Direction for hard turns and also slight turns
    '''
    def add_direction(self, df_path="", fields="Heading (degrees)"):
        df = pd.read_csv(df_path)
        sub_df = df[fields]
        averaged_vals = sub_df.rolling(self.window_size, closed="left", center=False).mean()
        
        # shift the values forward and backwards to create the lookback period
        backwards_vals = averaged_vals.shift(periods=self.lookback)
        forwards_vals = averaged_vals.shift(-1 * self.lookforward)
        
        # calculate the change in angle
        heading_delta = (forwards_vals - backwards_vals) % 360
        
        # convert to a hard right or left turn
        
        # TODO: Vectorize this quick
        # if labeler_regression_tag is False:
        #     x = heading_delta.apply(lambda angle: self.check_classifier_direction(angle))
        # else:
        #     x = heading_delta.apply(lambda angle: self.check_regression_direction(angle))

        def get_heading(angle):
            return angle

        classifier_direction = heading_delta.apply(lambda angle: self.check_classifier_direction(angle))
        regression_direction = heading_delta.apply(lambda angle: self.check_regression_direction(angle))
        angle = heading_delta.apply(lambda angle: get_heading(angle))
        df["direction"] = classifier_direction
        df["regression_direction"] = regression_direction
        df["angle"] = angle
        df["back_avg"] = backwards_vals
        df["forward_avg"] = forwards_vals
        return df