import numpy as np
import pandas as pd

__author__ = 'Michael Silverstein'

def generate_feature(means, stds, n, labels, feature=None, seed=None, MIN=None, MAX=None):
    """
    Generate feature data assuming features are normally distributed.
    For each input parameter, either a list (associated with the order of `labels`) or a single value can be passed. 
        Passing a single value will assume that all classes share that parameter
    Inputs:
    | means: <list> or <float> Mean(s) of feature distribution
    | stds: <list> or <float> Standard deviation(s) of feature distribution
    | n: <list> or <float> Number of samples for each class
    | labels: <array> List of labels (this order is associated with the other passed parameters)
    | feature: <str> Name of feature
    | seed: <int> Random seed for sampling
    | {MIN, MAX}: <float> Minimum and maximum thresholds
    Output:
    | data: <dataframe> || `feature` | class ||
    """
    # If any inputs are single values, convert them to lists
    if np.isscalar(means):
        means = [means]*len(labels)
    if np.isscalar(stds):
        stds = [stds]*len(labels)
    if np.isscalar(n):
        n = [n]*len(labels)
    # Assign parameters to each class
    params = {label: {'mean': m, 'std': s, 'n': size} for label, m, s, size in zip(labels, means, stds, n)}
    
    # Generate data
    if seed:
        np.random.seed(seed)
    ## For each class, sample `n` points from a normal distribution with that classes mean and standard deviation
    data = [[x, label] for label in labels for x in np.random.normal(params[label]['mean'], params[label]['std'], params[label]['n'])]
    
    # Place into dataframe
    if not feature:
        feature = 'feature'
    data = pd.DataFrame(data, columns=[feature, 'class'])
    # Apply thresholds
    data.loc[data[feature]>=MAX, feature] = MAX
    data.loc[data[feature]<=MIN, feature] = MIN
    return data

def rescale(x, axis=0):
    """
    Rescale a matrix X between [0, 1] along `axis`
    """
    rescaled = (x - x.min(axis))/(x.max(axis) - x.min(axis))
    return rescaled