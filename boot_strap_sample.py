import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def bootstrap(x, resamples=10000):
    
    """Draw bootstrap resamples from the array x.

    Parameters
    ----------
    x: np.array, shape (n, )
      The data to draw the bootstrap samples from.
    
    resamples: int
      The number of bootstrap samples to draw from x.
    
    Returns
    -------
    bootstrap_samples: np.array, shape (resamples, n)
      The bootsrap resamples from x.
    """
    result_list = []
    for i in range(resamples):
        bs_sample = np.random.choice(x, size = len(x), replace = True)
        result_list.append(np.array(bs_sample))
    return(np.array(result_list).reshape(resamples, len(x)))


def bootstrap_sample_medians(data, n_bootstrap_samples=10**4):
    bootstrap_sample_medians = []
    for i in range(n_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_sample_medians.append(np.median(bootstrap_sample))
    return bootstrap_sample_medians

def bootstrap_ci(sample, stat_function=np.mean, resamples=10000, ci=95):
    boot_samples = bootstrap(sample, resamples = resamples)
    
    boot_test_stat_list = []
    for i in range(resamples):
        boot_samples = bootstrap(sample, resamples = 1)
        boot_test_stat_list.append(stat_function(boot_samples))
   
    
    ci_levels = (100-ci)/2
    up = 100- ci_levels
    low = ci_levels
    
    return(boot_test_stat_list,np.percentile(boot_test_stat_list, q=[low, up]) )  

def plot_bootstrap_mean(data, test_statistic = np.mean, confidence_interval = 95):

    means, ci = bootstrap_ci(data, stat_function = test_statistic, ci =confidence_interval)


    fig, ax = plt.subplots(1, figsize=(12, 4))
    ax.hist(data, bins=25, density=True, color="black", alpha=0.6,
            label="SPINS recorded ARP by Geography")
    ax.hist(means, bins=25, density=True, color="red", alpha=0.75,
            label="Estimated True Mean Price")
    ax.legend()
    # ax.tick_params(axis='both', which='major', labelsize=15)
    _ = ax.set_title("Estimation of True Mean Price(10000 samples)", fontsize = 20)

def plot_bootstrap_median(data, test_statistic = np.median, confidence_interval = 95):

    means, ci = bootstrap_ci(data, stat_function = test_statistic, ci =confidence_interval)


    fig, ax = plt.subplots(1, figsize=(12, 4))
    ax.hist(data, bins=25, density=True, color="black", alpha=0.6,
            label="SPINS recorded ARP by Geography")
    ax.hist(means, bins=25, density=True, color="red", alpha=0.75,
            label="Estimated True Median Price")
    ax.legend()
    # ax.tick_params(axis='both', which='major', labelsize=15)
    _ = ax.set_title("Estimation of True Median Price (10000 samples)", fontsize = 20)
