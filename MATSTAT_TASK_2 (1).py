import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
import math

# alpha == 0.01 => kolmogorov_statistics == 1.0599
kolmogorov_statistics = 1.0599                  # choose your parameters here


data = pd.read_csv("iris.data")
data = data[data["species"] == "Iris-setosa"]   # choose your parameters here
data = data["petal_width"]                      # choose your parameters here

data.to_csv("ans", index=False)

def Sk_func(data, mean, var):
    sort_data = sorted(data)

    refactor = 0
    for i, x in enumerate(sort_data):
        refactor = max(refactor, i / len(sort_data) - sps.norm(loc=mean, scale=var).cdf(x))

    d_minus = 0
    for i, x in enumerate(sort_data):
        d_minus = max(d_minus, sps.norm(loc=mean, scale=var).cdf(x) - (i - 1) / len(data))
    Dn = max(d_minus, refactor)
    sk = (6 * len(data) * Dn + 1) / (6 * math.sqrt(len(data)))
    return sk


def show_info(data):
    # show a graph of the empirical distribution function
    sns.ecdfplot(data=data)
    plt.show()

    # show a histogram
    sns.histplot(data)
    plt.show()

    # show the nuclear estimate of the density function
    sns.kdeplot(data, bw_adjust=0.5)
    plt.show()

mean = data.mean()

# sample variance
sample_var = 0
for x in data:
    sample_var += (x - mean) ** 2
sample_var /= len(data)

# sample unbiased variance
sample_unbiased_var = 0
for x in data:
    sample_unbiased_var += (x - mean) ** 2
sample_unbiased_var /= (len(data) - 1)

# minimum sample statistics
min_sample_stat = data.min()

# maximum sample statistics
max_sample_stat = data.max()
scope = max_sample_stat - min_sample_stat

# median
var_series = data.sort_values()

# show results
median = (var_series[len(data) / 2] + var_series[len(data) / 2 + 1]) / 2
print("sample average: ", mean)
print("sample variance: ", sample_var)
print("sample unbiased variance: ", sample_unbiased_var)
print("minimum sample statistics: ", min_sample_stat)
print("maximum sample statistics: ", max_sample_stat)
print("scope: ", scope)
print("median: ", median)

show_info(data)
# is the normal distribution
Sk = Sk_func(data, mean, math.sqrt(sample_var))

print(" Sk: ", Sk)

if Sk <= kolmogorov_statistics:
    print("Is a normal distribution.")
else:
    print("Is not a normal distribution.")

