import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from math import sqrt

data = pd.read_csv("iris.data")                                     # Iris table
chi_square_table = pd.read_csv("alpha_table.csv", delimiter=" ")    # Chi-square table
student_table = pd.read_csv("student_table.csv", delimiter=" ")     # Student's table
X = "petal_length"          # choose your parameter here
Y = "sepal_width"           # choose your parameter here
TYPE = "Iris-setosa"        # choose your parameter here

data = data[data["species"] == TYPE]
data = data.sort_values(by=X)
X = data[X].to_numpy()
Y = data[Y].to_numpy()

data = np.concatenate(([X], [Y]), axis=0)
data_copy = data.copy()
# ---------------------------------------------------------------------- 3.1 start


def calculate_sums(data):
    strings = np.sum(data, axis=1)
    columns = np.sum(data, axis=0)
    general_sum = np.sum(strings)
    dof = (int(len(strings)) - 1) * (int(len(columns)) - 1)
    return columns, strings, general_sum, dof


def calculate_statistics_of_criterion(data, columns, strings, n):
    dims = np.shape(data)
    sum = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            sum += float(data[i][j] * data[i][j]) / (columns[i] * strings[j])
    sum = n * (sum - 1)
    return sum


def create_partition(data, partitions):
    dims = np.shape(data)
    table = np.zeros((partitions, partitions))

    for i in range(dims[0]):
        min_ = np.min(data[i])
        for j in range(dims[1]):
            data[i][j] -= min_

    delta_arr = []

    for i in range(dims[0]):
        delta_arr.append(float(np.max(data[i])) / partitions)

    for i in range(dims[0] - 1):
        for j in range(dims[1]):
            ii = min(math.floor(data[i][j] / delta_arr[i]), partitions - 1)
            jj = min(math.floor(data[i+1][j] / delta_arr[i+1]), partitions - 1)
            table[ii][jj] += 1

    return table


def check_independence(data, alpha=0.01, partitions=5):
    # data = np.array([[170, 179, 187, 189, 193, 199, 200, 207, 215, 216, 220, 225],
    #                  [152, 159, 162, 165, 170, 172, 177, 179, 184, 186, 190, 191]])
    data = create_partition(data, partitions)
    columns, strings, n, dof = calculate_sums(data)
    res = calculate_statistics_of_criterion(data, columns, strings, n)
    critical_val = float(chi_square_table.iloc[dof - 1]['a' + str(alpha)].replace(',', '.'))

    print("result is", res)
    print("function is", critical_val)
    if res > critical_val:
        print("the values are NOT independent")
    else:
        print("the values are independent")
    return n

# ---------------------------------------------------------------------- 3.1 end

# ---------------------------------------------------------------------- 3.2 start


def sample_average(data, dimension, power=1):
    dims = np.shape(data)
    s = 0
    for j in range(dims[1]):
        s += math.pow(data[dimension][j], power)
    return float(s) / dims[1]


def x_(data):
    return sample_average(data, 0)


def y_(data):
    return sample_average(data, 1)


def x2_(data):
    return sample_average(data, 0, 2)


def y2_(data):
    return sample_average(data, 1, 2)


def xy_(data):
    sum = 0
    dims = np.shape(data)
    for i in range(dims[0] - 1):
        for j in range(dims[1]):
            sum += data[i][j] * data[i+1][j]
    return float(sum) / dims[1]


def s_x(x2_, x_2):
    return math.sqrt(float(x2_) - x_2)


def s_y(y2_, y_2):                          # copy of s_x, just for convenience
    return math.sqrt(float(y2_) - y_2)


def s_xy(xy_, x_, y_):                      # ratio of covariance
    return xy_ - (x_ * y_)


def r_xy(s_xy, s_x, s_y):
    return float(s_xy)/(s_x * s_y)


def ratio_of_covariance_and_correlation(data):
    x_average = x_(data)
    y_average = y_(data)
    x2_average = x2_(data)
    y2_average = y2_(data)
    xy_average = xy_(data)
    x_2 = math.pow(x_average, 2)
    y_2 = math.pow(y_average, 2)
    Sx = s_x(x2_average, x_2)
    Sy = s_y(y2_average, y_2)
    covariance = s_xy(xy_average, x_average, y_average)     # covariance
    r = r_xy(covariance, Sx, Sy)                            # correlation
    print("correlation ", r)
    print("covariance ", covariance)
    return r, covariance


def distribution_of_statistics(r, n, alpha=0.01):
    T = float(r)*(math.sqrt(n)) / (math.sqrt(1 - math.pow(r, 2)))
    t = None
    if n > 31:
        t = float(student_table.iloc[30]['y' + str(1 - alpha)].replace(',', '.'))
    else:
        t = float(student_table.iloc[n - 1]['y' + str(1 - alpha)].replace(',', '.'))
    print("T == ", T)
    print("t == ", t)
    if (T > t):
        print("the hypothesis of the insignificance \n of the linear correlation coefficient is REJECTED")
    else:
        print("the hypothesis of the insignificance \n of the linear correlation coefficient is ACCEPTED")
    return T, t

# ---------------------------------------------------------------------- 3.2 end

# ---------------------------------------------------------------------- 3.3 start


def linear_regression(data, alpha):
    X = data[0]
    Y = data[1]
    interval = [np.min(X) - 0.1, np.max(X) + 0.1]
    n = X.size
    m = 1
    x_sr = X.sum() / n
    y_sr = Y.sum() / n
    # calculate Beta0 and Beta1 ratios
    b1 = (X * Y - n * x_sr * y_sr).sum() / (X ** 2 - n * (x_sr ** 2)).sum()
    b0 = y_sr - b1 * x_sr

    print("Beta0 = ", b0, ", Beta1 = ", b1)

    plt.title('Predict Y by X')
    plt.plot(X.tolist(), Y.tolist(), 'ro')
    plt.plot(interval, [interval[0] * b1 + b0, interval[1] * b1 + b0])
    plt.show()

    RSS = ((Y - (X * b1 + b0)) ** 2).sum()
    TSS = ((X - y_sr) ** 2).sum()

    r2 = 1 - RSS / TSS
    print("R^2 == ", r2)

    F = (r2 * (n - 1 - 1)) / ((1 - r2) * 1)
    fish = sps.f.ppf((1 - alpha), m, n - m - 1)

    print("F == ", F)
    print("Fisher's criterion == ", fish)
    if F > fish:
        print("The hypothesis is REJECTED at the significance level alpha = ", (1 - alpha) * 100, "%")
    else:
        print("The hypothesis is ACCEPTED at the significance level alpha = ", (1 - alpha) * 100, "%")

    return F, fish

# ---------------------------------------------------------------------- 3.3 end


print("-----------------------------------------------------------------------------------")
check_independence(data, 0.05)
print("-----------------------------------------------------------------------------------")
data = data_copy.copy()
correlation, covariance = ratio_of_covariance_and_correlation(data)
distribution_of_statistics(correlation, np.shape(data)[1] - 2, 0.05)
print("-----------------------------------------------------------------------------------")
data = data_copy.copy()
F, fish = linear_regression(data, 0.05)
print("-----------------------------------------------------------------------------------")
