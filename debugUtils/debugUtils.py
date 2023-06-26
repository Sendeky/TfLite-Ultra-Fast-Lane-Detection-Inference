import matplotlib.pyplot as plt
from scipy.signal import savgol_filter    


# plots smooth data using savgol_filter 
def plotSmoothData(d1, d2, win_len):
    x_filtered = d1[["x"]].apply(savgol_filter,  window_length=win_len, polyorder=2)
    print("x_filtered: ", x_filtered)
    plt.ion()
    # plt.plot(d2)
    plt.plot(x_filtered)
    plt.show()