import scipy
import scipy.optimize
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.mlab as mlab
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import h5py
import json
import uncertainties

plt.rcParams["figure.figsize"] = (3, 3)


class Experiment:
    def __init__(self, FileName=None):
        self.FileName = FileName
        self.f = h5py.File(FileName, 'r')
        self.keys = list(self.f.keys())
        self.archive_keys = list(self.f['archive'].keys())
        self.dataset_keys = list(self.f['datasets'].keys())
        self.scan_keys = list(self.f['datasets/scan/product'].keys())

        self.artiq_version = str(self.f['artiq_version'][()])

        ### Display Ion within Region of Interest (ROI) ###

    def archive(self):
        this = []
        for i in self.archive_keys:
            string = 'archive/' + i
            this.append(self.f[string][()])
        self.archive_info = pd.DataFrame(data=this, columns=['Data'], index=archive_keys)
        return self.archive_info

    def data(self):
        scan_strings = list(self.f['datasets/scan/product'].keys())
        rows = []

        scan_calls = []
        for i in scan_strings:
            scan_calls.append('datasets/scan/product/' + i)

        for i in range(len(list(self.f[scan_calls[0]]))):
            input = []
            for j in scan_calls:
                input.append(self.f[j][i])
            rows.append(input)

        result_strings = []
        for i in list(self.f['datasets/'].keys()):
            if i in ['dax', 'histogram_context', 'scan']:
                pass
            else:
                result_strings.append(i)

        results = []
        for j in result_strings:
            results = list((self.f['datasets/' + j][()]))
            for i in range(len(rows)):
                rows[i].append(float(results[i]))

        column_strings = scan_strings + result_strings

        self.data = pd.DataFrame(data=rows, columns=column_strings)
        return self.data

    def quick_plot(self):
        for i in self.scan_keys:
            for j in self.data.columns:
            plt.plot(self.data[i], self.data[j])
            plt.xlabel(i)
            plt.ylabel(j)
            plt.show()