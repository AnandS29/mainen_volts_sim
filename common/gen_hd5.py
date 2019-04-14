# see checkpoint
import yaml
import matplotlib.pyplot as plot
import numpy as np
from numpy import genfromtxt
import csv
import struct
import numpy
import pandas as pd
import h5py
nrnTypes = {
    2: numpy.short,
    3: numpy.float32,
    4: numpy.double,
    5: numpy.int,
}

subsequence_length = 4

def find_peaks(lis, min_value):
	inds = [i for i in range(1, len(lis) - 1) if lis[i] > lis[i - 1] and lis[i] >= lis[i + 1] and lis[i] >= min_value]
	return [lis[i] for i in inds], inds

# add min_value parameter to eliminate baselines below certain range
def find_baselines(lis, max_value, min_value):
	inds = [i for i in range(1, len(lis) - 1) if lis[i] < lis[i - 1] and lis[i] <= lis[i + 1] and lis[i] <= max_value and lis[i] >= min_value]
	return [lis[i] for i in inds], inds

# tolerate one exception
# need difference satisfy at least 1 
def if_decreasing(lst):
    abnormal = 0
    for i in range(len(lst) - 1):
        if lst[i] - lst[i + 1] < 1:
            abnormal += 1
    return abnormal < 2

# tolerate two exceptions
def if_increasing(lst):
    abnormal = 0
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            abnormal += 1
    return abnormal < 3

def find_monotonous_subsequence(values, f):
    i = 0
    start_index_list = []
    while i + subsequence_length <= len(values):
        if f(values[i: i + subsequence_length]):
            start_index_list.append(i)
        i = i + subsequence_length
    return start_index_list

def apply_to_subesquences(start_indexs, lst, f):
    return any([f(lst[i: i + subsequence_length]) for i in start_indexs])

# find the number of exceptions
def if_compact(lst):
    compactness = 300
    abnormal = 0
    for i in range(len(lst) - 1):
        if abs(lst[i] - lst[i + 1]) > compactness:
            abnormal += 1
    return abnormal

def find_action_potential(v, hpeaks, m):
    '''peaks_indexs = hpeaks[1]
    depth = m + 70
    potential_threshold = 1 / 2
    for i in peaks_indexs:
        ratio = max((v[i] - v[i - 1]) / depth, (v[i] - v[i + 1]) / depth)
        if ratio > potential_threshold:
            action_potential_indexs.append(i)
    return len(action_potential_indexs)'''
    action_potential_indexs = [k for k in range(len(hpeaks[1])) if hpeaks[0][k] > -5]
    #action_potential_indexs = []
    #for k in range(len(hpeaks[1])):
    #    if hpeaks[0][k] > -5:
            # fatal range
     #       if k < 500 or False:
     #           return -1
     #       action_potential_index.append(k)        
    return len(action_potential_indexs)

def id_maker(k, identification):
    '''if k // 10 == 0:
        k = '0000' + str(k)
    elif k // 100 == 0:
        k = '000' + str(k)
    elif k // 1000 == 0:
        k = '00' + str(k)
    elif k // 10000 == 0:
        k = '0' + str(k)'''
    return identification + '_' + str(k)

def label_ntraces(traces, iterable, identification):
    labels = []
    for i in iterable:
        volts = traces[i]
        # key "voltage" deleted
        label_and_action_potential = label(volts)
        labels.append([id_maker(i, identification), label_and_action_potential[0], label_and_action_potential[1]])
    return labels
def label_mutliple_ntraces(traces, iterable, identification):
    labels = []   
    for i in iterable:
        for vind in range(len(volts)):
            currvolts = traces[vind]
            
        # key "voltage" deleted
        label_and_action_potential = label(volts)
        labels.append([id_maker(i, identification), label_and_action_potential[0], label_and_action_potential[1]])
    return labels

def save_to_csv(res, csvfile):
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(res)

def readCSV(fileName, delim):
    df = np.genfromtxt(fileName, delimiter=delim)
    paramsList = [tuple(x) for x in df]
    return paramsList

result = []
pin = 30000

def label(volts):
    peaks = find_peaks(volts, -70)[0]
    #500-3k and 13k-15.5k are bad
    if peaks:
        maximum = max(peaks)
    else:
        maximum = -60
            
    if maximum > -20:
        helper_peaks = find_peaks(volts, -40)
        peaks_number = len(helper_peaks[0])
        '''if peaks_number < 3:
            return 1'''
        width = 3000
        refined_peaks = [helper_peaks[0][i] for i in range(len(helper_peaks[1])) if
                         helper_peaks[1][i] - helper_peaks[1][0] < width]
        refined_peaks_indexs = [helper_peaks[1][i] for i in range(len(helper_peaks[1])) if
                         helper_peaks[1][i] - helper_peaks[1][0] < width]
        maximum = max(refined_peaks)
        minimum = min(refined_peaks)
        difference = maximum - minimum
        peaks_length = len(refined_peaks)
        
        # checkpoint
        # set min to -80 here, modify if need
        baselines = find_baselines(volts, 0, -80)
        refined_baselines = [baselines[0][i] for i in range(len(baselines[1])) if
                             baselines[1][i] - baselines[1][0] < width]
        refined_baselines_indexs = [baselines[1][i] for i in range(len(baselines[1])) if
                             baselines[1][i] - baselines[1][0] < width]
        baselines_length = len(refined_baselines)

        scaled_upper_bound = 40
        scaled_lower_bound = 20
        peaks_decrease_start = find_monotonous_subsequence(refined_peaks, if_decreasing)
        weird = peaks_length >= 4 and len(peaks_decrease_start) > 0 and baselines_length >= subsequence_length and\
        any([if_compact(refined_peaks_indexs[peaks_decrease_start[i]: peaks_decrease_start[i] + subsequence_length]) < 3 and
             if_increasing(refined_baselines[peaks_decrease_start[i]: peaks_decrease_start[i] + subsequence_length]) and
             if_compact(refined_baselines_indexs[peaks_decrease_start[i] + 1: peaks_decrease_start[i] + subsequence_length]) < 2
             for i in range(len(peaks_decrease_start))])
        '''if difference < scaled_upper_bound and difference > scaled_lower_bound and \
                peaks_length > 2 and peaks_length < 6 and if_decreasing(refined_peaks[: 4]):'''
        number_of_action_potentials = find_action_potential(volts, helper_peaks, maximum)
        #if number_of_action_potentials == -1:
            # fatal range
         #   return -1, -1
        #depolarization block     
        for p in find_peaks(volts, -70)[1]:
            if (p>4000 and p<5400):
                return -3, -1
            if (p>20500 and p<22000):
                return -4, -1
        if number_of_action_potentials > 30:
            return -2, number_of_action_potentials
        if (difference > scaled_upper_bound and weird):
            return -1, number_of_action_potentials
        if difference < scaled_upper_bound and difference > scaled_lower_bound and weird:
            #return 0.7, number_of_action_potentials
            return 1, number_of_action_potentials
        else:
            return 1, number_of_action_potentials
        
    if maximum < -40:
        return 0, 0
    else:
        return 0, 0

def gen_data(stims_path, modelFolder, params_name, psize, pMatx, pSetsN, version, i = ""):
    import glob
    import os
    stims_name_list = glob.glob(stims_path + '*.csv')
    stims_name_list = [os.path.split(stim)[1] for stim in stims_name_list]
    #for i in range(pMatx.shape[1]):
        #plt.xlabel('Distance')
        #plt.ylabel('Param value [log10]')
        #plt.title('Param ' + str(i + 1))
        #plt.scatter([j for j in range(pMatx.shape[0])], pMatx[:,i], c = 'red', label = 'Unsorted') 
        #plt.scatter([j for j in range(pSortedMatx.shape[0])], pSortedMatx[:,i], c = 'blue', label = 'Sorted') 
        #plt.legend()
        #plt.show()
    import h5py
     # remove [1:3]
    for stimname in stims_name_list:
        stim_fn = stims_path + '/' +stimname
        stim = genfromtxt(stim_fn, delimiter=',')
        volts_path = modelFolder + "volts/"
        counter = 0
        volts_name = volts_path + 'pin_'+ str(psize) + '_params512_' + i + '_' + stimname
        volts_fn = modelFolder + volts_name
        try:
            all_volts = readCSV(volts_name, ' ')
        except:
            print("Failed: " + volts_name)
            continue
        print("Read ", volts_name)
        all_labels_new = label_ntraces(all_volts, range(psize), volts_name)
        outfile = modelFolder + "data/" + params_name + '_' + version + '_' + str(psize) + '_' + i + stimname[:-4]
        hf = h5py.File(outfile + '.h5', 'w')
        voltages = np.array(all_volts)
        hf.create_dataset("voltages",data=voltages)
        norm_par= np.array(pSetsN)
        hf.create_dataset("norm_par",data=norm_par)
        phys_par= np.array(pMatx)
        hf.create_dataset("phys_par",data=phys_par)
        phys_par= np.array(stim)
        hf.create_dataset("stim",data=stim)
        phys_par= np.array(all_labels_new)
        #hf.create_dataset("qa",data=all_labels_new)
        binQA = [1 if (int(q[1]) == 1 or int(q[1]) == 0.7) else 0 for q in all_labels_new]
        hf.create_dataset("binQA",data=binQA)
        
        base_vals = [20,30000,20,2000,200,0.3,0.1,3,0.3,0.1,3,0.1,0.75,30000]
        vary_vals = [[0.5*i, 2*i] for i in base_vals]
        vary_vals = [
            [10,250],
            [10,60000],
            [10,60000],
            [100,4000],
            [100,2000],
            [0.15,0.6], [0.05,0.2], [1.5,6],
            [0.15,0.6], [0.001,10], [0.03,300],
            [0.05,0.2],
            [0.3,1.5],
            [15000,60000]
        ]
        hf.create_dataset("phys_par_range", data=np.array(vary_vals))
        hf.close()
    del(stims_name_list[0:2])

import sys
k=sys.argv[1] #k in params512_k.csv
n=int(sys.argv[2]) #512
v=sys.argv[3] #10paramsv23
model=sys.argv[4] #mainen10v23

modelFolder = "/global/cscratch1/sd/asranjan/" + str(model) + "/"
stims_path = 'chirp23a/'#
params_name = "mainen"

import glob
import os
pMatxcsv = readCSV("params"+str(n)+"_"+ str(k) +".csv", " ")
pSetscsv = readCSV("params"+str(n)+"_"+ str(k) +"Sets.csv", " ")
gen_data(stims_path, modelFolder, params_name, n, pMatxcsv, pSetscsv, v, i=str(k))