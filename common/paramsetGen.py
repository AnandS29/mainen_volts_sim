import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Uniform function for sampling
def uniform(normDiff, currbase, lower_bound, upper_bound, numRows):
   values = np.zeros((numRows, ))
   sign = np.sign(normDiff)
   i = 0
   for x in np.nditer(normDiff):
       #  val = lower_bound + ((x + 4) * ((upper_bound - lower_bound) / 8)) should be equivalent to np.interp
       #val = np.interp(x, [-4, 4], [lower_bound, upper_bound])
       val = np.interp(x, [-1, 1], [lower_bound, upper_bound])
       values[i] = val
       i += 1
   return values

# The follow function helps parse csv files and generates a 2d matrix for running calculations on.
# Returns complete matrix of parsed csv and also a row vector (1 x 12) of base values for each param.
def parse_csv(file_name):
	func_names = ['Uniform'] # We are now using Uniform only
	with open(file_name, 'rt') as csvfile:
	    reader = csv.reader(csvfile)
	    numCols = len(next(reader)) # changed this line from reader.next() for python3
	    numRows = sum(1 for row in reader) + 1
	    csvfile.seek(0)
	    next(reader)
	    m = np.zeros((numRows - 1, 7)) # numRows - 1 because ignoring header
	    i = 0

	    for row in reader:
	        for k in range(3):
	            m[i, k] = float(row[k+1])
	        m[i, 3] = float(row[7])

	        if row[5] in func_names:
	            m[i, 4] = func_names.index(row[5]) + 1
	        else:
	            raise Exception('Bad function name')

	        m[i, 5] = row[6] == 'Open'

	        if row[10] == "":
	            m[i, 6] = 0.1
	        else:
	            m[i, 6] = float(row[10])
	        i += 1
	return m, m[:,0].reshape((1, m.shape[0]))

def calculate_pmatx(data, nSubZones, nPerSubZone, sample_params, norm, seed):
	'''
	This function takes in __data__, which is the parsed CSV matrix after calling the above function *parse_csv*, and returns three things:
	- pMatx, which is the final pin data after sampling in a range of [-4, 4] and applying the uniform function. Each param not in sample_params will be kept at 0 and not sampled.
	- pSortedMatx, which is calculated by sorting by the summed squared differences from the original param value after sampling in the range [-4, 4] and then applying the uniform function. This matrix was mainly used for plotting the sorted values vs unsorted values to look at the general relationship between the two.
	- pSetsN is the samples from [-4, 4] for each param.
	- pSortedSetsN, which is just the sorted squared differences from the original param value after sampling in the range [-4, 4]. This does not have the uniform function applied, because we use this matrix for the OAT analysis.
	'''
	np.random.seed(seed) # seed for consistent sampling
	outputRows = nSubZones * nPerSubZone
	nParams = (data[:, 5] == 1).sum() # 12
	freeParams = len(data) - nParams
	pSetsN = np.zeros((outputRows, nParams))
	counter = 0

	# map the base values to a value in the range [-4, 4]
	mapped_base_values = np.zeros(nParams)
	for i in range(nParams):
		base_value = data[freeParams + i, 0]
		lower_bound = data[freeParams + i, 1]
		upper_bound = data[freeParams + i, 2]
		mapped_base_values[i] = np.interp(base_value, [lower_bound, upper_bound], [-1, 1])

	for i in range(nSubZones):
	    # sample_params are the params to be sampled. Other params are set to the mapped base value to [-4, 4].
	    for p in range(1, nParams+1):
	    	if p in sample_params:
        		# pSetsN[counter:counter + nPerSubZone, p - 1] = (np.random.rand(nPerSubZone) * 2.0 - 1.0) * (4.0 * (i + 1.0) / float(nSubZones))
    			pSetsN[counter:counter + nPerSubZone, p - 1] = np.random.rand(nPerSubZone) * [1 if np.random.rand() >= 0.5 else -1 for i in range(nPerSubZone)] * 1.0
    		else:
    			pSetsN[counter:counter + nPerSubZone, p - 1] = mapped_base_values[p-1]
	    counter += nPerSubZone

	# 100 norm of difference between param value and the mapped base value to [-4, 4]
	adjustedPSetsN = np.zeros((outputRows, nParams))

	for j in range(nParams):
	    for i in range(pSetsN.shape[0]):
	        adjustedPSetsN[i, j] = (pSetsN[i, j] - mapped_base_values[j]) ** norm

	# Sorting
	pSum = np.sum(adjustedPSetsN, axis=1)
	sortedList = [j[0] for j in sorted(enumerate(pSum), key=lambda i: i[1])] # sort in ascending order
	pSortedSetsN = np.zeros((outputRows, nParams))
	for j in range(nParams):
	    for i in range(len(sortedList)):
	        pSortedSetsN[i, j] = pSetsN[sortedList[i], j]

	# Calculate pMatx and pSortedMatx
	pMatx = np.zeros((outputRows, nParams))
	pSortedMatx = np.zeros((outputRows, nParams))
	pCurrND = np.zeros((outputRows, 1))
	pSortedCurrND = np.zeros((outputRows, 1))

	for i in range(nParams):
	    pCurrND = pSetsN[:, i]               # unsorted params
	    pSortedCurrND = pSortedSetsN[:, i]   # sorted params
	    currBase = data[freeParams + i, 0]
	    currBaseVar = data[freeParams + i, 3]
	    func = data[freeParams + i, 4]
	    lastParam = data[freeParams + i, 6]
	    lb = data[freeParams + i, 1]
	    ub = data[freeParams + i, 2]
	    adjustedND = pCurrND
	    adjustedSortedND = pSortedCurrND

	    if func == 1:
        	pMatx[:, i] = uniform(adjustedND, currBase, lb, ub, outputRows)
       		pSortedMatx[:, i] = uniform(adjustedSortedND, currBase, lb, ub, outputRows)
	    else:
	        raise Exception('function name error')

	return pMatx, pSortedMatx, pSetsN, pSortedSetsN


# make 120 12x12 matrices and concatenate row-wise, creating 1440 x 12 matrix (is able to generalize for different dimensions)
# permute each row of original matrix one at a time. The diagonal will be original values shifted by dx
# For OAT analysis (one at a time)
def shift_by_dx(data, dx, sample_params):
	'''
	This function shifts each param one at a time by a given dx value.
	The function takes in two parameters, __data__ and __dx__. __data__ should be the __pSortedSetsN__ matrix returned from the above function, and __dx__ should be the appropriate dx to shift the params.
	Returns __augmented__, which will be a 1440 * 12 matrix if the original input has 120 rows.
	'''
	dataRows = data.shape[0]
	nParams = data.shape[1]
	num_sample_params = len(sample_params)							# number of params to sample
	augmented = np.zeros((num_sample_params * dataRows, nParams))   # 1440 x 12 matrix
	curr_data_row = 0
	for row in range(0, augmented.shape[0], num_sample_params):     # iterate through rows, step size of sample_params at a time
		for ind in range(num_sample_params):                   	    # iterate through columns, where total columns is 12
			curr_col = sample_params[ind] - 1
			augmented[row + ind, :] = data[curr_data_row, :]
			augmented[row + ind, curr_col] += dx                    # increment one param at a time per row
		curr_data_row += 1
	return augmented


def calculate_pmatx_dx(data, augmented):
	'''
	This function takes in the shifted dx matrix __augmented__ and calculates the final pMatx after applying the multiplicative or exponent function. The first argument __data__ should be the parsed csv from before.
	Returns the new pMatx after applying the appropriate function.
	'''
	nParams = (data[:, 5] == 1).sum() # 12
	freeParams = len(data) - nParams
	pMatx = np.zeros((augmented.shape[0], augmented.shape[1]))
	pCurrND = np.zeros((augmented.shape[0], 1))
	for i in range(nParams):
	    pCurrND = augmented[:, i]
	    currBase = data[freeParams + i, 0]
	    currBaseVar = data[freeParams + i, 3]
	    func = data[freeParams + i, 4]
	    lastParam = data[freeParams + i, 6]
	    lb = data[freeParams + i, 1]
	    ub = data[freeParams + i, 2]
	    adjustedND = pCurrND
	    if func == 1:
	        pMatx[:, i] = uniform(adjustedND, currBase, lb, ub, pMatx.shape[0])
	return pMatx

# This function plots each param. Displays scatter plots of sorted vs unsorted values for each param.
def plot_sorted(pMatx, pSortedMatx):
	for i in range(pMatx.shape[1]):
	    plt.xlabel('Distance')
	    plt.ylabel('Param value [log10]')
	    plt.title('Param ' + str(i + 1))
	    plt.scatter([j for j in range(pMatx.shape[0])], pMatx[:,i], c = 'red', label = 'Unsorted')
	    plt.scatter([j for j in range(pSortedMatx.shape[0])], pSortedMatx[:,i], c = 'blue', label = 'Sorted')
	    plt.legend()
	    plt.show()

def plot_pSetsN(pSetsN):
	for i in range(pSetsN.shape[1]):
		plt.xlabel('Sampled value before uniform')
		plt.title('Param ' + str(i + 1) + ' Distribution Before Applying Uniform')
		sns.distplot(pSetsN[:,i])
		plt.show()

def plot_pMatx(pMatx):
	for i in range(pMatx.shape[1]):
		plt.xlabel('Sampled value after uniform')
		plt.title('Param ' + str(i + 1) + ' Distribution After Applying Uniform')
		sns.distplot(pMatx[:,i])
		plt.show()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
  "--vary",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],  # default if nothing is provided
)
parser.add_argument(
  "--num",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[100] # default if nothing is provided
)
parser.add_argument(
  "--model",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=str,
  default=["mainen4v27"] # default if nothing is provided
)
parser.add_argument(
  "--start",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[1] # default if nothing is provided
)
args = parser.parse_args()

#params_to_vary, num of files, model folder name

# Set the file path for the params csv here.
file_path = 'mainen_params_wide_range.csv'
m = args.model[0]
modelFolder = "/global/cscratch1/sd/asranjan/" + m + "/"#
params_csv = modelFolder + "params/" + file_path #
params_to_vary = [1,2,3,4,5,6,7,8,13,14]#
stims_path = modelFolder + 'stims/'#
params_name = "mainen"
psize = 512*100 #

params_to_vary = args.vary
n = args.num[0]
psize = 512*n
# Set these values accordingly to get the desired number of output rows. outputRows = nSubZones * nPerSubZone
nSubZones = 100 #
nSubZones = n
nPerSubZone = 512 #
assert psize == nSubZones*nPerSubZone
# Set the norm value here.
norm = 100

# Set the values of dx here. dx_vals should be a list.
dx_vals = [0.001]

# Set params to sample here. Other params will be kept at base value. The default below is to sample all params.
params = params_to_vary

# Set a seed value here for generating the samples between -4 and 4.
seed = 1846353

import numpy as np
data, orig = parse_csv(params_csv)
pMatx, pSortedMatx, pSetsN, pSortedSetsN = calculate_pmatx(data, nSubZones, nPerSubZone, params, norm, seed)

n = 512
pMatrices = [pMatx[i:i + n] for i in range(0, len(pMatx), n)]
pSetsNs = [pSetsN[i:i + n] for i in range(0, len(pSetsN), n)]
print(sum([len(i) for i in pMatrices]))
print(sum([len(i) for i in pSetsNs]))

start = args.start[0]

for i in range(len(pMatrices)):
    np.savetxt(modelFolder + "params/params"+str(n)+"_" + str(i+start) +".csv", pMatrices[i], delimiter=" ")
    np.savetxt(modelFolder + "params/params"+str(n)+"_" + str(i+start) + "Sets.csv", pSetsNs[i], delimiter=" ")
