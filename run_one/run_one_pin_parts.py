import os
import neuron as nrn
import sys
i=sys.argv[3]
#name=sys.argv[4]

def load_and_run(run_file, stim_file, times_file, out_file, params_file):
	nrn.h.load_file(run_file)
	nrn.h.stimFile = stim_file
	nrn.h.timesFile = times_file
	nrn.h.outFile = out_file
    	nrn.h.paramsFile = params_file
        nrn.h("runModel()")
#             def get_mat_from_neuron():
#                 nrows = nrn.h("matOut.nrow")
#                 all_vecs = []
#                 for curr_row in range(nrows):
#                     nrn.h('tmp = matOut.getrow(curr_row)')
#                     all_vecs.append(nrn.h.tmp.to_python())
#                 return all_vecs
#             volts = get_mat_from_neuron()
#             print(volts)

run_file = './run_model_cori_pin_parts.hoc'
times = './times_0.02_23k.csv'
stims_path = './chirp23a/'
volts_path =  '/global/cscratch1/sd/asranjan/mainen4v27/volts/'
stims_name_list = os.listdir(stims_path)
for stim in stims_name_list:
	load_and_run(run_file, stims_path + stim, times, volts_path + 'pin_512_' + str(i) + "_" + stim, str(i) + ".csv")
	break
