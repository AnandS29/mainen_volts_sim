module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4

c='test'
stimName='chirp23a'
coreN=${c}
wrkDir=${SCRATCH}/${coreN}
mkdir -p wrkDir
mkdir -p ${wrkDir}/'data'
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'stims'
mkdir -p ${wrkDir}/'params'

cp /global/homes/a/asranjan/ML/run_mainen/${stimName}/${stimName}.csv ${wrkDir}/'stims'

python ../run_one/make_pset.py --params 1 2 3 4 5 6 7 8 9 10 11 12 13 14
sbatch sbatch_run.slr
