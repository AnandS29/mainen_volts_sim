module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4

c='mainen10v33'
#stimName='chirp23a'
stimName='stims'
coreN=${c}
wrkDir=${SCRATCH}/${coreN}
mkdir -p wrkDir
mkdir -p ${wrkDir}/'data'
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'stims'
mkdir -p ${wrkDir}/'params'

cp mainen_params_wide_range.csv ${wrkDir}/'params'
cp -R /global/homes/a/asranjan/ML/run_mainen/${stimName}/. ${wrkDir}/'stims'

python ../common/paramsetGen.py --num=100 --model "mainen10v33" --vary 1 2 3 4 5 6 7 8 13 14 --seed=1846353
sbatch sbatch_run.slr
