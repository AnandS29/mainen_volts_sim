module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4

c='mainen7v30'
stimName='chirp23a'
coreN=${c}
wrkDir=${SCRATCH}/${coreN}
mkdir -p wrkDir
mkdir -p ${wrkDir}/'data'
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'stims'
mkdir -p ${wrkDir}/'params'

cp mainen_params_wide_range.csv ${wrkDir}/'params'
cp /global/homes/a/asranjan/ML/run_mainen/${stimName}/${stimName}.csv ${wrkDir}/'stims'

python ../common/paramsetGen.py --num=100 --model "mainen7v30" --vary 2 4 5 7 8 9 14 --seed=1846353
sbatch sbatch_run.slr
