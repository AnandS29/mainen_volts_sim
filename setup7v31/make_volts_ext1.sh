module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4

c='mainen7v31'
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

python ../common/paramsetGen.py --num=1075 --model "mainen7v31" --vary 2 4 5 6 7 8 14 --seed=1846352 --start=101
sbatch sbatch_run.slr
