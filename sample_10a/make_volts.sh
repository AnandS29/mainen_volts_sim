module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4

c='sample_10a'
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

python ../common/paramsetGen.py --num=2 --model "sample_10a" --vary 2 4 7 8 --seed=1846353
sbatch sbatch_run.slr
