module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4
python ../common/paramsetGen.py --num=100 --model "mainen4v27" --vary 2 4 7 8
sbatch sbatch_run.slr
