module load tensorflow/intel-1.12.0-py36
module load python/3.6-anaconda-4.4
python ./paramsetGen.py --num=2000 --model "mainen10v26" --vary 1 2 3 4 5 6 7 8 13 14 --start=2001
sbatch sbatch_run.slr
