export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate gcltuner-env

CONFIG=projects/disco/experiments/ucit_llava_inst/config.py
WORKDIR=work_dirs/ucit_llava_inst/disco


bash projects/disco/scripts/train.sh 0 $CONFIG $WORKDIR
bash projects/disco/scripts/train.sh 1 $CONFIG $WORKDIR
bash projects/disco/scripts/train.sh 2 $CONFIG $WORKDIR
bash projects/disco/scripts/train.sh 3 $CONFIG $WORKDIR
bash projects/disco/scripts/train.sh 4 $CONFIG $WORKDIR
bash projects/disco/scripts/train.sh $CONFIG $WORKDIR
python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval


