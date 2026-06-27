export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate gcltuner-env


# vicuna + llava pretrain
CONFIG=projects/indivisual/experiments/ucit_llava_7b_v15_pret.py
WORKDIR=work_dirs/indivisual/ucit_llava_7b_v15_pret
bash projects/indivisual/scripts/train.sh 0 $CONFIG $WORKDIR
bash projects/indivisual/scripts/train.sh 1 $CONFIG $WORKDIR
bash projects/indivisual/scripts/train.sh 2 $CONFIG $WORKDIR
bash projects/indivisual/scripts/train.sh 3 $CONFIG $WORKDIR
bash projects/indivisual/scripts/train.sh 4 $CONFIG $WORKDIR
bash projects/indivisual/scripts/train.sh 5 $CONFIG $WORKDIR
python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval




