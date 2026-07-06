export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate gcltuner-env



CONFIG=projects/aligncl/experiments/ucit_llava_v15_7b_pret/config.py
TRAIN_SCRIPT=projects/aligncl/experiments/ucit_llava_v15_7b_pret/train.sh

WORKDIR=work_dirs/ucit_llava_v15_7b_pret/aligncl_itaa


# continually update gaussian statics and train router
bash projects/aligncl/scripts/train_router.sh  $CONFIG $WORKDIR work_dirs/ucit_clip_vit_large_p14_336

bash $TRAIN_SCRIPT 0 $CONFIG $WORKDIR

bash $TRAIN_SCRIPT 1 $CONFIG $WORKDIR

bash $TRAIN_SCRIPT 2 $CONFIG $WORKDIR

bash $TRAIN_SCRIPT 3 $CONFIG $WORKDIR
    
bash $TRAIN_SCRIPT 4 $CONFIG $WORKDIR

bash $TRAIN_SCRIPT 5 $CONFIG $WORKDIR

python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval

