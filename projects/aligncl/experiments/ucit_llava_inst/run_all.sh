export CUDA_VISIBLE_DEVICES=0,1,2,3
conda activate gcltuner-env

CONFIG=projects/aligncl/experiments/ucit_llava_inst/config.py
WORKDIR=work_dirs/ucit/llava_inst/aligncl_moelora_bs8

# continually update gaussian statics and train router
bash projects/aligncl/scripts/train_router.sh  $CONFIG $WORKDIR work_dirs/ucit_clip_vit_large_p14_336

bash projects/aligncl/scripts/train.sh 0 $CONFIG $WORKDIR
bash projects/aligncl/scripts/train.sh 1 $CONFIG $WORKDIR
bash projects/aligncl/scripts/train.sh 2 $CONFIG $WORKDIR
bash projects/aligncl/scripts/train.sh 3 $CONFIG $WORKDIR
bash projects/aligncl/scripts/train.sh 4 $CONFIG $WORKDIR
bash projects/aligncl/scripts/train.sh 5 $CONFIG $WORKDIR
python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval

