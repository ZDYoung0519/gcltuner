# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1,2,3
conda activate gcltuner-env

# vicuna + llava pretarin & instruction-train
CONFIG=projects/lora/experiments/ucit_llava_7b_v15_inst.py
WORKDIR=work_dirs/lora/ucit_llava_7b_v15_inst
bash projects/lora/scripts/train.sh 0 $CONFIG $WORKDIR
bash projects/lora/scripts/train.sh 1 $CONFIG $WORKDIR
bash projects/lora/scripts/train.sh 2 $CONFIG $WORKDIR
bash projects/lora/scripts/train.sh 3 $CONFIG $WORKDIR
bash projects/lora/scripts/train.sh 4 $CONFIG $WORKDIR
bash projects/lora/scripts/train.sh 5 $CONFIG $WORKDIR
python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval


