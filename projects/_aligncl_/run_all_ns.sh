export CUDA_VISIBLE_DEVICES=0,1,2,3

CLIP_EMBEDDING_DIR=work_dirs/ucit_clip_vit_large_p14_336
CONFIG=projects/aligncl/configs/ucit_vicuna_7b_v15_clip_vit_large_p14_336_none.py
WORKDIR=work_dirs/aligncl/ucit_clip_vit_large_p14_336_none_mpl_ns
PROBING_WORKDIR=work_dirs/lora_ft/ucit_vicuna_7b_v15_clip_vit_large_p14_336_none

# compute CLIP vision and embeddings
# bash projects/aligncl/scripts/compute_clip.sh $CONFIG $CLIP_EMBEDDING_DIR

# continually update gaussian statics and train router
bash projects/aligncl/scripts/train_router.sh  $CONFIG $WORKDIR $CLIP_EMBEDDING_DIR

# train & eval
bash projects/aligncl/scripts/train_ns.sh 0 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/scripts/train_ns.sh 1 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/scripts/train_ns.sh 2 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/scripts/train_ns.sh 3 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/scripts/train_ns.sh 4 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/scripts/train_ns.sh 5 $CONFIG $WORKDIR $PROBING_WORKDIR

