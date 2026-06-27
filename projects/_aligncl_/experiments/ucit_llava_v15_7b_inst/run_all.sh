export CUDA_VISIBLE_DEVICES=0,1,2,3

CLIP_EMBEDDING_DIR=work_dirs/ucit_clip_vit_large_p14_336
CONFIG=projects/aligncl/experiments/ucit_llava_v15_7b_inst/config.py
WORKDIR=work_dirs/aligncl/ucit_llava_v15_7b_inst

# compute CLIP vision and embeddings
# bash projects/aligncl/scripts/compute_clip.sh $CONFIG $CLIP_EMBEDDING_DIR

# continually update gaussian statics and train router
bash projects/aligncl/scripts/train_router.sh  $CONFIG $WORKDIR $CLIP_EMBEDDING_DIR

# train & eval
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 0 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 1 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 2 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 3 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 4 $CONFIG $WORKDIR $PROBING_WORKDIR
bash projects/aligncl/experiments/ucit_llava_v15_7b_inst/run_all.sh 5 $CONFIG $WORKDIR $PROBING_WORKDIR

