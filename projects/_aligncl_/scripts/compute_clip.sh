CONFIG=$1
CLIP_EMBEDDING_DIR=$2


IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

torchrun --nproc_per_node=$NGPUS projects/aligncl/tools/compute_clip_embeddings.py \
    $CONFIG \
    --output_dir $CLIP_EMBEDDING_DIR \
    --launcher pytorch

