CONFIG=$1
WORKDIR=$2
CLIP_EMBEDDING_DIR=$3

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

NUM_TASK=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(len(config.train_dataset))")
echo "Number of tasks: $NUM_TASK"

for ((TASKID=0; TASKID<$NUM_TASK; TASKID++)); do
    torchrun --nproc_per_node=$NGPUS projects/aligncl/tools/train_router.py \
        $CONFIG \
        --work-dir $WORKDIR/router/task$TASKID \
        --cur-task $TASKID \
        --saved-feature-dir $CLIP_EMBEDDING_DIR \
        --cov_mode var \
        --num_clusters_per_task 10 \
        --cfg-options batch_size=512 max_epochs=100 lr=0.001 optim_wrapper.optimizer.weight_decay=0.001 \
        --launcher pytorch
done

