#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
NGPUS=4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gcltuner-env

CONFIG=projects/aligncl/experiments/_abalation_router/config.py
WORKDIR=work_dirs/ucit_vicuna_v15_7b/aligncl_abaltion_router
CLIP_EMBEDDING_DIR="path/to/your/clip_embeddings"

NUM_TASK=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(len(config.train_dataset))")
echo "Number of tasks: $NUM_TASK"

COV_MOD=var
PORT_BASE=29500  # 基础端口

for NUM_CLUSTERS in 1 3 5 10 20; do
    echo "========== $(date): Running with num_clusters_per_task = $NUM_CLUSTERS =========="

    for ((TASKID=0; TASKID<$NUM_TASK; TASKID++)); do
        # 动态分配端口，避免冲突
        PORT=$((PORT_BASE + TASKID + NUM_CLUSTERS * 10))
        echo "  Task $TASKID with clusters=$NUM_CLUSTERS, port=$PORT"
        
        torchrun --nproc_per_node=$NGPUS --master_port=$PORT \
            projects/aligncl/tools/train_router.py \
            $CONFIG \
            --work-dir $WORKDIR/router_${COV_MOD}_${NUM_CLUSTERS}cluster/task$TASKID \
            --cur-task $TASKID \
            --saved-feature-dir $CLIP_EMBEDDING_DIR \
            --cov_mode $COV_MOD \
            --num_clusters_per_task $NUM_CLUSTERS \
            --cfg-options batch_size=512 max_epochs=100 lr=0.001 optim_wrapper.optimizer.weight_decay=0.001 \
            --launcher pytorch
        
    done
    
    echo "========== $(date): Completed: num_clusters_per_task = $NUM_CLUSTERS =========="
done

echo "========== $(date): ALL EXPERIMENTS COMPLETED! =========="