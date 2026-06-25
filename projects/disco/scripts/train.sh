TASKID=$1
CONFIG=$2
WORKDIR=$3

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

if [ "$TASKID" -gt 0 ]; then
    LASTCKPT=$(cat $WORKDIR/task$((TASKID-1))/last_checkpoint)
else
    LASTCKPT=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(config.pretrained_pth)")
fi

echo "Using Devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "Work dir: $WORKDIR"
echo "Pretrain checkpoint: $LASTCKPT"

# router path
ROUTER_PATH=$WORKDIR/router/task$TASKID/best.pth

train
torchrun --nproc_per_node=$NGPUS gcltuner/tools/train.py \
    $CONFIG  \
    --cur-task $TASKID \
    --cfg-options model.pretrained_pth=$LASTCKPT \
        model.cur_task=$TASKID \
        model.expert_router_args.pretrained_expert_router_path=$ROUTER_PATH \
        model.llm_lora.cur_task=0 \
    --work-dir $WORKDIR/task$TASKID \
    --launcher pytorch \
    --deepspeed deepspeed_zero2 \
    --seed 42

# get the trained path
CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)
echo "Checkpoint saved at $CKPT"

# test
torchrun --nproc_per_node=$NGPUS gcltuner/tools/test.py \
    $CONFIG \
    --cur-task $TASKID \
    --checkpoint $CKPT \
    --work-dir $WORKDIR/eval/task$TASKID \
    --cfg-options model.cur_task=$TASKID \
    --launcher pytorch 
