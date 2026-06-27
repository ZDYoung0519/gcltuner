TASKID=$1
CONFIG=$2
WORKDIR=$3
PROBING_WORKDIR=$4


if [ "$TASKID" -gt 0 ]; then
    LASTCKPT=$(cat $WORKDIR/task$((TASKID-1))/last_checkpoint)
else
    LASTCKPT=$(python -c "import mmengine; config = mmengine.Config.fromfile('$CONFIG'); print(config.pretrained_pth)")
fi

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using Devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
echo "Config file: $CONFIG"
echo "Work dir: $WORKDIR"
echo "Pretrain checkpoint: $LASTCKPT"

ROUTER_PATH=$WORKDIR/router/task$TASKID/best.pth
DYRA_INIT_FILE=$WORKDIR/task$TASKID/dyra_init.pth


# ##############  Train  ###################
torchrun --nproc_per_node=$NGPUS gcltuner/tools/train.py\
    $CONFIG  \
    --cur-task $TASKID \
    --cfg-options model.pretrained_pth=$LASTCKPT \
        model.cur_task=$TASKID \
        model.llm_lora.cur_task=$TASKID \
        model.llm_lora.lora_init_file=$DYRA_INIT_FILE \
        model.router_args.trained_router_path=$ROUTER_PATH \
        model.router_args.forward_cur_expert_only='True' \
    --work-dir $WORKDIR/task$TASKID \
    --launcher pytorch \
    --deepspeed deepspeed_zero2 \
    --seed 42

# get the trained path
CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)
echo "Checkpoint saved at $CKPT"

# As deepspeed only save trainable parameters
# We need to combine the current params and the previous ones
if [ "$TASKID" -gt 0 ]; then
    # combine weights
    python projects/aligncl/tools/merge_checkpoints.py --cur-ckpt $CKPT --pre-ckpt $LASTCKPT --output $WORKDIR/task$TASKID/final.pth
    # write to the file, so we can read it correctly during the next task
    echo "$WORKDIR/task$TASKID/final.pth" > $WORKDIR/task$TASKID/last_checkpoint
fi

CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)
echo "Checkpoint saved at $CKPT"


# ##############  Test  ###################
torchrun --nproc_per_node=$NGPUS gcltuner/tools/test.py \
    $CONFIG  \
    --cur-task $TASKID \
    --checkpoint $CKPT \
    --cfg-options model.pretrained_pth='None' \
        model.cur_task=$TASKID \
        model.llm_lora.cur_task=$TASKID \
        model.llm_lora.lora_init_file=$DYRA_INIT_FILE \
        model.router_args.trained_router_path=$ROUTER_PATH \
        model.router_args.forward_cur_expert_only='False' \
    --work-dir $WORKDIR/eval/task$TASKID \
    --launcher pytorch 

python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval
