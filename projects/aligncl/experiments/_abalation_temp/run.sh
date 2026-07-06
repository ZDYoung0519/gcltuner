WORKDIR=work_dirs/ucit_vicuna_v15_7b/aligncl_itaa_abalation_temp
CONFIG=projects/aligncl/experiments/_abalation_temp/config.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')

echo "Using Devices: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NGPUS"
for tempv in 0.01 0.1 0.5 1 3; do
    for templ in 0.01 0.1 0.5 1 3; do
        echo "========== $(date): Running with tempv = $tempv, templ = $templ =========="
        TASKID=5
        CKPT=$(cat $WORKDIR/task$TASKID/last_checkpoint)
        echo "Checkpoint saved at $CKPT"
        echo "Test start $CKPT"
        ROUTER_PATH=$WORKDIR/router/task$TASKID/best.pth
        torchrun --nproc_per_node=$NGPUS gcltuner/tools/test.py \
            $CONFIG \
            --cur-task $TASKID \
            --checkpoint $CKPT \
            --work-dir $WORKDIR/eval/task$TASKID \
            --cfg-options model.cur_task=$TASKID \
                model.expert_router_args.pretrained_expert_router_path=$ROUTER_PATH \
                model.llm_lora.cur_task=$TASKID \
                model.temperature_v=$tempv \
                model.temperature_l=$templ \
            --launcher pytorch 
        python gcltuner/tools/eval_cl.py --eval-dir $WORKDIR/eval
        mv $WORKDIR/eval/metric_matrix.csv $WORKDIR/eval/metric_matrix_tempv${tempv}_templ=${templ}.csv
    done
done

