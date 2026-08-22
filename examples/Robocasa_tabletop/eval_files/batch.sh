#!/bin/bash
# ============================================================
# RoboCasa Tabletop 全24任务自动评测脚本
# 使用方法: bash run_eval_all_tasks.sh
# ============================================================

STARVLA_DIR=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA
ROBOCASA_PYTHON=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/robocasa_fix/bin/python
CKPT_PATH=/mnt/nas/gezuhao/zhouyuchen/project/RSS-TRO/starVLA/playground/Checkpoints/qwenki_robocasa_30k/checkpoints/steps_30000_pytorch_model.pt
PORT=5678

# 日志目录（按时间命名，避免覆盖）
LOG_DIR="${STARVLA_DIR}/eval_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 24个任务列表
TASKS=(
  "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
  "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

TOTAL=${#TASKS[@]}
CURRENT=0

cd $STARVLA_DIR
export PYTHONPATH=/mnt/nas/gezuhao/zhouyuchen/miniconda3/envs/robocasa_fix/lib/python3.10/site-packages:$PYTHONPATH

echo "================================================"
echo "  RoboCasa Tabletop 全24任务评测"
echo "  开始时间: $(date)"
echo "  日志目录: ${LOG_DIR}"
echo "  模型: ${CKPT_PATH}"
echo "================================================"

# 汇总结果文件
RESULT_FILE="${LOG_DIR}/summary.txt"
echo "任务名称,成功率" > "$RESULT_FILE"

for TASK in "${TASKS[@]}"; do
    CURRENT=$((CURRENT + 1))
    TASK_LOG="${LOG_DIR}/$(echo $TASK | tr '/' '_').log"

    echo ""
    echo "================================================"
    echo "  [$CURRENT/$TOTAL] 正在评测: $TASK"
    echo "  日志: $TASK_LOG"
    echo "================================================"

    $ROBOCASA_PYTHON -m examples.Robocasa_tabletop.eval_files.simulation_env \
      --args.env_name "$TASK" \
      --args.port $PORT \
      --args.n_episodes 3 \
      --args.n_envs 1 \
      --args.max_episode_steps 360 \
      --args.n_action_steps 3 \
      --args.pretrained_path "$CKPT_PATH" \
      --args.video_out_path ./debug_videos2 \
      > "$TASK_LOG" 2>&1

    # 从日志里提取成功率
    SUCCESS_RATE=$(grep "Success rate:" "$TASK_LOG" | tail -1 | awk '{print $NF}')
    if [ -z "$SUCCESS_RATE" ]; then
        SUCCESS_RATE="ERROR"
    fi

    echo "  >> 成功率: $SUCCESS_RATE"
    echo "$TASK,$SUCCESS_RATE" >> "$RESULT_FILE"
done

echo ""
echo "================================================"
echo "  全部任务评测完成"
echo "  结束时间: $(date)"
echo "  汇总结果: ${RESULT_FILE}"
echo "================================================"
echo ""
echo "========== 评测汇总 =========="
cat "$RESULT_FILE"