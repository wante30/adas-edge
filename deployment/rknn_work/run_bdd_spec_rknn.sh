#!/usr/bin/env bash
set -e
source /home/jingyuwang/.venvs/rknn232/bin/activate
WORK=/mnt/c/Users/JingyuWang/adas-edge-b05o/deployment/rknn_work
DATA=/mnt/d/Datasets/adas_edge_training
OUT=$DATA/rknn_b05o

python $WORK/rknn_eval.py --kind multiclass --precision fp16 \
  --split-onnx $OUT/bdd_yolov8n_512x288_split_logits_clip.onnx \
  --rknn $OUT/bdd_fp16.rknn --onnx $DATA/b03c_rknn_validation_20261021/bdd_yolov8n_512x288.onnx \
  --images $DATA/b05n_bdd_multiclass/images/test --labels $DATA/b05n_bdd_multiclass/labels/test \
  --offset 200 --limit 800 --conf 0.001 --iou 0.7 --mode map --output $OUT/bdd_fp16_map.json

python $WORK/rknn_eval.py --kind multiclass --precision int8 \
  --split-onnx $OUT/bdd_yolov8n_512x288_split_logits_clip.onnx \
  --rknn $OUT/bdd_int8.rknn --onnx $DATA/b03c_rknn_validation_20261021/bdd_yolov8n_512x288.onnx \
  --calib-dir $OUT/calib_train_full_500 \
  --images $DATA/b05n_bdd_multiclass/images/test --labels $DATA/b05n_bdd_multiclass/labels/test \
  --offset 200 --limit 800 --conf 0.001 --iou 0.7 --mode map --output $OUT/bdd_int8_map.json

python $WORK/rknn_eval.py --kind single --precision fp16 \
  --split-onnx $OUT/b05l_p2_specialist_512x192_split_logits_clip.onnx \
  --rknn $OUT/spec_fp16.rknn --onnx $DATA/b05l_runs/v8n_p2/weights/best.onnx \
  --images $DATA/b03c_rknn_validation_20261021/board_holdout50 \
  --gt-dir $DATA/b03d_gt_hf_20261021 \
  --offset 0 --limit 50 --conf 0.001 --iou 0.5 --mode tlmap --output $OUT/spec_fp16_tlmap.json

python $WORK/rknn_eval.py --kind single --precision int8 \
  --split-onnx $OUT/b05l_p2_specialist_512x192_split_logits_clip.onnx \
  --rknn $OUT/spec_int8.rknn --onnx $DATA/b05l_runs/v8n_p2/weights/best.onnx \
  --calib-dir $OUT/calib_train_spec_500 \
  --images $DATA/b03c_rknn_validation_20261021/board_holdout50 \
  --gt-dir $DATA/b03d_gt_hf_20261021 \
  --offset 0 --limit 50 --conf 0.001 --iou 0.5 --mode tlmap --output $OUT/spec_int8_tlmap.json

echo ALL_DONE
