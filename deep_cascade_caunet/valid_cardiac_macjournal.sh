MODEL='deep-cascade-caunet_macjournal'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='cardiac'
MASK_TYPE='cartesian'

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
#ACC_FACTOR_4x


#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
CHECKPOINT='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
#ACC_FACTOR_5x




<<ACC_FACTOR_8x
ACC_FACTOR='8x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
ACC_FACTOR_8x
