MODEL='deep-cascade-caunet_macjournal'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='cardiac'
MASK_TYPE='cartesian'
echo ${MODEL}


#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
REPORT_PATH='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_5x


#<<ACC_FACTOR_8x
ACC_FACTOR='8x'
REPORT_PATH='/media/htic/srihd1/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_8x
