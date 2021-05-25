
# jemalloc:
   export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
   export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.1/lib/libjemalloc.so
#
# tcmalloc:
#  export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.8/install/lib/libtcmalloc.so

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

### extra arguments:
###  --channels_last: enable NHWC
###  --profile: enable profiler
EXTRA_ARGS=$@

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

### single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"

### choose config
CONFIG="fast_rcnn_R_50_FPN_1x.yaml"
#CONFIG="faster_rcnn_X_101_32x8d_FPN_3x.yaml"
#CONFIG="retinanet_R_50_FPN_3x.yaml"


export OMP_NUM_THREADS=$CORES
export DETECTRON2_DATASETS=../datasets
$PREFIX python benchmark.py --config-file ../configs/COCO-Detection/$CONFIG --task eval $EXTRA_ARGS
