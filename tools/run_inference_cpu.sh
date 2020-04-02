###
### Benchmark script for FRCNN with ResNext101_32x2d backbone
###
### 1. install detectron2 from branch "huiying"
###      git clone https://github.com/mingfeima/detectron2
###      cd detectron2
###      git checkout huiying
###      python setup.py install
###
### 2. prepare dataset
###      cd detectron2/datasets
###      ./prepare_for_tests.sh
###
###    the script will generate meta file under dir "coco"
###    download COCO dataset from the following link and unzip val2017.zip to detectron2/datasets/coco/val2017
###      http://images.cocodataset.org/zips/val2017.zip
###
###    the meta file generated needs to be renamed as
###      cd detectron2/datasets/coco/annotations
###      ln -s instances_val2017_100.json instances_val2017.json
###
###  3. config jemalloc
###    jemalloc is a custimized memory allocation, "cache" your malloc...
###      a) download from release: https://github.com/jemalloc/jemalloc/releases
###      b) tar -jxvf jemalloc-5.2.0.tar.bz2
###      c) ./configure
###         make
###      d) cd ./jemalloc-5.2.0/bin
###         chmod 777 jemalloc-config
###
###  4. launch the benchmark script
###       cd detectron2/tools
###       ./run_inference_cpu.sh
###
###
###  todo list: mingfei.ma@intel.com
###  [merged] BatchNorm preliminary optimization, vectorization
###  [merged] UpSample optimization
###  [      ] BatchNorm optimization
###  [      ] AvgPool optimization
###  [      ] _ROIAlign ??
###


### jemalloc config
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.0/lib/libjemalloc.so

### env config
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

### change me for core scaling, by default the script used 1 single socket to apply jemalloc
USED_NUM_CORES=$CORES
LAST_CORE=`expr $USED_NUM_CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

REFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export OMP_NUM_THREADS=$USED_NUM_CORES
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo -e "### using OMP_NUM_THREADS=$USED_NUM_CORES\n"
echo -e "### using $PREFIX\n"

export DETECTRON2_DATASETS=../datasets

### run performance benchmark
$PREFIX python benchmark.py --config-file ../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml --task eval

### run profiling
#$PREFIX python benchmark.py --config-file ../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml --task eval --profile
