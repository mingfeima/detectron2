CPU performance optimization on MaskedRCNN
-----

#### Optimization
I will gradually upstream CPU optimization to orinal repos but since this is somewhat long term.
Currently I put related patches at:
* **torch**: [test_channels_last_with_bfloat16_support](https://github.com/mingfeima/pytorch/tree/test_channels_last_with_bfloat16_support), this branch contains: a) channels last support; b) BFloat16 support; c) oneDNN upgrade to v2.2 (v2.1 has perf bug in weight reorder).
* **torchvision**: [cpu_opt](https://github.com/mingfeima/vision/tree/cpu_opt), this branch contains channels last suport on Float32. (Will cover BFloat16 later on).

#### Installation
* **torch**: follow official installation guide, build from source
* **torchvision**: `USE_OPENMP=1 USE_AVX512=1 python setup.py develop`

One major perf improvement comes from `ROIAlign` optimization, you have to use `USE_OPENMP=1` to enable the parallelization, and on NHWC memory format, the OP is fully vectorized but on NCHW it is almost impossible to do the job.

#### Datasets
```bash
### Download coco datasets from http://images.cocodataset.org/zips/val2017.zip
### abstract to datasets/coco/val2017
cd datasets; ./prepare_for_tests.sh

### rename meta file   
cp instances_val2017_100.json instances_val2017.json
```

#### Run CPU Performance Benchmark
```bash
### default: NCHW memory format
./run.sh

### Channels last memory format, aka. NHWC
./run.sh --channels_last

### use --profile if you want to collet profiler log
```
#### How to trigger channels last
```bash
### generally usage shall be:
input = input.to(memory_format=torch.channels_last)
model = model.to(memory_format=torch.channels_last)
### but since the input is 'List' on benchmark.py, only converting 'model' will also do the job
model = model.to(memory_format=torch.channels_last)
### basically CL propatation in Convolution is that either 'input' or 'weight' in CL format will trigger following CL path
###   aka. output will be CL and propagate.
```

#### Result On CLX single socket 20C
```bash
### with config "fast_rcnn_R_50_FPN_1x.yaml"
### default: 300 iters in 185.4384527085349 seconds.
### NCHW (opt): 300 iters in 80.56146793198423 seconds.
### NHWC (opt): 300 iters in 61.409820175002096 seconds.

### mkldnn 2.x has perf bug on NHWC weight reorder, update this later on.
```

#### Schedule
* [x] ROIAlign, ROIPool, PSROIAlign, PSROIPool channels last (fp32).
* [ ] NMS/BatchNMS (fp32).
* [ ] ROIOperator channels last (bf16).
* [ ] DenormConv2d channels last (fp32/bf16). Need to to evaluate priority (current impl should be rather slow).

NB: PSROIPool and PSROIAlign would end up with `gather` on channels dimension, so the perf is not optimal. Further optimization is possible but need to evaluate priority here.
