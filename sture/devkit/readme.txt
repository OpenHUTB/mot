MOT17:
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 61.5  92.9  0.99|546 147 244 155| 5253 43192  642  932|  56.3  83.5  56.9 


在MOT16 train上

测试集上的结果：
44.0  74.1  44.2
DMAN原始结果
Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 43.5  90.6  0.94|517  68 211 238| 4989 62430  216  819|  38.7  76.0  38.9 


非局部注意力没有训练的模型：
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.1  83.3  1.76|517  52 231 234| 9336 63921 1184 1661|  32.6  75.9  33.6 

MARS训练之后（没有效果）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.6  83.3  1.73|517  49 235 233| 9175 64482 1128 1650|  32.3  75.7  33.3 

MARS训练1000次以后
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.5  84.7  1.56|517  50 230 237| 8297 64569 1089 1655|  33.0  75.8  34.0 

MARS训练2000次以后
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.8  84.9  1.55|517  50 234 233| 8238 64237 1132 1648|  33.3  75.9  34.4 

(**************)MOT16训练400次后
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 39.8  93.5  0.57|517  55 213 249| 3037 66434  477  676|  36.6  76.7  37.1 

MOT16训练1000次后（变差了）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.1  87.9  1.17|517  51 233 233| 6231 65000  896 1208|  34.7  75.8  35.5 

MOT16训练1000次后（小学习率和步长，变差了）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  86.2  1.39|517  57 229 231| 7399 64092  962 1385|  34.4  76.0  35.2 


+MOT17训练50次后（大学习率，在MOT16迭代400次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.2  1.51|517  53 234 230| 8012 64176  993 1455|  33.7  75.7  34.6 

+MOT17训练370次后（大学习率，在MOT16迭代400次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.3  1.50|517  53 234 230| 7977 64192  987 1454|  33.7  75.7  34.6 


+MOT17训练200次后（小学习率，在MOT16迭代400次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.5  84.6  1.56|517  52 227 238| 8307 64613 1130 1641|  32.9  75.9  33.9 

+MOT17训练50次后，再用MOT16训练400次（大学习率，在MOT16迭代400次基础上）
********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.6  84.3  1.61|517  50 234 233| 8564 64473 1189 1759|  32.8  75.7  33.8 

+MOT17训练800次后（大学习率，在MOT16迭代400次基础上）
********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.2  1.51|517  53 235 229| 8004 64174  991 1454|  33.7  75.7  34.6 

+MOT17训练910次后（大学习率，在MOT16迭代400次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.3  1.50|517  53 234 230| 7989 64187  997 1457|  33.7  75.7  34.6 

+MOT16训练400次后（大学习率，在MOT17迭代基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.6  84.9  1.53|517  49 236 232| 8158 64455 1172 1693|  33.2  75.7  34.2 

+MOT16训练120次后（大学习率，在MOT17迭代基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  84.4  1.61|517  49 233 235| 8551 64112 1084 1530|  33.2  75.8  34.2 

 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  84.7  1.57|517  58 224 235| 8372 64044 1002 1434|  33.5  75.9  34.4 

+MOT17训练1300次后（大学习率，在MOT16迭代400次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.2  1.51|517  53 234 230| 8012 64176  993 1455|  33.7  75.7  34.6 

+MOT16训练1000次后（在MOT17迭代基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  84.7  1.58|517  52 226 239| 8391 64004 1056 1472|  33.5  75.7  34.4 

+MOT17训练2700次后（大学习率，在MOT16迭代400次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.5  1.49|517  55 231 231| 7897 64025  984 1445|  34.0  75.7  34.9 

+MOT16训练80次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.4  1.49|517  55 231 231| 7941 64105  971 1457|  33.9  75.7  34.7 


+MOT16训练20次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.4  1.49|517  55 231 231| 7895 64071  965 1459|  33.9  75.8  34.8 

+MOT16训练40次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.5  1.48|517  55 231 231| 7866 64028  985 1443|  34.0  75.7  34.9 

+MOT16训练80次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.9  85.3  1.50|517  53 234 230| 7989 64187  997 1457|  33.7  75.7  34.6 

+MOT16训练200次后（在MOT17迭代2700次基础上）
Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.5  1.48|517  55 231 231| 7869 64032  987 1444|  34.0  75.7  34.9 

+MOT16训练400次后（在MOT17迭代2700次基础上）
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.1  85.2  1.52|517  55 231 231| 8078 63949  995 1454|  33.9  75.7  34.8 

+MOT16训练600次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.5  1.48|517  55 231 231| 7869 64032  987 1444|  34.0  75.7  34.9 

+MOT16训练1000次后（在MOT17迭代2700次基础上）
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 42.0  85.5  1.48|517  55 231 231| 7867 64030  986 1445|  34.0  75.7  34.9 



MOT16训练_次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 40.0  89.7  0.95|517  55 217 245| 5064 66190  604  839|  34.9  76.5  35.5 

MOT16训练5次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练20次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练40次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练300次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练400次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练450次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 

MOT16训练550次
 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 38.0  95.6  0.36|517  50 202 265| 1935 68483  587  676|  35.7  77.2  36.2 


MOT16_0训练5次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 41.3  86.5  1.34|517  55 226 236| 7099 64784  888 1171|  34.1  75.9  34.9 

MOT16_0训练400次
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL 
 40.0  91.1  0.81|517  52 219 246| 4297 66224  581  810|  35.6  76.6  36.1 

MOT16_0训练550次




Multiple Object Tracking Challenge
==================================
---- http://motchallenge.net -----
----------------------------------

Version 1.1

This development kit provides scripts to evaluate tracking results.
Please report bugs to anton.milan@adelaide.edu.au


Requirements
============
- MATLAB
- Benchmark data 
  e.g. 2DMOT2015, available here: http://motchallenge.net/data/2D_MOT_2015/
  
  

Usage
=====

To compute the evaluation for the included demo, which corresponds to 
the results of the CEM tracker (continuous energy minimization) on the 
training set of the '2015 MOT 2DMark', start MATLAB, cd into devkit 
directory and run

	benchmarkDir = '../data/2DMOT2015/train/';
	allMets = evaluateTracking('c2-train.txt', 'res/data/', benchmarkDir);

Replace the value for benchmarkDir accordingly.

You should see the following output (be patient, it may take a minute):

Sequences: 
    'TUD-Stadtmitte'
    'TUD-Campus'
    'PETS09-S2L1'
    'ETH-Bahnhof'
    'ETH-Sunnyday'
    'ETH-Pedcross2'
    'ADL-Rundle-6'
    'ADL-Rundle-8'
    'KITTI-13'
    'KITTI-17'
    'Venice-2'
Evaluating ...
        ... TUD-Stadtmitte
*** 2D (Bounding Box overlap) ***
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 60.9  94.0  0.25| 10   5   4   1|   45   452    7    6|  56.4  65.4  56.9

        ... TUD-Campus
*** 2D (Bounding Box overlap) ***
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 58.2  94.1  0.18|  8   1   6   1|   13   150    7    7|  52.6  72.3  54.3

..................
..................
..................

 ********************* Your Benchmark Results (2D) ***********************
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 45.3  71.7  1.30|500  81 161 258| 7129 21842  220  338|  26.8  72.4  27.4 



Details
=======
The evaluation script accepts 3 arguments:

1)
sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
evaluated in a single run. These files are inside the ./seqmaps folder.

2)
The folder containing the tracking results. Each one should be saved in a
separate .txt file with the name of the respective sequence (see ./res/data)

3)
The folder containing the benchmark sequences.

The results will be shown for each individual sequence, as well as for the
entire benchmark.




Directory structure
===================
	

./res
----------
This directory contains 
  - the tracking results for each sequence in a subfolder data  
  - eval.txt, which shows all metrics for this demo
  
  
  
./utils
-------
Various scripts and functions used for evaluation.


./seqmaps
---------
Sequence lists for different benchmarks




Version history
===============
1.1.1 - Oct 10, 2016
  - Included camera projections scripts
	
1.1 - Feb 25, 2016
  - Included evaluation for the new MOT16 benchmark

1.0.5 - Nov 10, 2015
  - Fixed bug where result has only one frame
  - Fixed bug where results have extreme values for IDs
  - Results may now contain invalid frames, IDs, which will be ignored

1.0.4 - Oct 08, 2015
  - Fixed bug where result has more frames than ground truth

1.0.3 - Jul 04, 2015
  - Removed spurious frames from ETH-Pedcross2 result (thanks Nikos)
  
1.0.2 - Mar 11, 2015
  - Fix to exclude small bounding boxes from ground truth
  - Special case of empty mapping fixed

1.0.1 - Feb 06, 2015
  - Fixes in 3D evaluation (thanks Michael)

1.00 - Jan 23, 2015
  - initial release
