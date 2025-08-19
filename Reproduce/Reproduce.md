# ::Cube3

## 运行命令

```bash
###### Train cost-to-go function
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1000000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30
cp -r saved_models/cube3/current/* saved_models/cube3/target/  # manually update target network
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 1200000 --loss_thresh 0.06 --back_max 30 --num_update_procs 30

###### Solve with A* search, use --verbose for more information
python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/current/ --env cube3 --weight 0.6 --batch_size 10000 --results_dir results/cube3/ --language cpp --nnet_batch_size 10000

###### Compare solutions to shortest path
python scripts/compare_solutions.py --soln1 data/cube3/test/data_0.pkl --soln2 results/cube3/results.pkl

# 0.3M itr
python ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 300000 --loss_thresh 0.06 --back_max 30 --num_update_procs 32

```

## 原作数据分析

我们从`./saved_models/cube3/output.txt`中提取了 update number 和迭代次数的对应关系。以下是提取的代码和结果：

```txt
Number 1 first appears, Itr: Itr: 5000, lr: 9.97E-04, loss: 0.90, targ_ctg: 1.88, nnet_ctg: 0.96, Time: 7.69
Number 2 first appears, Itr: Itr: 10000, lr: 9.93E-04, loss: 0.87, targ_ctg: 2.77, nnet_ctg: 1.89, Time: 7.02
Number 3 first appears, Itr: Itr: 15000, lr: 9.90E-04, loss: 0.81, targ_ctg: 3.59, nnet_ctg: 2.76, Time: 6.96
Number 4 first appears, Itr: Itr: 20000, lr: 9.86E-04, loss: 0.78, targ_ctg: 4.39, nnet_ctg: 3.59, Time: 6.68
Number 5 first appears, Itr: Itr: 25000, lr: 9.83E-04, loss: 0.72, targ_ctg: 5.13, nnet_ctg: 4.39, Time: 6.82
Number 6 first appears, Itr: Itr: 30000, lr: 9.79E-04, loss: 0.70, targ_ctg: 5.81, nnet_ctg: 5.10, Time: 7.14
Number 7 first appears, Itr: Itr: 35000, lr: 9.76E-04, loss: 0.65, targ_ctg: 6.51, nnet_ctg: 5.85, Time: 9.91
Number 8 first appears, Itr: Itr: 40000, lr: 9.72E-04, loss: 0.58, targ_ctg: 7.13, nnet_ctg: 6.54, Time: 7.62
Number 9 first appears, Itr: Itr: 45000, lr: 9.69E-04, loss: 0.54, targ_ctg: 7.64, nnet_ctg: 7.09, Time: 7.01
Number 10 first appears, Itr: Itr: 50000, lr: 9.66E-04, loss: 0.46, targ_ctg: 8.08, nnet_ctg: 7.60, Time: 9.76
Number 11 first appears, Itr: Itr: 55000, lr: 9.62E-04, loss: 0.39, targ_ctg: 8.58, nnet_ctg: 8.17, Time: 7.45
Number 12 first appears, Itr: Itr: 90000, lr: 9.39E-04, loss: 0.32, targ_ctg: 8.81, nnet_ctg: 8.47, Time: 6.66
Number 13 first appears, Itr: Itr: 195000, lr: 8.72E-04, loss: 0.25, targ_ctg: 9.11, nnet_ctg: 8.84, Time: 7.57
```

我们发现前11次是每个epoch就会更新一次target network，之后增速放缓，第12次更新经过7次epoch，第13次更新经过21次epoch，之后的201次epoch都没有再更新过。
注意到以下数据：

```txt
Back Steps: 28, %Solved: 0.00, avgSolveSteps: 0.00, CTG Mean(Std/Min/Max): 13.31(0.66/10.10/14.46)
Back Steps: 30, %Solved: 0.30, avgSolveSteps: 14.00, CTG Mean(Std/Min/Max): 13.33(0.62/10.38/14.38)
Test time: 10.05
Last loss was 0.097411
Done
device: cuda:0, devices: [0, 1, 2, 3], on_gpu: True
Updating cost-to-go with value iteration
0.02% (Total time: 4.40)
11.12% (Total time: 56.66)
22.24% (Total time: 109.39)
33.33% (Total time: 162.29)
44.45% (Total time: 215.22)
55.55% (Total time: 268.45)
66.67% (Total time: 321.62)
77.76% (Total time: 374.69)
88.88% (Total time: 427.93)
100.00% (Total time: 480.50)
Cost-to-go (mean/min/max): 9.55/0.00/15.51
Training model for update number 14 for 5000 iterations
Itr: 1025000, lr: 4.88E-04, loss: 0.19, targ_ctg: 9.49, nnet_ctg: 9.31, Time: 15.06
Itr: 1025100, lr: 4.88E-04, loss: 0.12, targ_ctg: 9.59, nnet_ctg: 9.60, Time: 6.77
```

第1M次Itr之后停滞了，之后手动更新一次（也就是第14次），重启训练，经过40次epoch结束再没更新过。

由此我们可以推论：

1. 训练到第13次number的时候大概已经收敛，经过39次epoch，大概8.6小时。后续的201次epoch，还需约44.6小时。
2. 若有必要进行后面的201次epoch，那么我们完全可以在train current network的时候用target network来 update cost-to-go with value iteration。节约时间。

注：

1. 每个epoch的时间以800秒计，约合13.3分钟，0.222小时。
2. 每个epoch的数据生成也需要GPU推理，因此，需要仔细设计，优化并行处理能力。

### 为什么中间手动复制 current network 到 target network

```bash
# DeepCubeA使用双网络结构：
# - current network: 实时更新的网络
# - target network: 用于计算Q-learning目标值的稳定网络

# 正常情况下target网络每隔一定steps自动更新
# 但这里手动更新是为了：
# 1. 在训练中期强制同步两个网络
# 2. 避免current和target差距过大导致训练不稳定

# 等价于一次性训练1.2M iterations，但中间手动同步了target网络
python ctg_approx/avi.py --env cube3 --max_itrs 1200000 --loss_thresh 0.06

```

