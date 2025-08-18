下面用直观的线性例子把整个训练流程和关键参数串起来解释（尽量包含数值计算，便于理解）。

**场景基础**（用于举例）
- 状态链： S0 -> S1 -> S2 -> S3(目标)，每步代价 c=1，真实 cost-to-go h*=[3,2,1,0]。  
- 假设初始 target network 估计 h_target0=[10,10,10,0]（即目标除外估计很差）。

**常见运行参数**（举例）
- back_max = 2  
- states_per_update = 20000  
- max_update_steps = 2  
- update_method = GBFS  
- batch_size = 1000  
- epochs_per_update = 1  
- num_update_procs = 1  
- update_nnet_batch_size = 10000  
- lr = 0.001, lr_d = 0.9999993  
- loss_thresh = 0.05  
- single_gpu_training = False（默认可能包装 DataParallel）  
- max_itrs = 200000

## 🎯 关键节点

1) **数据生成（backwards sampling）**

- back_max 控制“从目标向后随机走”的最大步数。back_max=2 → 起点可能是 S2 或 S1（S0 只有在 back_max≥3 时出现）。  
- 代码会从目标反向随机退步生成大约 states_per_update 个起点（或分摊到多份，见第 3 点）。这些起点是训练的“原始样本来源”。

2) **向前扩展 / max_update_steps（把更多状态加入训练集）**

- update_steps = min(update_num+1, max_update_steps)。若 update_num 初始 0，则 update_steps = 1，随后会增加直到 max_update_steps。  
- max_update_steps=2 意味着每个起点会做最多 2 步前向搜索（用 GBFS/A*），搜索过程中遇到的中间状态也加入训练集。  
- 举例：起点 S1，用 2 步向前会遇到 S1->S2->S3，把 S1、S2、S3 都放入训练集（S3 是 goal，ctg=0）。

3) **outputs.shape[0] 与 states_per_update 的关系**

- 如果 max_update_steps>1，do_update 会把 states_per_update 分成 update_steps 份，每份做前向扩展，最终 outputs.shape[0]（训练样本数）通常会大于或≈states_per_update（因为扩展产生额外中间状态）。  
- 近似简单情形（不考虑重复）：outputs ≈ states_per_update * average_path_length_from_expansion。

4) **备份（backup）如何生成训练目标**

- “一步备份”公式（Bellman 备份）：
  target(s) = min_a [ c(s,a) + h_target(s') ]  
- 用例（一阶备份）：
  - 对 S2: target = 1 + h_target(S3) = 1 + 0 = 1  
  - 对 S1: target = 1 + h_target(S2) = 1 + 10 = 11  
  - 对 S0: target = 1 + h_target(S1) = 1 + 10 = 11
- 若允许两步扩展，等效做两次一阶备份（或直接两步 bootstrap），结果会把信息传播两层（S1 从 11 变成 2，S0 变成 12，见前面示例）。

5) **训练 current network（train）与计数 itr**

- outputs.shape[0] = N（本次产生的训练样本数）。  
- num_train_itrs = epochs_per_update * ceil(N / batch_size)。 例：N≈20000, batch_size=1000 → ceil=20 → num_train_itrs=20（若 epochs_per_update=1）。  
- itr += num_train_itrs：itr 是“累计的梯度更新步数”，用于 lr 衰减和 max_itrs 终止判断。  
- lr 按 lr * (lr_d ** itr) 衰减；举例 lr0=0.001, lr_d≈0.9999993：
  - itr=20 → lr ≈ 0.001 * 0.9999993^20 ≈ 0.000999986  
  - 许多轮后（itr 很大）lr 会显著下降。

6) **什么时候把 current 复制到 target（update）**

- 每次训练后计算 last_loss（train_nnet 返回）。若 last_loss < loss_thresh（例 0.05），则执行：
  copy_files(curr_dir, targ_dir); update_num += 1  
- 这样 target 网络被“更新”为最近训练出的 current；后续 do_update 会以新的 target 来做备份，使目标逐步改善。

7) **num_update_procs 与 update_nnet_batch_size**

- num_update_procs 控制并行 worker 数量，这些 worker 读取 targ_dir 下的模型并并行评估 h_target（加速 do_update 的预测步骤）。  
- update_nnet_batch_size 是每个进程做批量预测时的 batch 大小，显存不足时调小。

8) **single_gpu_training 与 DataParallel**

- 在 main 中：若 on_gpu 且 not single_gpu_training，则 nnet = nn.DataParallel(nnet) ——训练时会跨所有 CUDA_VISIBLE_DEVICES 并行。  
- single_gpu_training 会避免 DataParallel，仅在一个 GPU 上训练（但 update workers 仍可使用所有可见 GPU 来评估 target）。

9) **测试阶段（gbfs_test）**

- 每轮训练后用当前 nnet 构造 heuristic_fn 做测试： max_solve_steps = min(update_num+1, back_max)（允许的搜索深度用于测试）。  
- gbfs_test 用 heuristic_fn 在 num_test 个样本上评估解率 / 平均步数等，给训练过程的性能信号。

## ⚙️ 完整流程

1. 初始：载入 current（若无则随机初始），targ_dir 可能为空 → all_zeros True。  
2. 启动 heur_fn runner（多个进程）用 targ_dir 的模型做并行预测（若 targ_dir 为空则返回默认估计）。  
3. do_update：
   - 从目标向后随机退步生成起点（back_max 控制距离分布）。  
   - 对每个起点做最多 update_steps 的前向搜索（max_update_steps 控制上限），把遇到的状态收集为训练样本。  
   - 用 target network 对后继状态评估 h_target 并做 min-over-actions 的备份，得到 outputs（训练目标）。  
4. 停止 heur procs。  
5. train_nnet 用 states_nnet + outputs 在 current 上做 num_train_itrs 次梯度更新（按 batch_size、epochs_per_update）。更新 itr。  
6. 保存 current（state_dict + itr + update_num）。  
7. 测试 gbfs_test（评估当前 heuristic）。  
8. 若 last_loss < loss_thresh → copy current → target 并 update_num++（下轮开始 update_steps 会变大直到 max_update_steps）。  
9. 如果 itr >= max_itrs 则结束，否则回到第2步。

## 📝 Note:

- 备份（Bellman 一步）用 target 的估计自举出训练目标；把 current 拟合这些目标并在 loss 足够小后把 current 复制为新的 target，就像“稳定的值迭代”循序推进；多次循环后，估计趋近真实 ctg（在理想化条件下）。  
- back_max 决定“你训练哪些难度的起点”；max_update_steps 决定“在每个起点上往前看多远以把价值信息传播到更多状态”。
