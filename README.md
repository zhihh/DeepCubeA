# DeepCubeA

> - 🎯 **兼容环境** - DeepCubeA代码已完全适配现代PyTorch + GPU环境！
> - 🎉 **详细注释** - 加入详细的注释和案例分析，助力快速理解上手！
> - 🔧 **快速上手** - 提供小批量的测试代码，在本地机器快速跑出结果！

英文原版: [README_EN.md](./README_EN.md)

## 🚀 快速开始

```bash
# 1. 设置环境
export CUDA_VISIBLE_DEVICES=0
source setup.sh

# 2. 快速训练测试

python ctg_approx/avi.py --env puzzle15 --states_per_update 500000 --batch_size 100 --nnet_name puzzle15_test --max_itrs 20000 --loss_thresh 0.1 --back_max 50 --num_update_procs 3
## 若训练不走了，提升 max_itrs

# 3. A*搜索测试 (30秒)  
python search_methods/astar.py --states data/puzzle15/test/data_test.pkl --model_dir saved_models/puzzle15/current/ --env puzzle15 --weight 0.8 --batch_size 200 --results_dir results/puzzle15_test/ --language cpp --nnet_batch_size 100
## batch_size, for A* search
## nnet_batch_size, for neural network inference
## 由于快速训练的模型质量优先，启发式函数估值不准，导致A*搜索效率低下，因此我们使用已经训练好的模型进行搜索，并且使用小批量数据集，快速测试。

# 4. 查看结果
python scripts/compare_solutions.py --soln1 data/puzzle15/test/data_test.pkl --soln2 results/puzzle15_test/results.pkl
## 查看小批量数据集和上一步的结果
```

## ✅ 已验证环境

- **GPU**: NVIDIA GTX 1650 (4GB) 
- **Python**: 3.9+ 
- **PyTorch**: 2.5.1+
- **CUDA**: 12.0+

## 🎮 支持的环境

| 环境 | 代码 | 难度 | 状态 |
|------|------|------|------|
| 15拼图 | `puzzle15` | ⭐ | ✅ 兼容 |
| 24拼图 | `puzzle24` | ⭐⭐ | ✅ 兼容 |
| 35拼图 | `puzzle35` | ⭐⭐⭐ | ✅ 兼容 |
| 48拼图 | `puzzle48` | ⭐⭐⭐⭐ | ✅ 兼容 |
| 魔方3x3 | `cube3` | ⭐⭐⭐⭐ | ✅ 兼容 |
| 关灯游戏 | `lightsout7` | ⭐⭐⭐ | ✅ 兼容 |
| 推箱子 | `sokoban` | ⭐⭐⭐⭐⭐ | ✅ 兼容 |


## 🔥 常用配置

🔗参考[train.sh](./train.sh)

### 💨 快速测试 (适合调试)
```bash
python ctg_approx/avi.py --env puzzle15 --states_per_update 1000 --batch_size 100 --nnet_name test --max_itrs 10 --loss_thresh 0.5 --back_max 20 --num_update_procs 2 --num_test 50
```

### 🎯 标准训练 
```bash
python ctg_approx/avi.py --env puzzle15 --states_per_update 20000 --batch_size 1000 --nnet_name standard --max_itrs 500 --loss_thresh 0.1 --back_max 50 --num_update_procs 6 --num_test 200
```

### 🏆 高质量训练 
```bash
python ctg_approx/avi.py --env puzzle15 --states_per_update 50000 --batch_size 2000 --nnet_name high_quality --max_itrs 1000 --loss_thresh 0.05 --back_max 100 --num_update_procs 8 --num_test 500
```

## 💡 核心修复

1. ✅ **PyTorch兼容性** - 修复tensor类型转换和API变化
2. ✅ **Numpy兼容性** - 修复已弃用的类型定义
3. ✅ **多进程优化** - 修复multiprocessing类型注解问题
4. ✅ **GPU内存管理** - 优化显存使用，支持小显存GPU
5. ✅ **参数自动调节** - 针对不同GPU提供最优配置

## 🚨 故障排除

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | 减小 `--batch_size` 和 `--nnet_batch_size` |
| 训练太快结束 | 降低 `--loss_thresh`，增大 `--back_max` |
| 搜索太慢 | 使用 `--language cpp`，增大 `--weight` |
| 解质量差 | 降低 `--weight`，启用 `--max_update_steps` |

## 📈 监控工具

```bash
# GPU使用率
watch -n 1 nvidia-smi

# 训练进度
tail -f saved_models/*/output.txt

# 磁盘使用
du -sh saved_models/*/
```

## :gear: 其他

**A*搜索快速测试中的小批量数据集生成**：

```python
import pickle
import numpy as np

# 读取原始数据
data = pickle.load(open('data/puzzle15/test/data_0.pkl', 'rb'))
print('原始状态数量:', len(data['states']))

num = 10

# 取前num个状态进行测试
data_test = {
    'states': data['states'][:num],
    'solutions': data['solutions'][:num],
    'num_nodes_generated': data['num_nodes_generated'][:num],
    'times': data['times'][:num]
}

# 保存小测试集
pickle.dump(data_test, open('data/puzzle15/test/data_test.pkl', 'wb'))
print('小测试集状态数量:', len(data_test['states']))
```

**CPP版本的ASTAR编译**:

```bash
# 测试环境缺少boost库，安装依赖（不缺就不装，缺啥装啥）
sudo apt update && sudo apt install -y libboost-all-dev

# 编译CPP代码
cd cpp
make

# 本仓库已经编译好了
```

## 🏆 贡献者

- **原始论文**: [Solving the Rubik's Cube with Deep Reinforcement Learning and Search](https://www.nature.com/articles/s42256-019-0070-z)
- **原始代码**: [Github](https://github.com/forestagostinelli/DeepCubeA)
- **兼容适配**: 本仓库
- **实例分析**: [ExampleAnalysis.md](ExampleAnalysis.md) by GPT-5

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---
