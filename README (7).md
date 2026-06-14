# 🎬 电影推荐系统 — 多方法对比 + DeepSeek LLM

基于 **MovieLens 1M** 数据集，实现并对比 5 种推荐算法，并集成 **DeepSeek LLM** 提供可解释的智能推荐。

---

## 📌 项目简介

本项目以 Jupyter Notebook 的形式，从零到一构建了一个完整的电影推荐系统，涵盖数据加载、模型训练、评估指标计算、可视化展示，以及大语言模型增强推荐。适合作为推荐系统的入门与进阶学习案例。

---

## 📊 数据集

使用 [MovieLens 1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip) 数据集，**运行时自动下载**，无需手动准备。

| 文件 | 内容 | 规模 |
|------|------|------|
| `ratings.dat` | 用户对电影的评分（1-5 星） | ~100 万条 |
| `movies.dat` | 电影 ID、片名、类型 | 3,883 部 |
| `users.dat` | 用户性别、年龄、职业等 | 6,040 位 |

数据稀疏度约 **95.8%**，即大多数用户只评价了少数电影，是典型的协同过滤场景。

---

## 🧠 实现的算法

### 1. User-CF（基于用户的协同过滤）
计算用户间的余弦相似度，找到 Top-K 相似邻居，加权聚合邻居的评分来预测目标用户的偏好。

### 2. Item-CF（基于物品的协同过滤）
在物品维度上计算余弦相似度，根据用户历史高分电影，找到风格相近的未看电影推荐。

### 3. SVD（矩阵分解）
使用 `scipy.sparse.linalg.svds` 对用户-电影评分矩阵进行奇异值分解，保留前 50 个潜在因子，通过低秩近似填充缺失评分。训练集 RMSE 会在训练完成后输出。

### 4. Content-Based（基于内容）
对电影类型标签（genres）做 TF-IDF 向量化，计算电影间的余弦相似度。根据用户历史高分电影，推荐内容风格相近的候选电影。

### 5. Hybrid（混合推荐）
融合以上四种方法的推荐结果，对各方法的评分做归一化后加权求和：

| 方法 | 权重 |
|------|------|
| User-CF | 0.30 |
| Item-CF | 0.30 |
| Content-Based | 0.20 |
| SVD | 0.20 |

### 6. DeepSeek LLM（大语言模型推荐）
调用 DeepSeek Chat API，将用户历史高分电影和候选电影以自然语言形式传入 Prompt，由模型输出推荐列表及推荐理由，并总结用户偏好画像。返回结构化 JSON 格式。

---

## 📐 评估指标

在时间序列划分的训练集（80%）和测试集（20%）上，对每种算法计算以下指标（Top-10 推荐列表）：

| 指标 | 含义 |
|------|------|
| Precision@10 | 推荐列表中用户实际喜欢（≥4 分）的比例 |
| Recall@10 | 用户喜欢的电影中被推荐到的比例 |
| Hit Rate | 至少命中 1 部喜欢电影的用户比例 |
| F1-Score | Precision 与 Recall 的调和均值 |

---

## 📈 可视化输出

运行结束后生成两张图表并保存到 `/kaggle/working/`：

**`recommendation_comparison.png`** — 各算法 Precision / Recall / F1 柱状图对比

**`user_{id}_recommendations.png`** — 指定用户的观影历史 + 六种方法推荐结果展示

---

## 🚀 快速开始

### 环境要求

```bash
pip install scikit-surprise
# 其余依赖（pandas、numpy、scikit-learn、scipy、matplotlib、requests）
# 标准 Kaggle / Colab 环境已内置
```

### 运行方式

在 Kaggle Notebook 或本地 Jupyter 中按顺序执行 Cell 1 → Cell 12。

```
Cell 1   安装依赖
Cell 2   下载数据集、加载数据、预览
Cell 3   训练 User-CF
Cell 4   训练 Item-CF
Cell 5   训练 SVD（输出训练集 RMSE）
Cell 6   训练 Content-Based
Cell 7   初始化 Hybrid + DeepSeek LLM
Cell 8   评估所有方法，输出对比表格
Cell 9   可视化 — 方法精度对比图
Cell 10  单用户推荐演示（含 DeepSeek）
Cell 11  可视化 — 用户推荐展示图
Cell 12  查看输出文件列表
```

### 配置 DeepSeek API Key

在 Cell 7 中，默认会从环境变量读取：

```python
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'your-api-key-here')
```

推荐在 Kaggle Secrets 或本地环境变量中设置 `DEEPSEEK_API_KEY`，避免明文暴露密钥。

---

## 📁 项目结构

```
notebook.ipynb        # 主 Notebook，包含全部代码
README.md             # 项目说明文档
/kaggle/working/
  ├── recommendation_comparison.png    # 算法评估对比图
  └── user_{id}_recommendations.png   # 单用户推荐展示图
```

---

## 🔧 可调参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `n_neighbors` | UserCF / ItemCF | 20 | 邻居数量，越大结果越稳定但更慢 |
| `n_factors` | SVD | 50 | 潜在因子数，越大拟合能力越强 |
| `weights` | Hybrid | 见上表 | 各方法融合权重，可自行调整 |
| `test_users` | evaluate() | 50 | 评估时采样的用户数量 |
| `n` | recommend() | 10 | 推荐列表长度 |

---

## 📝 注意事项

- SVD 和 User-CF / Item-CF 均对数据做了子集采样（Top 用户 × Top 电影）以控制内存和速度，完整运行约需 **5-10 分钟**。
- DeepSeek 调用依赖网络连接，单次调用超时设为 30 秒，若失败会返回空列表并打印错误信息，不影响其他方法运行。
- 评估使用时间序列划分而非随机划分，更贴近真实推荐场景。

---

## 🗺️ 延伸方向

- 引入隐式反馈（点击、收藏）替代显式评分
- 使用 BPR（贝叶斯个性化排序）或 LightFM 提升排序效果
- 加入用户画像特征（年龄、性别、职业）做人口统计融合
- 将 DeepSeek 推荐理由接入前端，实现可解释推荐界面
- 部署为 FastAPI 或 Gradio 服务
