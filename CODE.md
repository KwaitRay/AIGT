# 项目代码

根据逐字稿独立整理的可运行参考实现，用于解释设计和验证关键逻辑。它不代表历史生产代码，也不包含实际训练语料或模型权重。

## 从案例跑通链路

在仓库根目录使用 Python 3.10 以上版本：

```powershell
python -m pip install -e ".[training,test]"
python -m aigt demo
python -m aigt export
python -m aigt.training.smoke
python -m pytest -q
```

天气案例先问杭州明天下午是否适合外出，再改成上海。程序继承原日期、更新城市，排除错误城市、错误日期和请求时刻之后发布的资料，构建回答并生成离线检查报告。

生成文件都在不提交的 `outputs/` 中：

| 文件 | 内容 |
|---|---|
| `demo.json` | 解析后的需求、检索证据、模型可见上下文、回答、耗时、客观检查结果 |
| `datasets/` | 按来源组划分的 SFT、DPO、OPD 数据与来源清单 |
| `training-smoke/report.json` | 三次真实梯度更新的损失、参数变化、教师冻结检查 |
| `training-smoke/tiny.pt` | 微型模型检查点 |

默认检索使用确定性哈希向量，回答使用案例模板，方便离线验证数据流。天气全部为合成数据。微型模型使用合成整数词表，训练检查证明梯度链路可执行，不衡量问答能力。

## 代码与模块对应

| 模块 | 文件 | 已实现 |
|---|---|---|
| 多轮请求 | `aigt/pipeline.py` | 案例城市和时间继承、证据组装、回答长度口径、阶段耗时 |
| 检索 | `aigt/retrieval.py` | BM25 与向量独立召回、加权倒数排名融合、可选重排、元数据过滤 |
| 模型接入 | `aigt/models.py` | Qwen 向量模型、Qwen 重排模型、兼容聊天接口的客户端 |
| 四层记忆 | `aigt/memory.py` | 用户与作用域隔离、来源记录、去重、冲突检测、旧事件拒绝、删除墓碑 |
| 评测 | `aigt/evaluation.py` | 命中率、召回率、归一化折损累计增益、案例检查、六维评测结果结构 |
| 数据构建 | `aigt/data.py` | 来源组划分、偏好对导出、OPD 只导出输入、排除私有评分字段 |
| 训练目标 | `aigt/training/losses.py` | 回答掩码 SFT、冻结参考策略的 DPO、三种全词表蒸馏损失 |
| 训练验证 | `aigt/training/smoke.py` | 学生在线采样、同前缀教师反馈、参数更新与教师冻结检查 |
| 框架接入 | `aigt/training/launch.py` | ms-swift 参数构造、数据文件检查、默认只预览命令 |

记忆的语义抽取、互补合并和主题拆分需要上游提供已验证的键及作用域；当前实现处理确定性存储更新。数据库保存当前快照，不支持任意历史版本重建，训练时应为每条轨迹创建独立的初始快照。删除后的跨缓存传播需要服务层接入。

## 接入真实模型

生成服务已经部署后，可以替换离线回答器：

```powershell
python -m aigt demo --base-url http://localhost:8000/v1 --model YOUR_SERVED_MODEL
```

鉴权密钥通过环境变量 `AIGT_API_KEY` 读取。当前客户端不流式输出，所以只记录完成耗时，首字时延留空；不能把总生成时间当成首字时延。

检索侧安装 `pip install -e ".[models]"` 后，在自己的入口中组合：

```python
from aigt.models import QwenEncoder, QwenReranker
from aigt.retrieval import HybridRetriever

retriever = HybridRetriever(
    documents,
    encoder=QwenEncoder("Qwen/Qwen3-Embedding-8B", device="cuda:0"),
    reranker=QwenReranker("Qwen/Qwen3-Reranker-8B", device="cuda:1"),
)
```

向量模型取最后一个有效位置并归一化；重排模型根据“yes/no”的相对概率打分。适配器尚未用真实权重进行 GPU 验证；长资料会截断，生产环境应先按完整证据单元切块。元数据过滤依赖解析正确，不能修复原始表格错行、日期错绑等问题。默认规则解析器只覆盖示例城市和时间，其他场景应传入明确的 `Query`。

## 接入训练框架

先导出数据，再预览命令：

```powershell
python -m aigt.training.launch sft --model BASE_CHECKPOINT
python -m aigt.training.launch dpo --model SFT_CHECKPOINT
python -m aigt.training.launch opd --model STUDENT_CHECKPOINT --teacher TEACHER_CHECKPOINT --divergence reverse
```

确认所安装 ms-swift 版本支持相应参数、模型模板和训练目标后，增加 `--run` 才执行。默认参数是演示起点，学习率、批量和上下文长度需要实验选择。不同阶段应指定不同 `--output`。

SFT 使用全参数训练选项；DPO 使用所加载策略的参考模型机制；OPD 对接 GKD，学生在线生成比例设为 1，散度参数分别映射前向 0、对称混合 0.5、反向 1。它与使用教师反馈作为奖励的强化学习蒸馏是不同路径。

教师与学生必须具有一致的词表、分词器和输入模板。本地损失只能检查张量维度，不能代替词元编号语义一致性校验。微型验证使用完全相同的整数词表，因此满足这个条件。

本仓库没有执行 235B 全参数训练，也没有实现集群的张量、流水线、上下文和专家并行配置。大模型训练需在 Linux GPU 环境结合目标框架版本单独确定资源与并行策略。这里不把一个通用启动命令当作已验证的集群配置。

## 数据与评测边界

模型读取对话、检索证据和授权记忆。私有验证器、标准答案评分字段、最终奖励不进入生成输入。SFT 的优选回答和 DPO 的优劣回答仅作为相应训练目标；OPD 由学生重新生成。

演示数据只有 20 条，覆盖天气、算术和翻译，仅用于格式与执行检查，不代表项目的真实分布或评测集。正式数据需先完成近重复聚类和来源划分，再按用户、会话及时间防泄漏切分；现有导出器依据传入的来源组进行确定性划分。

六维结果中不能由客观规则判定的项目保持空值，需人工或经校准的评审模型补充。中文的“80 词”产品计数口径尚未确认，程序不会擅自转换为 80 个汉字。

## 上游参考

- [ms-swift](https://github.com/modelscope/ms-swift)：训练与蒸馏框架。
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)：向量模型协议。
- [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)：重排模型协议。

仓库只提交独立的项目层实现，不包含本地课程包、第三方框架副本和私有材料。
