# AIGT

云端 AI 眼镜项目的单文件交互展示。下载并直接打开 [index.html](index.html)，无需构建、安装或联网加载资源。

## 七个模块

- 总览：眼镜显示概念图、在线服务与离线训练、多轮天气案例、六维质检。
- 检索：固定候选的排序与上下文截取，检索模型及错误归因。
- 监督微调：天气、数学、翻译样本，监督位置及采样配比。
- 偏好优化：可靠偏好对、DPO 损失交互计算与实验对照。
- 同策略蒸馏：学生采样、教师反馈、散度比较和时延预算。
- 记忆：四层体系、主题切换、约束更新、异步乱序与删除。
- 训练工程：专家负载、分配守恒、显存假设与四类并行。

模块地址为 `#overview`、`#rag`、`#sft`、`#dpo`、`#opd`、`#memory`、`#engineering`。页面资源全部内嵌。

## 数据与审阅口径

根据千问 AI 眼镜项目逐字稿整理；项目配置、教学算例与待验证方案分开呈现，详见 [审阅记录](REVIEW.md)。展示中的天气不是实时天气，交互计算不代表项目实测收益。

## 发布

在仓库 Settings → Pages 中选择 Deploy from a branch，分支 main，目录 /(root)，保存。发布成功后的默认地址是 https://kwaitray.github.io/AIGT/ 。代码推送不等于 Pages 已开启。

## 验证

本地 Chromium 已验证7个模块在1440像素桌面和390像素手机布局下无页面横向溢出、无脚本错误。已检查23项交互与数值行为，包括候选截取、DPO损失、前向与反向相对熵、相同分布零散度、时延边界、记忆更新、128专家分配总量和负载指标。

## 方法参考

- [向量模型](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [重排模型](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
- [混合专家结构参考](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)
- [直接偏好优化](https://arxiv.org/abs/2305.18290)
- [同策略蒸馏](https://arxiv.org/abs/2306.13649)
- [Megatron 并行策略](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [长期记忆评测](https://arxiv.org/abs/2410.10813)
