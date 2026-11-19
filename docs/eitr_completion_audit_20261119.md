# EITR v2 必要实现完整性验收

日期：2026-11-19；版本：0.2.0。对应研究设计 v2 的正常图记忆 QA 工具主线与缩减实验矩阵。

第一版交付只完成主链路，遗漏了会影响委员会权重的别名合并，以及原生对照、离线缓存和评测闭环。这是实现与验收不足，不能让后续实验代替代码完整性检查。本次按设计逐项补齐。旧验收记录保留为历史，以下结论取代其“P5 未接入”等状态。

## 必要算法与基础设施对照

| 设计要求 | 实现/验收位置 | 状态 |
|---|---|---|
| 事实 cue 均分、完整来源组平衡、H/IDF/sqrt/L2 | native.py / index.py；独立分布公式参照 | 已实现并验证 |
| 正 cosine top-k、零向量与共享 cue 跨 tag | index.py；稀疏矩阵与直接公式核对 | 已实现并验证 |
| R_fact 来源平衡均值、fact ID 严格对齐 | index.py / align-fact-vectors；真实 2,048 维缓存 | 已实现并验证 |
| 原生 key–tag–fact，注入 speaker 单独标记 | native.py；原生导入回归 | 已实现并验证 |
| cue 规范化与已验证别名合并 | query.py；herbs/mixed herbs/train 反例 | 已修复并验证 |
| 歧义候选不当作已知同义词合并 | query.py；模糊匹配碰撞回归 | 已实现并验证 |
| 公平种子截断、组内 max、组间平均 | query.py / candidates.py | 已实现并验证 |
| 两跳、浅路径保留、零输出桥接、工作上限 | candidates.py；随机小图完整路径穷举 | 已实现并验证 |
| 精确种子范围，扩展 tag 原生范围 | candidates.py；跨 key 同 tag 反例 | 已实现并验证 |
| 固定完整上下文动作、候选上限与去重 | evidence.py / candidates.py | 已实现并验证 |
| 精确 max 覆盖边际、相关性底项、支持因子 | scheduler.py；每步独立穷举收益比 | 已实现并验证 |
| 实际整包 token 成本、预留、回收、重算 | scheduler.py；预算/重叠/覆盖饱和回归 | 已实现并验证 |
| 无映射回退、孤立入口、相同 cue 不同经历 | pipeline.py；专门反例 | 已实现并验证 |
| 固定原文 atom/BM25 离线缓存 | evidence.py / prepare-corpus | 已补齐并验证 |
| 原始对话日期核验 | dates.py；精确原文匹配及错误拒绝 | 已补齐并验证 |
| P0/P1/P2/P3/P4 与两项委员会消融 | pipeline.py / scheduler.py | 已实现并验证 |
| 正式 controlled-MRAgent P5 | native_runtime.py / controlled.py / tool_backend.py | 已接入实际原生循环并验证 |
| P5 七种工具、共享映射/原文/回答器 | 实际上游 ToolBridge 集成测试 | 已实现并验证 |
| 完整截断记录、分数、选中/备选动作 | candidates.py / scheduler.py / pipeline.py | 已补齐并验证 |
| 精确回放与跨 Python hash seed 稳定性 | experiment.replay / 跨进程测试 | 已实现并验证 |
| 多 owner 协议冻结、预检和批量执行 | experiment.py / batch.py | 已补齐并验证 |
| 标签隔离、独立 judge、回答文件完整性 | evaluate.grade_batch / 批量 hash | 已补齐并验证 |
| 来源召回、阶段失败定位与配对汇总 | evaluate.py | 已补齐并验证 |
| 失败费用、真实独立计时、p50/p95、摊销 | accounting.py / benchmark | 已补齐并验证 |

## 算法修正和工程修正

1. 已验证别名此前会被当成多个委员。现在唯一精确/别名 key 身份合并后重新等权，只保留一个覆盖维度；仅仅模糊匹配到相同候选不构成合并依据。
2. 原生 import 此前没有恢复运行时注入的单来源 speaker 导航边。现在按原生构图规则恢复，并标为 injected_speaker，仅用于寻址，不污染 H/V。无法按该规则核验的多来源/缺 speaker 事实显式记录，不猜测关联。
3. 原文切分和 BM25 此前每题重复初始化。现在支持快照缓存与批次复用；加载校验原文、跨度、上下文、配置、词频统计与实现 hash。
4. 浮点相关性累加此前遍历 set。现在按稳定 atom ID 累加，避免进程 hash seed 改变微小舍入与 tie-break。
5. 已补齐 direct/per-key/per-tag/global cap 和图前沿记录；候选池身份不掺入前端网络耗时。
6. 失败答案前的检索包也会落盘，完成文件带 hash；未知 API 用量保持 null，不能因复用或失败记成免费。

## P5 的明确实验定义

P5 实际执行固定本地 MRAgent checkout 的 `LLM.chat_with_tools_once`，使用其真实 Prompts、TOOLS、ToolBridge 和 JSON parser。加载前核对源码 SHA256，不导入自动初始化 API 客户端的上游应用。没有把旧 `P5_proxy` 改名为 P5。

受控部分符合主设计：统一入口和廉价初筛、统一原文与上下文输出、统一预算、最后使用共同回答器；原生循环内部生成的停止答案被丢弃但照常计费。七种原生工具都有可运行适配，保留 topic 与 person/aspect 原生来源关系。真实事件日期以原文会话时间为准，不把 rewrite 时间冒充事实。

该结果只能命名为 controlled-MRAgent P5。原生全流程 M0、其他公开系统的完整复现与后续表示消融，属于设计中的独立/后续实验项。不能把这份实现声称为所有论文基线和全部未来实验已经完成。

## 自动化验收

完整命令：

```bash
EITR_NATIVE_ROOT=/path/MRAgent python -m pytest -q
```

本机完整回归 **143 项通过**，含旧 EICR 76 项、新 EITR 67 项。新测试包括真实上游控制代码的 5 项集成检查；不提供 checkout 时这 5 项明确跳过，不能把跳过说成验证通过。

0.2.0 的 wheel/sdist 构建成功；wheel 已在独立临时目录离线安装，实际执行 canonicalize → build → prepare-corpus → compare，确认导入的是安装包而非工作区源码。入口、模块、源码包中的测试/验收脚本/文档均已检查。

除局部单元测试外，已用独立数学参照验证分布/余弦、随机小图完整两跳路径、每一步动态贪心；已在多个 Python hash seed 子进程检查选择结果；已覆盖协议文件漂移、标签混入、缓存失配、模型截断、失败费用与保留检索包。

## 真实缓存工程验收

验收脚本：`scripts/eitr_acceptance.py`。输入为本地 conv-44 原生缓存及其旧来源范围声明，校验原始 JSON/NPZ SHA256 后重新导入。

- 675 条原文通过与原始对话的逐条精确匹配，核实会话日期。
- 主图 335 个 tag、1,623 条事实；R_cue 生成 5,135 条有向边。
- 真实缓存的 2,048 维事实向量可生成 R_fact 图并运行相同调度。
- 保留合法有事实来源的原生 topic 与 personal/aspect 导航关系；不是新抽取节点。
- P0–P5 均执行证据预算检查；P2/P3/P4 候选池完全一致。
- P0–P4 及 R_fact 的 P3/P4 可从日志精确重放到相同动作、覆盖与证据包。
- P5 使用脚本化工具请求驱动真实上游循环，校验工具调用协议和原文输出；没有真实模型调用。

本地完整验收产物放在 Git 忽略的 `artifacts/eitr_final_acceptance_20261119/`，公开文档只保留汇总。运行时、数据及上游源码 hash 记录在 summary.json；该记录对应本次最终源码，不拿旧冒烟输出冒充新版本。

## 完成边界与实验规则

本次完成的是设计 v2 首轮所需的算法、P0–P5 与实验闭环代码，未开展新方法的真实模型质量实验。测试通过不构成效果保证，也不能证明绝无 bug。

下一次研究实验必须先冻结：正常题及 owner/snapshot、真实日期、模型/tokenizer、预算、评分协议。先在一个预算点比较 P0–P5；P3/P4 的差异用固定候选池、回放和失败诊断定位。实测成本包含前端、重试、探索、回答和本地计算。只有独立请求测量进入延迟曲线，缺失用量不得补零。若结果不佳，先按已记录阶段排查，再判断方法，不能跳过归因直接转向。
