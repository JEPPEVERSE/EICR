# EITR：成熟检索组件适配与验证报告

2026-11-19。此次工作针对已定位的候选池内排序损失，保留旧算法作为可复现对照，接入专用语义精排、独立去重和图扩散对照，以及 MRAgent 风格前端。已运行真实 API；实现可用与方法有效是两个不同结论。

## 1. 实现范围与实验合同

| 实验臂 | 候选生成 | 选择分数/机制 | 用途 |
|---|---|---|---|
| P3 | 现有 direct + 有界 tag 图扩展 | 原整题 BM25 与实际 token 贪心 | 旧排序对照 |
| C_rerank | 与 P3 完全相同 | 专用 query–passage reranker，保留原预算预留与贪心 | 单独测语义排序 |
| C_mmr | 与 C_rerank 完全相同 | 同一组语义分数 + 有条件的 MMR 去重 | 单独测去重 |
| C_ppr_rerank | seed 个性化 PPR + direct 候选 | 同一精排与装包流程 | 单独测候选生成 |
| C_ppr_mmr | 与 C_ppr_rerank 相同 | PPR + 精排 + MMR | 可选组合，非默认必跑 |

旧 P0–P5 和委员会消融仍可运行。`compare` 在旧图候选组内检查 `pool_hash` 完全相同，PPR 明确标为改变候选的实验臂。排序不会访问 gold；gold 只进入后置诊断和独立 grader。小规模实验优先跑 P3 / C_rerank，只有明确的重复损失才使用 MMR，只有候选损失才比较 PPR，无需每次跑满所有组合。

### 语义精排

`src/eitr/rerank.py` 实现真实 DashScope Text Rerank HTTP adapter。本次实际调用 `gte-rerank-v2`；未用 BM25 冒充神经重排，也未把某个未接入的本地 BGE/ColBERT 模型算作实现。服务内部架构与权重版本不在此适配器控制内，hosted model alias 不能当作不可变模型版本；实际请求、分数与响应 hash 均保存。

每个候选 atom 是焦点 passage，同时呈现候选池内的相邻上下文，保留 speaker、日期和原文。对邻居再次取上下文时与候选池求交，不读池外内容。模型返回每条候选的 `[0,1]` 分数；这是相关性分数，不是事实正确性、答案充分性或“同一事件”的概率。

[官方 API 文档](https://help.aliyun.com/zh/model-studio/text-rerank-api) 明确指出分数主要用于同一请求内比较。因此同一候选池整批评分，不将不同 batch 的原始分数混比。服务上限 500 条，超限明确失败；`configs/eitr_cascade_pilot.json` 把所有比较臂共同限制到 500 个候选 atom。禁止只裁剪语义臂。首次固定历史池验证的四个池本身均小于 500，不需要改变历史候选。

缓存键包括 question、corpus hash、atom ID、实际模型输入 hash、provider/model 身份。缓存损坏、错问题、错候选均拒绝。配对语义臂即使没有磁盘缓存也共享同一次评分。没有模型时不偷偷回退成 BM25。分数缓存包含原始使用量；缓存 replay 不代表部署重排免费。

### MMR

`src/eitr/cascade.py` 将 [MMR](https://doi.org/10.1145/290941.291025) 的相关性与冗余权衡适配到完整上下文 action：

\[
\operatorname{priority}(b\mid R)=\frac{\sum_{a\in b\setminus R}L_a[\lambda r(q,a)-(1-\lambda)\rho(a,R)]}{\max(1,\operatorname{tokens}(R\cup b)-\operatorname{tokens}(R))}.
\]

`L_a` 是原文 atom token 数。冗余 `rho` 是与已选 atom 的最大 TF-IDF cosine，默认只在相同已知 speaker、session 和日期内，且相似度 ≥ 0.85 时生效。默认 `lambda=0.85`，未经开发集调参。不同人的相似经历可能正是多跳回答需要的互补证据，不按重复处理。这里使用 TF-IDF 计算冗余；未声称启用了另一套 dense encoder。

选择保留完整 action、实际重渲染 token 计费、来源跨度去重和总预算。不能选择负收益动作，seed 阶段的 `allow_zero` 仅允许零值。`lambda=1` 必须退化到纯语义相关性排序，已有回归测试。此贪心与预算预留策略没有宣称全局最优保证。

### PPR

`src/eitr/diffusion.py` 对现有 owner/snapshot 的有向加权 tag 图进行 group-wise personalized PageRank。dangling mass 回到相应 seed 分布；保留 seed，用聚合 PPR mass 排序其余候选。先扩散，再截取输出 tag，避免把扩散中途被砍掉的向量冒充全图稳态。每轮记录真实边工作量、L1 变化、收敛误差界和停止理由。默认重启概率 0.15、最多 100 轮、容差 `1e-8`、边预算 50,000,000。

PPR mass 不等于旧 max-path support，因此不送入 P4 的 support 乘法。候选器显式读取 PPR 的 `candidate_order`。它仍受输出 tag 上限、每 tag 原文 shortlist 和全局候选上限约束；不能保证修复所有召回损失。RRF 仅提供经过测试的独立函数，没有偷偷加入未经验证的 dense 检索通道。

## 2. MRAgent 的借鉴方式

可以借鉴，而且现有离线资产已经来自 MRAgent 的 rewrite/tag/keyword 提取结果。继续使用原来的 pack 导入、ID 和合法 key/tag/fact 关系，无需为了更名重建图。新前端实现：问题 → keywords/alternatives → 既有 key 匹配 → 结合整题给已有 tag 评分 → 有界 seed set。

常规情况最多两次 LLM 调用，完全无匹配则只有第一次。默认全题最多提供 64 个 group/tag 对评分，避免在通常 2048 输出 token 配置下要求模型返回数百个 tag ID/score。严格拒绝伪造 tag、缺失 ID、非法分数，不静默当作成功；已计费失败调用写入日志。关键词变体是检索假设，不是已证明的实体别名。`question_time` 目前仅记录，不用未证实时间硬过滤证据。

来源明确标记为 [MRAgent](https://github.com/Ji-shuo/MRAgent) 风格适配，不能作为本论文新颖性。它不是完整原生 agent 重现；离线抽取继续用上游现成基础设施，未增加第二套离线抽取器。详见 [前端说明](eitr_native_frontend.md)。

## 3. 宽松 judge

核对上游固定 commit 的 `eval/judge.py` 后，确认模板允许“touches on the same topic”判正确。这比语义答案正确更宽松。现有严格口径保留为默认 `semantic_strict`，新增 `mragent_lenient`。两者分别输出文件、profile hash 和指标，不覆盖历史评分，也不混合汇总。

所有比较方法使用同一口径、同一 judge 模型；judge 看不到方法名或成本。`correct` 与 `citations_valid`、`answer_contract_valid`、`valid_and_correct` 分开：宽松判对不能把越界引用洗成合法，合法引用也不自动证明蕴含关系。来源召回完全不受 judge 放宽影响。

适配固定了上游模板/文件/commit hash，但不能称为复现原论文数字：judge 模型、category 5 的特殊字符串评估和 F1 仍有区别。论文可以用兼容口径做公平对齐，同时报告严格结果与引用审计。详见 [评分说明](eitr_judge_profiles_20261119.md)。

## 4. 运行方式与成本约束

将 `examples/eitr_reranker.json` 的 `cache_dir` 改成实际绝对路径；凭据从显式 `key_env` 读，配置中不要放 key。示例：

```bash
eitr batch --index /absolute/index.json --corpus /absolute/corpus.json \
  --questions /absolute/questions.json --config configs/eitr_cascade_pilot.json \
  --arms P3 C_rerank C_mmr --native-frontend --generate \
  --reranker-spec /absolute/reranker.json \
  --tokenizer json --tokenizer-file /absolute/tokenizer.json \
  --tokenizer-sha256 YOUR_VERIFIED_SHA256 \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen3.5-27b --key-env QWEN_API_KEY \
  --no-enable-thinking --no-json-mode --out /absolute/new-batch
```

若要只测排序，使用相同的原 literal frontend，不同时换入口映射。`--native-frontend` 与 `--llm-frontend` 互斥。正式协议的 `frontend` 支持 `mragent`，并将完整 `reranker` 设置纳入 freeze/preflight/experiment；`model` 可显式包含 `enable_thinking` 和 `json_mode`，实际传递到 frontend/answer/native tool backend。首次原生前端 smoke 揭露了旧协议未记录脚本 Qwen 兼容参数的缺口；失败日志保留，修复后使用新协议，不覆盖第一次结果。

分别评分：

```bash
eitr grade --batch /absolute/new-batch --labels /absolute/labels.json \
  --index /absolute/index.json --judge-profile semantic_strict \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen3.5-27b --key-env QWEN_API_KEY --no-enable-thinking --no-json-mode \
  --out /absolute/strict-grades
# 换成 --judge-profile mragent_lenient，并使用另一输出目录。
```

`calls.rerank` 单独记录真实 reranker usage；不会把聊天模型价格套到重排 token 上。缺少独立重排单价时美元成本为未知。实际部署端到端 latency 排除缓存命中；缓存原始秒数不会相加伪装实测 latency。托管重排的 token 总数与聊天推理 token 不可直接当作同价费用。

## 5. 真实验证结果

固定历史四题验证使用上一轮真实抽取输出，保持全部 action、candidate atom、原文预算完全相同。P3 的四份原文包逐字复现；C_rerank 四份包均改变；C_mmr 四份包均与 C_rerank 相同。来源召回如下：

| 问题 | 候选来源召回 | P3 最终召回 | C_rerank 最终召回 | C_mmr 最终召回 |
|---|---:|---:|---:|---:|
| conv-41-qa003 | 1/2 | 0/2 | 0/2 | 0/2 |
| conv-41-qa041 | 1/1 | 0/1 | 0/1 | 0/1 |
| conv-42-qa028 | 1/1 | 1/1 | 1/1 | 1/1 |
| conv-42-qa166 | 1/1 | 1/1 | 1/1 | 1/1 |

宏平均来源召回均为 50%，微计数均为 2/5；不能把没有改善写成成功。首次真实 rerank 共 4 次、471,474 个服务报告 token，每次 HTTP 调用约 0.76–1.07 秒；这是模型服务调用时间，不是端到端检索时延，也不是降本证明。

另预先固定两题（conv-41-qa003、conv-42-qa028），启用新前端，运行 P3 / C_rerank / C_mmr / C_ppr_rerank 与真实回答。四臂均成功完成。志愿活动题的候选来源由旧前端 1/2 变成 2/2，但四臂最终仍为 0/2；冰淇淋日期题四臂候选与最终均为 1/1。这证明新入口映射在该个案补齐了候选，却没有消除最终选择损失。两题不能估计总体准确率或证明非劣效。

双口径评分与最终测试数见本报告末尾的验收记录。

## 6. 解释与下一轮边界

本轮没有得到“换一种成熟重排器就能解决问题”的证据。旧数学委员会无效不能据此被排除，也不能据此认定新模型有效。实证上可以区分：前端在一个个案改善了可达性，整题相关性重排改变了排序，但没有选出所需的互补来源。PPR 在这两题上也没有额外最终召回收益。

多实体共同属性问题要求同时满足人物、活动及共享关系。对整道题给单条 passage 一个相关性分数，仍可能优先选中两个人各自很相关、却不同的活动。这个解释与当前日志相容，尚未被独立大样本证实。后续先检查失败题中相关 gold passage 的语义排名、人物归属与实际装包位置，不在此轮追加新的委员会或为两题调参数。

已有资产与方法边界不变：当前图表示、BM25 per-tag shortlist、所有候选上限仍会限制召回；简单 MMR 不保证识别全部语义重复；保守的同来源归属保护可能让去重不起作用；预算预留仍沿用旧规则以维持排序消融；P5 原生工具循环此前的错误也没有被本轮自动解决。不能把这些组件接入等同于已经获得 Findings 级效果或低成本优势。

成熟工作来源：[BERT passage reranking](https://arxiv.org/abs/1901.04085)、[MMR](https://doi.org/10.1145/290941.291025)、[HippoRAG 2](https://arxiv.org/abs/2502.14802)、[RRF](https://doi.org/10.1145/1571941.1572114)。它们分别指导候选后精排、冗余控制、query-conditioned 图扩散和可选排名融合；这里没有宣称对这些成熟算法本身的创新。

## 7. 最终验收记录

- 完整测试命令：`EITR_NATIVE_ROOT=../../Reproductions/MRAgent .venv/bin/pytest -q`。结果 **252 passed**，包含真实上游接口集成；wheel 构建另行通过。
- 固定历史候选池：最终代码下 12/12 运行确定性 replay 通过；P3 历史证据包全部逐字复现；重排分数完全使用首次真实调用缓存，无重新采样。
- 新前端端到端：2 题 × 4 臂 = 8/8 回放通过；两个 PPR 运行均达到配置收敛判据；答案及引用范围合同 8/8 有效。
- 两种 judge 均独立实际调用：四臂在 `semantic_strict`、`mragent_lenient` 下均为 **1/2**。宽松口径没有在这两题上改变胜负。
- 该两题平均每请求的前端 usage 为 4,914 token；P3 回答为 2,422 token；C_rerank/C_mmr 回答为 2,496.5 token，另外重排为 117,520.5 个服务 token；PPR 重排为 122,118.5 token。这些是不同服务的使用量，应分阶段展示，不能用聊天 token 单价统算。因此本轮不宣称实现美元降本。
- 失败记录：首次端到端请求因通用模型配置缺少 Qwen 兼容参数而 HTTP 失败，原日志保留；修复后另建 `native_protocol_v2.json`。没有挑选成功答案替换错误答案，没有用 gold 调整模型或超参数。

本地原始证据（因包含问答和记忆，不随代码提交）：

- `artifacts/eitr_cascade_20261119/validation-plan-final.json`
- `artifacts/eitr_cascade_20261119/fixed_pool_final/`
- `artifacts/eitr_cascade_20261119/fixed-pool-metrics-final.json`
- `artifacts/eitr_cascade_20261119/native_protocol_v2.json`
- `artifacts/eitr_cascade_20261119/native_live_v2/`
- `artifacts/eitr_cascade_20261119/native-grades.json`
- `artifacts/eitr_cascade_20261119/native-replay.json`
- `artifacts/eitr_cascade_20261119/rerank-cache/`

实现与这些记录支持“链路可运行、改动可隔离、失败可追踪”；目前不支持“精排已改善效果”“MMR 有增益”“PPR 有增益”或“新系统不弱于原生 MRAgent”。
