# v3：问题条件下的 tag 矩阵检索与重排

2026-11-21。本文对应 `eitr matrix-*` 可执行主线，版本 0.3.1。历史 `P*`、`C_*` 与入口扰动命令保留用于复现，不是新主线的默认调度。新主线没有委员会投票、入口重要性动态配额、循环 LLM 探索或新增 judge。

## 实现范围

| 设计模块 | 实现 | 可审计产物 |
|---|---|---|
| MRAgent 静态数据接入 | 复用原始导入、合法来源和前缀检查；也可直接读取历史 index 中的 canonical snapshot | owner / snapshot / content hash |
| 下游事实矩阵 | `matrix_embeddings.py`、`matrix_index.py` | 单一事实矩阵 X、稀疏 tag×fact B、各 tag 行索引与来源权重 |
| tag–tag 数学关系 | `matrix_index.relation` | 平均关联、双向覆盖、最大桥接及实际事实对 |
| 一次问题解析 | `matrix_query.py` | 需求、主体、组合算子、原问字面约束、原始响应、降级原因 |
| 问题–tag 匹配 | `matrix_retrieval.match` | 需求×tag 分数与最佳事实 ID；保留整题通道 |
| 有界候选构建 | `matrix_retrieval.candidates` | 直接通道、tag 路由、可选图扩展、全上下文动作、裁剪数 |
| 成熟重排器 | 复用 `rerank.py` 的 DashScope `gte-rerank-v2` 接口 | 一次请求内的原文语义分数、文档哈希、调用账 |
| 固定预算选择 | `matrix_retrieval.select` | 逐需求首轮＋全局相关密度补齐、实际渲染成本与原文跨度 |
| 原文回答 | `matrix_pipeline.py`、`PlannedAnswer` | 原问题、原文、组合要求、合法引用、失败响应留存 |
| 实验闭环 | `matrix_experiment.py`、`matrix_cli.py` | 冻结、预检、多 owner 批处理、逐题日志、回放、隔离评分 |

这些是实现层面的完成情况，不是准确率或降本已验证的结论。静态事实、tag 的抽取质量仍依赖上游；没有把 MRAgent 的图结构改名作为新贡献。

## 数学对象与实际计算

事实按 `fact_id` 固定顺序排列。编码器的模型、版本声明、维度、query/document 前缀和 endpoint hash 共同定义身份；输出校验维度、有限数和非零范数后逐行 L2 归一化：

\[
X\in\mathbb R^{N\times d},\qquad B_{ti}=\mathbf1[i\in I_t],\qquad M_t=X[I_t,:].
\]

B 用 CSR 稀疏矩阵存储。X 在索引文件中存储一次，tag 的矩阵是行切片，不重复复制。来源分组定义为完全相同的 `origins` 集合，先等权分组、再等权分配组内事实：

\[
w_{ti}=\frac{1}{G_t\,|g(i)|}.
\]

这防止同一来源组因抽取出更多事实而线性增权，但**不意味着重叠来源组统计独立**。在线 query–tag 的 max 匹配不使用这些均值权重；权重用于 tag–tag 的平均和方向覆盖。

核函数及三种关系为：

\[
k(x,y)=\max(0,x^\top y),\quad
S_{avg}(t,u)=\sum_{i,j}w_{ti}w_{uj}k(x_i,x_j),
\]
\[
D(t\to u)=\sum_i w_{ti}\max_j k(x_i,x_j),\quad
S_{bridge}(t,u)=\max_{i,j}k(x_i,x_j).
\]

实现还保存 `D(u→t)` 和桥接最大值的 witness fact IDs。它们是不同含义的关联量，不相加解释成同事件概率。核矩阵分块计算，不分配全局 fact×fact 矩阵。

图构建是可选项，默认 `graph_top_k=0`。启用后先按来源平衡质心选出候选 tag 对（双向近邻的并集），对这些对精算上述关系，按 `coverage_forward` 保留出边。因此：**保留边的数值是精确的，边集合是近似的**。质心在此只用于降低离线候选边计算量，不替代主线中的 tag 多行表示。质心抵消可能遗漏桥接边，这是图消融的明确局限。

问题计划给出需求向量 Q，并始终把完整原问题作为第 0 行：

\[
A_{ji}=\max(0,q_j^\top x_i),\quad
R_{jt}=\max_{i\in I_t}A_{ji},\quad
W_{jt}=\arg\max_{i\in I_t}A_{ji}.
\]

系统保留每个需求的 R 行与 W，不把多个主体先平均成一个向量。空 tag 得分 0，witness 为 null。该计算没有校准成概率，也不保证向量能严格区分主体、否定或关系方向。

## 解析、路由、重排和回答的边界

1. 前端只进行一次问题计划调用，替换旧的关键词候选截断＋tag 名称 LLM 评分。需求文本可改写，但 anchors、subjects、constraints、非平凡 operator_anchor 必须出现在原问题中；交集、分主体和比较还检查各主体在需求里的绑定。字面 grounding 能阻止新增未出现的实体字段，**不能证明自由改写语义一定正确**。
2. 无前端、调用失败、无效 JSON/字段或 unknown 组合，显式降级到完整原问；保留失败调用账和状态。供应的 plan 仅能在 `supplied` 模式进入，问题文件禁止 labels/answers 等字段。
3. 对原问和各需求分别选择 top tags、每 tag 的 top facts；同时保留直接事实通道。固定轮转合并队列，不根据入口“重要性”动态改变预算。图臂最多两跳，只允许正覆盖权重的边，先按 `R[j,u] × D(t→u)` 排扩展候选，再以边强度打破零分或同分的并列，限定每跳新增 tag 数。目标 tag 的问题相似度为零时仍可由正权重桥接边召回，且转为原文候选时不再被零相似度过滤；这类证据由 reranker 进一步判别。
4. 事实 provenance 只有 source 粒度，不能伪造 fact 的精确字符区间。因此每个入选事实的全部原始 source atoms 都有机会成为 core，并在事实/来源之间轮转，避免长来源只保留开头。遇到候选上限时，core 与其邻接上下文完整保留或完整拒绝。
5. reranker 对候选原文 atom 及已准入的邻接上下文打分，保留说话者和日期；一个问题的候选最多 500 个，拒绝拆成多次请求后混比分数。默认 480 留在服务上限内。重排查询使用原问题。
6. 装包先按需求固定顺序各尝试选一个动作，优先级为原文 token 加权的 `rerank_score × requirement_similarity` 增益除以实际新增渲染成本；然后以整题 rerank 相关密度填充。后者不是新委员会。整包 token 数包含渲染头部、日期等，跨度合并后重新计数。固定轮转无法在预算过小时保证每个需求都覆盖，日志保留需求相似度；它不是证据充分性判断。
7. 回答阶段同时收到原问题、原文证据和 advisory plan。intersection 要求逐主体建立关系后才声称共同值；group_by_subject 分别作答；缺失部分要表达未知。计划是提示，不是证据，不通过相似度直接计算答案交集。日期约束不根据未验证事件日期硬过滤，source timestamp 不等于 event date。引用仍须落入实际可见原文跨度。
8. 回答合同无效或传输失败时，run 标记 `answer_failed`，保留 `raw_answer`、错误类型和调用成本，不自动重试或替换成看似有效的答案。评分对保留的原始答案文本照常作盲评，同时单独报告合同无效；若根本没有答案文本，按失败计错，不调用 judge，也不删题。evidence-only 运行不能评分。

## 收窄后的对照

| arm | 候选 | 选择 |
|---|---|---|
| `direct_rerank` | 完整原问＋相同需求的直接事实匹配，不计算 tag 分数 | 同一 reranker、同一选择器与预算 |
| `matrix_rerank` | 直接保底＋问题–tag 矩阵路由 | 同上，主方法 |
| `matrix_graph_rerank` | 上述候选＋有界 tag 图扩展 | 同上，辅助图消融 |
| `matrix_embedding` | 与 `matrix_rerank` 相同的候选内容 | 以整题事实相似度代替 reranker，重排器消融 |

直接检索控制臂可使用完整 `max_core_atoms` 对应的事实排名额度，不被主线较小的 `direct_facts` 保底额度限制。它共享前端、编码器、tokenizer、上下文和回答提示。对照应同时报告候选数、实际证据量、正确率和分阶段成本；不能因主方法读了更多文档就声称 tag 表示更优。原生 MRAgent 通过历史固定协议单独运行，不伪装成这四个模块消融臂。

## 可复现运行

在仓库根目录安装 `pip install -e '.[test,tokens,pilot]'`。模板见 `examples/matrix/`；需要把绝对路径、模型和 tokenizer SHA256 替换为实际配置。模板不含 API key，代码不自动读取 `.env`。`revision` 是服务身份声明；托管 alias 不保证权重不变，实际 embedding 和 rerank 输出由记录文件固定。embedding 请求会显式发送 `dimensions`；Python API 省略缓存参数时也会继承 backend 的 cache_only/cache_dir/batch_size，显式传参可覆盖。

准备矩阵索引（`--snapshot` 同时接受 canonical snapshot 或包含该 snapshot 的历史 index）：

```bash
eitr matrix-build --snapshot /absolute/path/to/old-index.json \
  --embedding-spec /absolute/path/to/embedding.json \
  --graph-top-k 8 --graph-shortlist 32 \
  --out /absolute/path/to/matrix-index.json

eitr matrix-prepare-corpus --index /absolute/path/to/matrix-index.json \
  --config examples/matrix/config.json --tokenizer-spec /absolute/path/to/tokenizer.json \
  --out /absolute/path/to/corpus.json
```

若现有 corpus 的 snapshot、atom 配置和 tokenizer 相同，它可直接复用。已有向量必须是带完整身份和事实顺序的 `eitr-matrix-embedding-request-1` 记录，使用 `--embedding-record` 输入；不会猜测旧 embedding 文件来自哪个编码器。

单题端到端：

```bash
eitr matrix-prepare-query --index /absolute/path/to/matrix-index.json \
  --question 'What animal do both Nate and Joanna like?' \
  --embedding-spec /absolute/path/to/embedding.json \
  --model-spec /absolute/path/to/model.json \
  --plan-cache-dir /absolute/path/to/plan-cache \
  --out /absolute/path/to/prepared.json

eitr matrix-retrieve --index /absolute/path/to/matrix-index.json \
  --corpus /absolute/path/to/corpus.json --tokenizer-spec /absolute/path/to/tokenizer.json \
  --question 'What animal do both Nate and Joanna like?' \
  --prepared /absolute/path/to/prepared.json --arm matrix_rerank \
  --config examples/matrix/config.json --reranker-spec /absolute/path/to/reranker.json \
  --answer-model-spec /absolute/path/to/model.json --out /absolute/path/to/run.json
```

`matrix-prepare-query --plan examples/matrix/plan.json` 可替代 live 前端；不传两者则 whole-query 降级模式。准备和检索步骤必须使用相同问题与编码器身份。

正式小规模实验使用 `examples/matrix/protocol.template.json` 填好路径后：

```bash
eitr matrix-freeze --spec /absolute/path/to/spec.json --out /absolute/path/to/protocol.json
eitr matrix-preflight --protocol /absolute/path/to/protocol.json --out /absolute/path/to/preflight.json
eitr matrix-experiment --protocol /absolute/path/to/protocol.json --out /absolute/path/to/new-experiment
```

问题行只允许 `question_id, owner_id, snapshot_id, question`，supplied 模式额外允许 `plan`。冻结/预检会只读检查所需缓存链，并固定当时已存在的计划、查询 embedding、rerank 缓存内容。缺少必需的 cache-only 文件会提前失败；whole_query/supplied 计划可确定性生成，不要求计划缓存。依赖尚未生成的 live plan 却要求下游 cache-only 的组合会被拒绝，需先准备计划。首次运行合法新增的缓存不会使原协议失效，但已固定缓存不能变更。全部 arm 共享同一题计划和查询 embedding，调用账仍按独立部署计入每个 arm；准备耗时单独记录，不能把排除了准备阶段的 retrieval_total 当端到端延迟。重复实验需新输出目录；缓存命中保留原调用成本。

输出含实验 manifest、每个 owner/snapshot 的 batch manifest、逐题 rows。准备完一题后即先落盘，再运行各实验臂；失败时写入题号、arm、prepared_query、failure 状态与已有调用账，之前完成的题目和当前已完成的 arm 不丢失。

```bash
eitr matrix-replay --index /absolute/path/to/matrix-index.json \
  --corpus /absolute/path/to/corpus.json --tokenizer-spec /absolute/path/to/tokenizer.json \
  --run /absolute/path/to/single-run.json --out /absolute/path/to/replay.json

eitr matrix-grade --index /absolute/path/to/matrix-index.json \
  --batch /absolute/path/to/owner-batch --labels /absolute/path/to/labels.json \
  --model-spec /absolute/path/to/judge-model.json --judge-profile semantic_strict \
  --out /absolute/path/to/new-grading

eitr summarize --rows /absolute/path/to/new-grading/rows.json --bootstrap 2000 --out /absolute/path/to/summary.json
```

批结果中每个 `rows[i]` 是可单独保存并回放的 run。回放校验 code/index/plan/vector/score hashes，重新计算匹配、witness、候选、选择和原文证据包；不重新调用模型，也不声称重现生成的随机性。宽松评分可另跑 `mragent_lenient`，不能与 strict 混合汇总。

`eitr cost-report` 支持 runs 列表；ledger 和 cost-report 汇总的 `by_billing_kind` 单独列出 chat、embedding、rerank。旧单一 chat 价格表不应用到混合账单，混合时 USD 返回 unknown；离线 embedding 成本在 index/build metrics 中，不混入在线。不能由“总 token”直接推导混合服务的美元节省。

无需 API 的 wiring smoke：

```bash
python examples/matrix/offline_smoke.py --out /tmp/new-matrix-smoke
pytest tests/eitr/test_matrix.py -q
pytest -q
python -m build --no-isolation
```

smoke 使用明确标注的 synthetic embedding、synthetic reranker 和字符计数，仅验证链路与回放，不能当模型效果实验。

## 已知的研究风险

- 最大事实匹配有 tag 大小带来的极值偏差，仍需与强直接检索对照验证，不能由矩阵形式推定质量提升。
- 质心 shortlist 可能漏掉多模态 tag 的桥接关系；无图主线是必要控制。
- 事实抽取遗漏、错误主体绑定、被上游改写错的事实，矩阵表示不能自动修复。原文重排与回答提供后续核验机会，不提供正确性保证。
- 当前查询端扫描所有事实行，复杂度约 O(JNd)，尚未实现 ANN/分层剪枝。它去掉的是 LLM 多轮探索，不是证明向量计算优于成熟 ANN。
- 本次不包含训练式 attention、多头可学习权重、分层聚类界限、独立委员会或额外 judge；这些不属于已收窄的当前主线。

## 审计后的能力边界

- “都／分别”通过需求分解和回答约束实现；系统没有符号化的主体—关系事实库，因此不是程序执行的答案集合交集/逐主体蕴含验证。
- 目前无独立的 `matrix-benchmark` 全流程延迟基准；批运行提供准备、检索、重排和回答分段时间。共享准备或缓存回放的时间不得冒充独立请求延迟，cost-report 中对应端到端延迟保持 null。
- 混合服务已分账，但没有多模型独立美元价目表配置；在补齐真实服务价格前，USD 保持 unknown。
- 本版本修改运行源码，旧 frozen protocol 和 run 的严格源码校验会失败；旧代码提交仍可复现旧实验，新实验需重新 freeze。
