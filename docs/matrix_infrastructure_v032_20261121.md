# v0.3.2 实验基础设施：精确建图、独立测量、计费和摊销

本次工作补齐上轮明确缺少的三项设施，并将它们接入可冻结、可回放的实验流程。**本次只做基础设施验收，不运行真实模型效果实验。** 前端逻辑仍采用需求分解＋原文回答，没有扩展为新的符号推理方法。

## 1. 全量精确建图与近似图对照

`matrix-build --graph-mode exact` 遍历所有非空 tag 的无序对，分块计算原有平均关联、双向覆盖、桥接最大值及 witness。每处理一对，更新两侧的 top-k，内存中每个 tag 最多暂存 k+1 条边，不需要保存全量 fact×fact 或 tag×tag 矩阵。

**“精确”指对全部 tag 对计算后得到精确的有向 top-k 图，并不意味着保留所有边。** 需要完整 tag 关系时将 `graph-top-k` 设为 tag 数减一。CLI 默认图模式是 exact；`graph-top-k=0` 仍不建辅助图。Python 的 build_index 保留原 shortlist 默认，调用时应显式传 graph_mode，避免实验配置含糊。

```bash
eitr matrix-build --snapshot /absolute/path/to/canonical-snapshot.json \
  --embedding-spec /absolute/path/to/embedding.json \
  --graph-mode exact --graph-top-k 8 --block 128 \
  --out /absolute/path/to/exact-index.json

eitr matrix-build --snapshot /absolute/path/to/canonical-snapshot.json \
  --embedding-spec /absolute/path/to/embedding.json \
  --graph-mode shortlist --graph-top-k 8 --graph-shortlist 32 \
  --out /absolute/path/to/approximate-index.json

eitr matrix-compare-graphs --exact-index /absolute/path/to/exact-index.json \
  --approximate-index /absolute/path/to/approximate-index.json \
  --out /absolute/path/to/graph-comparison.json
```

比较要求 snapshot、embedding record、top-k 完全一致。报告给出正权重边召回率、逐 tag 漏边及漏掉的方向覆盖质量；零权重边不计入分母。它衡量结构近似误差，不能替代答案质量实验。

构建侧输出新的 `*.metrics.json`：绑定 owner、snapshot、snapshot hash、index hash、embedding record hash，记录 embedding/graph 阶段时间、进程 wall/CPU/RSS，以及历史部署调用账和本次真实调用账。指标文件带独立 hash。精确与近似索引可复用同一份 embedding 缓存。

## 2. 独立请求测速

原 `matrix-experiment` 继续用于共享前端、共享查询向量的算法消融；它适合隔离检索差异，不冒充独立部署测量。

新增入口：

```bash
eitr matrix-benchmark-freeze --spec /absolute/path/to/benchmark-spec.json \
  --out /absolute/path/to/benchmark-protocol.json

eitr matrix-benchmark-preflight --protocol /absolute/path/to/benchmark-protocol.json \
  --out /absolute/path/to/benchmark-preflight.json

eitr matrix-benchmark --protocol /absolute/path/to/benchmark-protocol.json \
  --out /absolute/path/to/new-benchmark
```

`examples/matrix/benchmark.template.json` 的三个顶层字段：

- `experiment`：既有矩阵实验配置，指定资产、无标签问题、模型、tokenizer 和 arms。
- `settings`：`repeats`、`warmup_repeats`、`seed`、`cache_policy`、`load_mode`。
- `prices`：完整的服务价格对象或 null。该对象纳入协议 hash，不依赖会变动的外部价格文件。

每个 arm 每次都独立执行问题计划、查询 embedding、检索、重排、装包和回答。排序按固定随机种子在 question/trial 分组内打乱 arm 顺序，并打乱题目分组，避免所有方法总按同一先后次序运行。warmup 在正式请求前执行，记录成本但不进入正式延迟/质量统计。

三种缓存策略：

| cache_policy | 行为 | 延迟解释 |
|---|---|---|
| bypass | 旁路本地 plan、query embedding、rerank 缓存；不读、不写这些缓存 | 用于独立部署测量 |
| reuse | 允许复用现有本地缓存 | 记录命中；任何准备或重排缓存命中的请求均不进入 uncached 延迟统计 |
| only | 前端/查询向量/重排只读缓存；预检检查依赖完整性 | 用于离线检查；回答在 generate=true 时仍独立调用模型 |

`cache_policy` 覆盖 experiment 内各阶段的 cache_only 配置，并在冻结协议的 effective experiment 中显式记录。bypass 预检不读取或固定那些不应使用的本地缓存。它**不能关闭服务端 prompt caching**；后端返回缓存 token 数时单独保存和计价。

两种加载模式：

- warm：index/corpus 预加载，计时起点在每次问题处理之前。
- cold：把 index/corpus 的对象加载与验证放入每次请求计时；**不清空 OS 页缓存、不重启进程、不保证服务端冷启动**。tokenizer 初始化、后端对象创建和日志写盘不在请求计时内。

每个 run 包含 `benchmark.wall_seconds`、load/prepare 时间、原有分阶段时间、cache 标记、实际调用列表 `actual_calls`、部署调用列表 `calls`。二者用途不同：

- `calls` 保留缓存里历史调用，用于估算独立部署的 API 费用。
- `actual_calls` 只包含本次真实发出的请求，用于统计实验实际 API 花费。

失败请求保留为 `eitr-matrix-failure-1`，包含失败阶段、已完成准备、实际及部署调用账。已知 provider/contract 错误不会导致丢题，其他编程异常会中止并保留 journal。准备阶段中途失败，也保存已完成的前端计划及其缓存部署成本。

延迟分位数目前是**成功且无本地缓存命中的独立请求**的分位数，同时显式报告总请求数、失败数、预热失败和正式失败。不能只引用成功延迟而隐去失败率。generate=false 的运行只报告检索相关时间，不冒充含回答的端到端测量。

## 3. 多服务价格与实际实验账单

`examples/matrix/prices.template.json` 提供格式。模板金额为 null，表示尚未填写，**不是零价格，也不是可直接用于正式计费的配置**。填写实际费率后，把完整对象放入 benchmark spec 的 prices。

每行费率以 `(billing_kind, model, endpoint_hash)` 精确匹配，支持当前三个后端：chat_tokens、embedding_tokens、rerank_tokens。费用按输入/输出每百万 token 的美元费率计算。不同模型、不同 endpoint 不会共用一个价格。

endpoint_hash 用项目的 `digest()` 对字符串计算：chat 为去掉末尾 `/` 的 api_base；embedding/rerank 为完整 endpoint。修改 endpoint/model 后须同步修改价格身份。价格对象还记录日期、来源；若原价为人民币等其他币种，需要明确换算成 USD，并在 source 记载汇率依据。程序不猜测价格、不自动联网抓取价格。

`cached_input_per_million=null` 表示该行采用统一输入费率，不应用缓存折扣。若服务确有缓存折扣，应填入对应数字；此时 provider 不报告缓存输入 token 数，费用将保持 unknown，不能把全部 token 当成无缓存或全部缓存。旧的单一 chat 价格表保持兼容，但永远不会套用到 embedding/rerank。

```bash
eitr matrix-cost-report --benchmark /absolute/path/to/new-benchmark \
  --out /absolute/path/to/cost-report.json
```

默认使用冻结价格；`--prices` 可以明确提供新价目表重新核算。输出包括：

- 每个 arm 的部署 API 均价、分服务用量、成功延迟分位数及失败数。
- 含预热和失败调用在内的实际实验 API 花费。
- 完整费用未知时的 `known_usd` 小计、未计价调用和原因。小计不冒充完整账单。

benchmark 输出的 summary 只计 provider API 费用。CPU 费用通过下一节生命周期配置加入。不能把混合服务 token 总量直接当作费用。

## 4. 离线构建成本与摊销

```bash
eitr matrix-build-cost --metrics /absolute/path/to/exact-index.json.metrics.json \
  --prices /absolute/path/to/prices.json --compute-usd-per-cpu-hour 0.12 \
  --out /absolute/path/to/build-cost.json

eitr matrix-cost-report --benchmark /absolute/path/to/new-benchmark \
  --builds-spec /absolute/path/to/lifecycle.json \
  --out /absolute/path/to/cost-with-lifecycle.json
```

上面 0.12 **只是命令格式示例，不是实测/推荐价格**。实际填写自己计算资源的 CPU-hour 价格；0 表示显式只比较 provider 费用。未提供计算价格则总费用保持 unknown。

比较独立基础设施成本时，直接检索应使用 no-graph index，图方法使用 exact index；分别冻结各自的 benchmark（同一问题/重复配置），然后合并核算。不能给加载了 exact index 的在线 run 配上 no-graph 构建账单；程序会拒绝 index hash 不匹配。

```bash
eitr matrix-cost-report --benchmarks /absolute/path/to/direct-benchmark /absolute/path/to/graph-benchmark \
  --builds-spec /absolute/path/to/lifecycle.json --out /absolute/path/to/paired-lifecycle.json
```

合并时要求原题/试次/arm 不重复，冻结价格一致，或显式指定统一重计价格。分开的计时批次仍可能有服务负载漂移，时间比较需结合重复和失败率，不自动作因果或显著性声明。

`examples/matrix/lifecycle.template.json` 按 owner/snapshot 指定 baseline/method、双方的 build receipt 文件、每份快照预计查询数和 CPU 费率。校验内容包括：

- 同一记忆内容、同一 owner/snapshot，在线 run 的 index hash 与所列构建收据对应。
- 在线 question/trial 完整配对；不能拿不同题目的均值直接计算摊销。
- 不重复列同一构建收据或重复计入同一个 embedding 组件。
- 金额使用全部请求尝试，含失败；独立延迟只使用双方同时满足条件的配对样本，并报告样本数。

基本公式：

\[
\Delta C_{build}=C_{build,method}-C_{build,baseline},\qquad
s=C_{query,baseline}-C_{query,method}.
\]

新增建图成本为正且每题节省为正时，回本查询数为 `ceil(ΔC_build / s)`；指定查询量 N 时净节省为 `N*s − ΔC_build`。若在线更贵，没有“长期回本”结论；所有原始增量、均价及指定 N 下总额都保留，不只输出一个回本数字。

金额包括明确配置的离线/在线 CPU 费用及 provider 费用。时间包括实际测量的建图时间与配对请求时间。复用/导入 embedding 的构建不能冒充完整冷建图；若两侧共享完全相同的 embedding 组件，可消去共同项，报告其余构建时间的增量，同时设置 `shared_embedding_time_excluded=true`。此时显示的构建总额不包含共同的历史 embedding 时间。

当前构建收据从已有 canonical snapshot 开始；原始记忆的 LLM 抽取成本不在这一阶段。若两臂共享该上游，增量比较可消去它；不能把本报告的金额称为包含原始抽取的一切生命周期开销。

## 5. 回放、失败分母与重复评分

benchmark 保存 `jobs/000000.json` 等逐次 run，并导出 `grading/` 下按 snapshot/trial 分开的评分 batch，manifest 给出具体路径。预热不会进入评分 batch，正式运行失败会保留其中。沿用已有严格/宽松盲评：

```bash
eitr matrix-grade --index /absolute/path/to/index.json \
  --batch /absolute/path/to/new-benchmark/grading/HASH \
  --labels /absolute/path/to/labels.json --model-spec /absolute/path/to/judge-model.json \
  --judge-profile semantic_strict --out /absolute/path/to/graded-trial-0

eitr matrix-summarize-repeats --inputs /absolute/path/to/repeated-grades.json \
  --draws 2000 --seed 0 --out /absolute/path/to/repeated-summary.json
```

重复评分清单格式见 `examples/matrix/repeated-grades.template.json`。先求每道题在重复中的平均正确率，再比较方法并按 owner 聚类 bootstrap。重复次数不增加题数。缺少题目/arm/trial、重复样本或混合评分口径会被拒绝；少量 owner 不会自动支持非劣效结论。

单条成功 job 可直接传入已有 `matrix-replay --run ...`，重算矩阵匹配、候选和原文包；失败 job 只校验失败收据，不声称重现成功检索。汇总加载器检查逐 job hash、顺序、数量与冻结计划，防止静默少算运行。

## 6. 离线 embedding 的恢复

每批 embedding 在 `cache_dir/parts/` 中立即保存，绑定父请求、事实顺序和偏移。后续批次失败后，重跑相同请求会复用成功批次。失败的调用账保存到 `cache_dir/failures/`，不覆盖前一次失败记录。

最终完整 record 的 calls 是各成功批次的部署成本；本次 backend.calls 是新发出的请求，二者不会因恢复混淆。完全本地恢复可从完整的 parts 重组 record，再执行 freeze/preflight；预检要求统一的完整 query record，不能把一个缺少最终批次的目录当作完整缓存。

## 7. 验收与后续实验门槛

新增测试覆盖精确 top-k 对全对参考、图漏边比较、三服务价格、缓存折扣缺失、独立请求、缓存复用不混入延迟、冷对象加载、前端缓存后 embedding 失败、四臂失败日志及评分分母、预热花费、冻结价格与源码、重复统计、配对摊销及分批恢复。

进入真实小规模验证前，需要填写实际模型/价格/路径，重新构建或复用匹配资产并冻结协议。建议先做共享准备的算法消融，再用 bypass 独立基准测成本；不以一次共享缓存运行证明降本。所有这些入口现已实现，本次交付不产生新的准确率或真实降本结果。

## 本次验收记录

- 新增基础设施测试：18 项，涵盖上述成功与失败路径；全部通过。
- 全仓测试启用本地 MRAgent 集成路径：**308 passed，0 skipped**。
- 四个矩阵 arm 的合成 smoke、成功运行回放、失败收据校验、精确/近似图比较通过。
- CLI 冻结→预检→独立 benchmark→成本汇总→评分 batch 导出已在无网络的模拟后端上跑通，含重复、预热、冷对象加载和三种语义重排臂失败的场景。
- sdist/wheel 构建与 diff 格式检查通过。
- 本轮 provider 调用均为 mock 或合成后端，没有调用真实模型，没有产生真实题目准确率或服务费用结果。费率模板保留待填值，未虚构当前服务价格。
