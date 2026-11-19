# EITR v2：实现与运行说明

2026-11-19。实现的是正常图记忆 QA 的低成本检索工具；不运行旧 EICR 的入口错误诊断。方法主线是下游 cue 分布索引、有界 tag 候选生成、多入口支持与已读原文覆盖反馈。结构梯子仅是备选研究框架。

本次完成本地实现与离线验收，没有调用真实 LLM，也没有新方法准确率结果。

## 已实现及明确边界

| 部分 | 状态 |
|---|---|
| 原生 pack 导入、scope/provenance 校验 | 完成；日期需要 manifest 的真实来源声明 |
| 来源平衡 cue 分布、IDF、平方根、稀疏 top-k 图 | 完成，R_cue |
| 下游事实向量均值对照 | 完成，R_fact；需要 caller 提供对齐且标明 encoder 的向量 |
| 一次 cue 前端、精确/别名/词面映射、入口公平截断 | 完成，复用原问题 span 验证；失败有界回退 |
| 分组图扩展、两跳深度状态、无输出桥接 | 完成 |
| 完整动作候选池、日期、邻接上下文和原文引用 | 完成；上下文是同会话相邻 atom，不是整轮无限展开 |
| P0–P4、两项委员会消融、strict_seed | 完成，可用相同候选池回放 |
| 一轮回答、引用验证、无标签批量运行、描述性配对汇总 | 完成；评分标签与运行输入隔离 |
| 原生 MRAgent / 正式 controlled-MRAgent P5 | **尚未实现为此运行器的可比臂**；旧仓库结果不能直接搬入 |
| 受控原生 key–tag 工具循环 | 已提供 `P5_proxy`，用于检查低成本工具循环接口，**不是原生 MRAgent，也不能冒充设计中的 P5** |
| SwiftMem/HyperMem 完整复现 | 不在本次实现内；正式比较前仍需核验 |

`P5_proxy` 以显式 JSON 选择已有 key–tag 对，通过已展示来源发现更多 key，再调用共用回答器。它缺少原生 MRAgent 的完整 topic/personality 工具、时间规划和原始提示词。日志固定标明这些差别。没有为“补齐六臂”把自定义循环改名为 MRAgent。

## 安装

```bash
python -m pip install -e '.[test,tokens]'
python -m pytest -q
```

使用模型提供的 `tokenizer.json` 时安装 `.[pilot]`。`python -m eitr` 不要求重新安装命令入口；从源码运行可设置 `PYTHONPATH=src`。

## 不联网的合成端到端示例

在仓库根目录执行，所有输出路径必须尚不存在：

```bash
python -m eitr canonicalize --snapshot examples/eitr/raw_snapshot.json --out artifacts/eitr-demo/snapshot.json
python -m eitr build --snapshot artifacts/eitr-demo/snapshot.json --config configs/eitr_v2.json --out artifacts/eitr-demo/index.json
python -m eitr audit --index artifacts/eitr-demo/index.json
python -m eitr compare --index artifacts/eitr-demo/index.json --question 'Which herbs did I use for halloumi salad?' --tokenizer characters --config configs/eitr_v2.json --out artifacts/eitr-demo/comparison.json
python -m eitr batch --index artifacts/eitr-demo/index.json --questions examples/eitr/questions.json --tokenizer characters --out artifacts/eitr-demo/batch
```

`characters` 只用于测试，不能解释为模型 token。CLI 禁止用它发起真实模型请求。真实运行必须选择相应的 tokenizer；`cl100k_base` 等 encoding 名也不是模型适配性的自动证明。

## 从 MRAgent 原生缓存导入

```bash
python -m eitr import-mragent --pack /path/to/conv.json --manifest /path/to/manifest.json --out artifacts/eitr-owner/snapshot.json
python -m eitr build --snapshot artifacts/eitr-owner/snapshot.json --out artifacts/eitr-owner/index.json
```

manifest 最小结构：

```json
{
  "owner_id": "conv-X",
  "snapshot_id": "frozen-snapshot",
  "allowed_source_ids": ["D1:1", "D1:2"],
  "extraction_prefix_verified": true,
  "keyword_cache_is_pre_injection": true,
  "session_datetimes": {"D1": "2024-09-15"},
  "attestation": "Describe the verified source horizon and date provenance"
}
```

时间映射键使用 raw session ID（如 `D1`），不是 rewrite session key（如 `session_1`）。没有核实的日期保持 `null`，不从事实 rewrite 的时间猜测。两项 true 是数据提供者声明，程序能检查关系一致性，不能证明虚假声明。

原生关键词应取运行时注入 speaker 之前的缓存。如果缓存中已经混入 speaker，应在 canonical 输入中明确用 `injected_speaker` 标注，而不是做上述声明。该边可用于原生寻址，但不进入统计特征。

旧 EICR `records.json` 的规范化边可能已经丢失 `edge_origin`；不能直接改 schema 名称后导入。应回到原生 pack，或提供保留来源标记的 canonical 原始数据。所有事实 origins 必须属于允许范围；不允许跨 owner/snapshot 合并。标签名称不自动 casefold 合并。

## 数学和调度

对事实 cue 均分，先在相同完整 origin 集合内平均，再在 tag 下平均为 H。设 d_c=log((M+1)/(df(c)+1))，使用 `normalize(sqrt(H)*sqrt(d))`。零向量保留直接读取，不补随机边。

候选支持：每个 query cue 组中，取已探索最大路径积；组间等权求和。组内候选数量不会重复计票。路径状态保留 group/tag/depth，最多两跳；达到 tag 数上限仍可更新已有 tag 的其他组支持。边预算不足完整一行时停止，记录截断。

种子只读取实际 `(key,tag)` 的事实并集；再次通过别的路径到达种子也不扩大其读取范围。非种子 tag 使用原生 tag 全部事实。候选入池按完整 core＋上下文动作进行，不截掉半个上下文。

覆盖状态 `C_j=max_{e in R} r_j(e)`，新增覆盖：

```text
delta_cov = mean_j max(0, max_{e in new} r_j(e) - C_j)
rel_density = sum_{e in new} text_tokens(e) * r_q(e) / max(1, actual_delta_tokens)
cov_density = reference_tokens * delta_cov / max(1, actual_delta_tokens)
priority = [epsilon + (1-epsilon)*support] * [eta*rel_density + (1-eta)*cov_density]
```

只除一次实际成本；两个密度项已包含分母。密度写法与原参考长度归一化收益比仅差固定常数，排序不变。相同覆盖收益下偏好便宜动作是成本目标的一部分，仍应检查是否牺牲语境完整性。

整题相关性底项保护相同 cue 的后续证据；`C=1` 不触发提前停止。覆盖是文本相似性代理，不是答案充分性。R 只在完整动作正式入包后更新。相同原文 atom 只计一次，时间、speaker、来源头和分隔符全部计入 token。全包重新渲染并计数，不能按独立块成本相加；暂时装不下的动作以后可以重试。

预留按 direct 1/3、seed 1/6 执行，剩余配额全部回收到公共池。种子按组轮转，预算允许时给予直接读取机会；并不承诺所有入口都能入包。

## 对照配置

| CLI arm | 行为 |
|---|---|
| P0 | 直接检索，完整预算，不需要 cue 提取模型 |
| P1 | seed＋direct，不扩展 |
| P2 | 同图固定候选池，按 tag 均分配额；剩余额度按固定 tag 顺序回收 |
| P3 | 同候选池，整题相关性/真实成本贪心、原文去重 |
| P4 | 完整委员会 |
| P4_no_support | 关闭入口支持因子 |
| P4_no_coverage | 关闭覆盖反馈，保留入口支持 |
| strict_seed | 仅种子原生范围，不回退 direct |

P2/P3/P4 和两个消融的 `pool_hash` 必须相同，`compare` 会断言。无映射时 P1–P4 使用与 P0 相同的直接池及完整预算；strict_seed 明确输出无证据。

```bash
python -m eitr retrieve --index artifacts/eitr-demo/index.json --question 'Which herbs?' --arm P4 --tokenizer characters --out artifacts/eitr-demo/p4.json
python -m eitr compare --index artifacts/eitr-demo/index.json --question 'Which herbs?' --arms P3 P4 P4_no_support P4_no_coverage --tokenizer characters --out artifacts/eitr-demo/ablations.json
```

本版的配置是 `configs/eitr_v2.json` 的扁平、严格 schema；研究文档的嵌套 `defaults_v2.json` 仅是设计说明，不能直接作为执行配置。未知字段报错。改 top_k 需要重建索引；改预算可复用相同离线索引。

## R_fact 对照

`build --fact-vectors aligned.json` 接受：

```json
{
  "snapshot_hash": "canonical snapshot content_hash",
  "encoder": "fixed encoder and revision",
  "fact_ids": ["exact ordered fact IDs from snapshot"],
  "vectors": [[0.1, 0.2]]
}
```

每条事实向量先单位化，按与 R_cue 相同的来源组规则平均，再单位化 tag 向量。顺序、维度、非有限数、零向量及 snapshot mismatch 均验证。向量生成成本外部单列；此命令不会调用 embedding API，也不会把旧 fact 向量当成 raw atom 向量。

## 可选模型接口

不指定标志则完全不联网。`--llm-frontend` 单独启用一次提取，格式失败最多重试一次；传输失败不重复扣费重试。`--generate` 只调用共用最终回答器。`compare` 只做检索；批量回答用 `batch --generate`。

```bash
python -m eitr retrieve --index artifacts/owner/index.json --question 'Your question' --arm P4 --tokenizer json --tokenizer-file /path/tokenizer.json --tokenizer-sha256 VERIFIED_SHA256 --llm-frontend --generate --api-base https://your-provider.example/v1 --model explicit-model --key-env YOUR_API_KEY --out artifacts/owner/result.json
```

密钥从环境变量读，不写入配置。固定 tokenizer JSON SHA256，拒绝不匹配文件。也可用 `--tokenizer cl100k_base` 等显式 encoding。

已缓存的前端 `--frontend` 必须携带 `question_hash`、`snapshot_hash`、`mentions`。每个 surface 必须是原问题中的字面子串；有歧义的重复位置必须给准确 Unicode offsets。默认保守字面扫描仅适合离线验收，不代表 LLM 抽取能力。

## 批量输入、失败与成本

questions JSON 是列表，每行只能包含 `question_id, owner_id, snapshot_id, question, mentions?`；出现 gold/reference/answer 等其他字段则拒绝。所有行先验证 scope，再开始模型调用。输出按 question_id 哈希命名，避免路径注入。每题完成立即落盘；失败保存调用账与失败类型，不静默继续、覆盖或自动恢复。

同题且证据包完全相同才复用回答，并保留原始调用用量以还原独立部署成本。`frontend_shared`、setup 时间、实际回答复用都有标记。P0 不分摊不存在的模型前端；P3 不计算也不收费其不使用的 cue-BM25 列。`compare` 的分臂时延不含共享前端/初始化，不能直接报为独立端到端时延。

日志包括 scope/hash、query groups、所有图路径与截断、候选池、成功入包步骤、覆盖前后状态、收益密度、真实 token 增量、BM25 扫描数、边访问数、分阶段 wall time、调用 traces、最终引用。当前 BM25 实现扫描原文索引，约 O(原文atom数×查询数)；不是规模无关或毫秒级保证。计入本地 CPU 和 pack 重渲染成本。

`eitr summarize --rows scored.json --out summary.json` 只做逐 owner、逐题配对描述。评分数据每题每臂一行：`question_id, owner_id, arm, correct:boolean, online_tokens:int`。重复回答先在题内处理；本命令拒绝将同题重复当独立样本。没有自动质量评审或非劣效宣称。

## 验证记录与下一步

见 `docs/eitr_validation_20261119.md`。下一轮效果实验前仍要：冻结正常题及协议、核验真实日期、接入真实前端和回答模型、完整实现/对齐原生 MRAgent 基线。主方法代码已经可以运行；P5_proxy 的结果必须保持独立名称，正式 P5 未完成时不能宣称六臂已全部就绪。
