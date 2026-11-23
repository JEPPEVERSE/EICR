# EICR

2026-11-23：已回退到答案优化之前的 native-tag 实现，保留廉价 tag 选择器和原生回答。修正前 12 个实现文件指纹完全一致，回退后 371 项测试通过；答案候选及负结果归档保留。完整请求的成本优势与质量折中、缓存敏感性和离线摊销见[质量—成本分析](docs/native_tag_quality_cost_20261123.md)。

**2026-11-22：当前研究方向为「保留原生 MRAgent，仅替换 tag 评分/选择」。** 作用域为 `(key, tag)`；原生事件读取、时间/上下文/topic/persona 工具、动态探索和回答保留。独立 `native-tag` 入口、原生两处评分接线、PAVA 校准、冻结运行与盲评报告见[实现与运行说明](docs/native_tag_selector_implementation_20261122.md)。[实验设计方案](docs/native_tag_selector_experiment_v1_20261122.md)与[设计参数](docs/native_tag_selector_experiment_defaults_v1_20261122.json)保留。**同标准 N/S 比较已完成 32 题、128 次请求：平均 API 费下降 59.82%，原始严格正确率 N 45/64、S 42/64；未决复核使结论不确定，未进入确认轮。** 详见[配对比较报告](docs/native_tag_comparison_report_20261122.md)。首轮因影子统计错误停止的[诊断报告](docs/native_tag_diagnostic_report_20261122.md)及[代码修复](docs/native_tag_live_code_repair_20261122.md)保留；此前覆盖审计见[审计修复报告](docs/native_tag_selector_repair_20261122.md)。

**实验基础设施已补齐（0.3.2）**：全量精确 tag top-k、近似图漏边对照、独立请求 benchmark、预热/重复/失败处理、多模型服务计费、离线成本摊销、重复题目统计与分批 embedding 恢复。运行方式见[基础设施说明](docs/matrix_infrastructure_v032_20261121.md)。本轮只做基础设施验证，未运行真实模型效果实验。

**2026-11-21：当前主线升级为 v0.3.2「问题条件下的 tag 矩阵检索＋成熟重排器」。** 使用 `eitr matrix-*`：一次问题需求解析 → 下游事实多行矩阵匹配 → 固定预算候选 → 原文重排、装包和回答。tag 图是可选消融；当前主线不再使用委员会和入口重要性动态配额。包括 live embedding 接口、协议冻结、批量实验、完整回放和隔离评分。本轮复核修复了图扩展过滤、缓存预检、计划记录与评分校验问题，见[审计修复报告](docs/matrix_v3_review_20261121.md)。详见[数学与运行说明](docs/matrix_v3_implementation_20261121.md)、[实现验收报告](docs/matrix_v3_completion_20261121.md)。

```bash
python examples/matrix/offline_smoke.py --out /tmp/new-matrix-smoke
```

上面是无 API 的合成链路检查，不是效果实验。真实服务配置模板位于 `examples/matrix/`。**下面的 P*/C* 和 eicr 内容均为历史版本，保留复现，不代表 v3 主线。**

**历史 v2（2026-11-19）：研究主线曾转向 EITR 图记忆低成本检索，版本 0.2.0。** `src/eitr/` 实现下游 cue 分布、tag 关联图、动态覆盖调度、P0–P5、离线原文索引、协议冻结、回放和独立评分。P5 加载校验过的 MRAgent 实际工具循环；原生 M0 不混入受控消融。使用 `eitr` 命令，详见 [运行说明](docs/eitr_implementation_v2.md)和[完整性验收](docs/eitr_completion_audit_20261119.md)。下文与 `eicr` 命令保留旧入口扰动研究，两个索引格式不可混用。

**2026-11-19 级联检索适配：** 新增专用语义 reranker、同候选池 MMR 对照、独立 PPR 候选臂、MRAgent 风格关键词→静态 tag 前端，以及严格/上游宽松双评分口径。旧 P0–P5 保留；使用方式与真实验证见 [级联实现报告](docs/eitr_cascade_report_20261119.md)。

**16题真实验证：** 固定候选池下，语义精排使严格答对从8/16增至10/16；MMR与PPR未增加答对题数。引用合同、双口径评分与成本限制见 [实验报告](docs/eitr_pilot16_report_20261119.md)。

Evidence-Induced Cue Representation：从已有 cue 的下游证据计算稀疏表示，用于入口扰动下的关联诊断与有限预算查证。

Cue 是已有的语义标签。其向量来自 cue–evidence 关联，不通过 cue 文字的 embedding 编码器生成。一个 cue 可以同时连接多个情境；系统不要求先构造完整事件实体。Cosine 表示这份记忆中的证据关联，不能直接解释成入口错误概率或同事件概率。

## 已实现

- 冻结 MRAgent pack 导入、ID 连接、合法来源/前缀检查、精确去重与来源族。
- 稀疏直接表示 `X = R sqrt(W)`；可选 native-topic 一步结构表示 `Y = XT`。
- 弱边衰减自环、分通道归一化、固定 lambda 的余弦混合、来源贡献解释。
- 已有 cue 的精确/显式别名映射；歧义和未映射保留；相同行合并诊断。
- 方向覆盖、共同支持、leave-one-out 查证优先级、默认关闭的冻结策略。
- 原问、答案需求、候选核心和独立 cue 队列；实际渲染 token 配额与跨度去重。
- BM25，以及接受显式缓存向量的 dense+BM25 RRF；分块/问题/向量对齐清单。
- 可选 OpenAI-compatible 问题前端与原文回答接口；引用范围校验与脱敏调用记录。
- 索引保存/哈希审计、CLI、合成示例、数学/反例/端到端测试、CI。

当前是研究实现。未选择开发集路由阈值、未宣称概率校准，也未运行真实模型效果实验。默认路由 `ORIGINAL`，仍输出全部可用诊断。历史设计文档保留其当时的“尚未实现”状态；实现与规格的对应和边界见 [实现说明](docs/implementation.md)。

`0.1.1` 修复了结构索引保存、跨问题诊断混用、来源隔离、HTTP 重定向认证头风险及多项输入/预算问题，见 [代码复核报告](docs/review_20261118.md)。索引格式升级为 `eicr-index-2`，策略格式为 `eicr-policy-2`；请重建旧索引、策略和 dense bundle。已存在的输出不会自动覆盖。

## 安装与验证

需要 Python ≥3.10。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,tokens]'
pytest -q
```

完全离线、无 API 调用的合成示例：

```bash
eicr build --snapshot examples/synthetic_snapshot.json --config configs/direct.json --out artifacts/direct
eicr audit-index --index artifacts/direct
eicr diagnose --index artifacts/direct --question 'halloumi pinch paprika shopping' --out outputs/diagnostics.json
eicr run --index artifacts/direct --question 'halloumi pinch paprika shopping' --tokenizer characters --out outputs/evidence.json
```

`characters` 是 Unicode 字符计数，仅用于测试与演示，不能冒充语言模型 token。输出包含映射、分数、路由理由、候选原文、实际证据包和预算账；不指定模型时 `answer=null`、状态为 `evidence_only`。输出路径已存在时拒绝覆盖。

比较可选结构配置：

```bash
eicr build --snapshot examples/synthetic_snapshot.json --config configs/native_topic.json --out artifacts/native-topic
eicr audit-index --index artifacts/native-topic
```

直接主配置 lambda=0；结构候选 beta=.25、lambda=.2。这些是设计起点，不是已证明最优值。

## 输入格式

[examples/synthetic_snapshot.json](examples/synthetic_snapshot.json) 给出了完整的最小输入。核心字段：

```text
owner_id, snapshot_id
sources[]: source_id, text
cues[]: cue_id, text, aliases?, scope_role?
evidence[]: evidence_id, text, origins[], tags?, topics?, metadata?
edges[]: cue_id, evidence_id, edge_origin?  # v1 二值直接边
provenance: allowed_source_ids[], extraction_prefix_verified
```

`aliases` 必须是 `{text, source_ids}` 的显式记录；不按文本相似合并实体。`scope_role="scope"` 的 owner 等节点不投票。topic ID 应已包含 owner/session 身份，不能跨会话同名合并。来源族是显式来源重叠形成的连通分量，不是事件，也不是独立性证明。

`extraction_prefix_verified` 是数据提供者对抽取所见来源范围的声明；程序检查记录一致性，不能独立证明一份虚假的声明。事实的全部 origins 必须合法；未来事实整体排除。别名/派生链接越界、悬空 ID 和来源缺失使构建失败。没有 keywords 的原文仍可用于原问检索。

从冻结 MRAgent pack 导入：

```bash
eicr import --pack /path/to/conversation-pack.json --manifest /path/to/legal-prefix.json --out data/snapshot.json
eicr build --snapshot data/snapshot.json --out artifacts/conversation
```

manifest 必须包含 `owner_id`、`snapshot_id`、`allowed_source_ids`、`extraction_prefix_verified=true`；可提供 `attestation` 和 `scope_cue_texts`。导入不执行 MRAgent Agent，不自动补 speaker 边，不使用其缺失时间回填作为真实事件时间。

## 前端与路由

默认前端是保守的已有标签字面扫描，适合调试，不能保证完整理解自然语言。`--frontend` 可传入含 `mentions`、`answer_need`、`query_shape` 的 JSON；mentions 使用原问题的 Unicode 码点半开区间。模糊候选不投票，未映射不等于入口错误。

至少三个有效诊断组时计算：

```text
check_priority(i) = 其余组的加权内部关联 − i 与其余组的加权关联
```

完整相同的 R 行只算一份结构观察。两组只能对称查证；精确包含、歧义、截断、多目标输入、同分等均有明确降级路径。没有冻结 policy 时始终保留原问策略。

开发集完成选择后，提供一个包含 `thresholds` 和 `selection_manifest` 的 JSON：

```bash
eicr freeze-policy --index artifacts/conversation --selection /path/to/development-selection.json --out outputs/policy.json
```

四个阈值为 `tau_delta`、`tau_core`、`tau_margin`、`tau_pair`。选择清单须含 `status="development_selected"`、`development_split_hash`、`selection_rule`、`grouping`、`confirmation_untouched=true`。该命令验证并冻结选择记录，不会自行完成调参，更不证明所提供的声明真实。policy 绑定确切 index hash；不兼容时走原问。

## 原文预算与模型

| 模式 | 原文 token 配额 |
|---|---|
| ORIGINAL | 原问 6000 |
| SYMMETRIC_CHECK | 原问 4000；两侧各 1000 |
| SINGLE_CUE_CHECK | 原问 3000；答需、核心、独立 cue 各 1000 |

只合并实际选中的重叠跨度，按新增渲染文本计费。相同来源的未覆盖跨度仍收费。独立 cue 队列不加其他 cue 的日期/地点硬过滤，也不按核心相似度删除反证；按来源族轮转，有限预算仍会遗漏证据。

真实模型运行必须指定对应的 tiktoken encoding；若目标模型使用另一种 tokenizer，在 Python API 传入具有 `count(text)` 和固定 `identity` 的计数器。编码初次加载可能需要下载 tokenizer 词表；离线时应预先缓存。

```bash
export OPENAI_API_KEY='your-key'
eicr run --index artifacts/conversation --question 'Your original question' \
  --tokenizer cl100k_base --api-base https://your-provider.example/v1 --model YOUR_MODEL \
  --llm-frontend --generate --out outputs/answer.json
```

仅指定 `--llm-frontend` / `--generate` 才会调用相应接口。默认每次最多一次回答，前端最多一次格式重试。模型、endpoint、凭证环境变量由用户明确提供；代码不读取本地 `.env`、不写凭证。兼容接口须支持 JSON object 与 `max_tokens`。问题与记忆会发往该显式指定的 endpoint。

HTTP 重定向一律拒绝，需填写服务的最终 API 地址。网络/认证配置失败不作为格式错误重试；响应体默认上限4 MiB，仅接受正常完成的 JSON 对象。最终原文块按装包结果原样交给模型，队列归因留在本地 trace。

Python `ChatBackend` 可显式设置 `temperature=0`、`enable_thinking=False` 和 `json_mode=False`。最后一个选项只关闭服务端强制 JSON 模式，客户端仍严格要求 JSON 对象；本次 Qwen 开发诊断使用了这一兼容设置。LLM 前端可返回原问题中的逐字文本，由程序定位唯一出现的位置；重复文本没有有效位置时仍拒绝猜测。

默认检索是 BM25，输出会标 `bm25_only`；不会声称已运行 dense RAG。启用 hybrid 时传 `--dense bundle.npz --dense-manifest bundle.json`。bundle 的数组为 `chunk_vectors` 和 `query_vector`；manifest 须含 `npz_sha256`、`index_hash`、`tokenizer`、`chunk_order_hash`、`question_hash`、`dense_identity`。生成方式见 [实现说明](docs/implementation.md)。

## 文档与研究边界

- [详细设计 v1（中文，历史设计）](docs/design_v1_zh.md)
- [最初的可行性与图结构说明](docs/concept_note.md)
- [历史设计参数](docs/design_defaults.json)：不是可执行 IndexConfig。
- [实现对应、接口与限制](docs/implementation.md)
- [验证报告](docs/validation.md)
- [0.1.1 代码复核与修复记录](docs/review_20261118.md)
- [小规模开发诊断协议与复现](docs/pilot_protocol_20261118.md)
- [小规模测试结果](docs/pilot_results_20261118.md)

开发诊断脚本位于 `scripts/run_development_pilot.py`，单独安装 `.[pilot]` 依赖；该显式启动的脚本可从指定工作区的 `main_lab/.env` 读取 QWEN_API_KEY。本地数据、模型回答和调用缓存保存在被 Git 忽略的 artifacts/。

离线评价只汇总外部审定的 `correct` 标签，`audit-splits` 检查 memory/question-family 泄漏；不会以词面重叠假装完整 QA 评测，也不会从无标签数据产生错误概率。

同图输入无法恢复未保留的动作、主客体、否定、计划/完成或时间归属。多个 cue 共同指向错误但自洽的经历也可能检测不到。方法价值需要通过同来源、同前端、同实际预算的强对照验证。

## 继承与引用

Cue 概念和冻结资产适配基于 [MRAgent](https://github.com/Ji-shuo/MRAgent)。此项目不包含其语料、缓存、第三方源代码或模型权重；导入器依据冻结记录合同独立实现。加权关联、cosine、Bhattacharyya/Hellinger 与图平滑均有既有研究基础，方法名不构成首创声明。后续论文应补充正式文献核查与引用。
