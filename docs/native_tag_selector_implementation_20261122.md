# 原生 tag 选择器 v1：实现与运行说明

实现依据为 Notion Research 中 [20261122 方案](https://www.notion.so/42083f108b4f80b1bb6cdf7a8e6921d0)，并核对了前轮矩阵路由止损报告。代码位于 `src/native_tag/`，入口为 `native-tag` 或 `PYTHONPATH=src python -m native_tag.cli`。这是独立于旧 `eitr matrix-*` 检索流程的适配。

当前协议格式为 `native-tag-protocol-2`，包括审计后的来源链、失败账与复核修复。旧协议、旧校准器和旧 checks 不能通过补字段或重算 hash 冒充新验收，必须重新生成对应证据。

真实诊断之后补充了[运行问题修复](native_tag_live_code_repair_20261122.md)。CLI 在未指定 `PYTHONHASHSEED` 或指定为 `random` 时，会在处理命令前重新启动解释器并固定为 `0`；显式指定的固定种子会保留。协议同时保存种子和实际字符串 hash 探针，跨进程种子不同或仅在 Python 启动后修改环境变量都会被 preflight 拒绝。原生集合迭代和并列排序代码未改。直接执行自定义 Python 准备/运行脚本时，应统一使用 `PYTHONHASHSEED=0 python ...`，不能只在脚本内设置该变量。更新安装前也可使用 `PYTHONPATH=src python -m native_tag.cli` 进入新的启动路径。

本次交付代码和离线验收，不包含真实模型的校准、Stage A 经济结论或 Stage C 效果结果。缺失模型身份、费率、校准支持或验收收据时，主比较会拒绝冻结。文中的示例费率不提供真实金额。

## 1. 保留与替换

`runtime.py` 为当前本地 MRAgent 建立独立 `reference/`、`instrumented/` 副本，保存文件 hash、上游 HEAD、运行文件的本地 tracked diff 和 untracked 文件身份。用户的 `Reproductions/MRAgent` 不改动。副本中只有以下两个调用表达式被替换：

- `Agent.select_key_tag` 中赋给 `key_out` 的原生 `chat_text`；
- `MemoryController.query_event_keywords_single` 中对应的 `chat_text`。

AST 用来精确定位已命名的方法及赋值，再进行原位置文本替换，不对提示词作模糊匹配。接口接收原上下文、key、原有序 tag 列表、stage、scope，以及执行原调用的闭包。排序、0 分排除、TAG_LIMIT、初始阶段 ≥0.7 提前读取、去重和所有读操作都仍由原函数执行。原生扩展阶段没有提前读取分支；报告不为它虚构提前读取。

私有运行命名空间编译完整原生类、工具 schema、提示词与工具循环，不导入上游 CLI 初始化或读取其凭据。配置中的数值预算从冻结源码抽取；provider/model/endpoint 在协议中显式固定。工具、persona/topic/time/context 索引由原生 `store_event_new` 建立，不把 EITR canonical graph 当成原生图。原始问句向量输入沿用 `data/embed_rewrite.py:embed_question`；题型格式化之后的评分问句不能冒充同一输入。

**复现限制：** 这是冻结的本地修改版参考，不是干净上游论文复现。为逐次记录 HTTP 尝试，SDK 隐式重试统一为 0，保留原生外层重试与 JSON 解析重试。此项在 manifest 披露、两臂一致。主回答格式与原生循环保持不变，原生没有返回可解析答案就计失败，不补充答案修复调用。运行器目前针对方案中的 LoCoMo 正常 1–4 类题；不声称支持 LM 的独立协议。

## 2. 选择器与校准约定

`core.py` 使用一个事件向量表与 `(key,tag)→event IDs` 视图。完整来源集合排序去重，同一集合的重写事件固定选择字典序最小的 event ID 作为评分代表；该选择不依赖问题或教师分数。仅评分视图去重，原生工具仍返回原来的全部事件。保留 encoded rewrite text 与原生返回原文的区别。

仅对当前候选所需事件计算点积，共享重叠行。分数固定为 top-3 非负 cosine 的平均值，空集合为 0；float64 计分。scope、模型版本、维度或向量错误均为合同错误，不走 native fallback。只有 `EncoderUnavailable` 可触发一次原生评分闭包；底层网络重试仍分别记账。

PAVA 按每个请求总权重 1 拟合；相同原始分数先合并。initial/expansion 各要求至少 20 个有效调用并覆盖 0、(0,0.7)、[0.7,1] 教师分数；不足时使用满足同样条件的 pooled 映射，否则 `ready=false`。映射为单调分段线性插值，端点夹取。cheap 同分保持候选列表顺序，native 同分保持教师响应对象顺序。没有隐含零分阈值或其他评分器搜索。

开发题最多 16 道，按冻结 ID 和问句文本 hash 与主评估隔离。校准器固定原生源码、完整实现依赖、聊天教师模型/endpoint、查询编码器配置和执行环境，并绑定开发协议/题目/图/运行记录。它可以用于同一冻结配置的新 owner/snapshot；每次查询仍由 view 验证该图作用域。校准器保留训练 view hash，不把它误用成只允许训练图的白名单。

`shadow=true` 仅用于 development：返回原生结果，另外记录 cheap 分数、选中/零分/提前读取集合差异与来源可达性。它不改变原生状态，不能作为真实 cheap 轨迹的质量结果。

## 3. 安装与冻结原生源码

```bash
# 在 main_lab/eicr_v1 内；离线测试仅需现有 test 依赖。
python -m pip install -e '.[test,native-tag]'
native-tag snapshot --source /absolute/path/to/Reproductions/MRAgent \
  --out /absolute/path/to/new-runtime
native-tag check --runtime /absolute/path/to/new-runtime \
  --out /absolute/path/to/new-checks
```

`check` 执行真实冻结原生类的离线测试，包括完整外部响应回放、教师分数透传和 128 次完整流水线模拟。它生成带测试证据映射的 `checks.json`，不能把未执行或 skipped 测试算作通过。需从包含 `tests/` 的源码分发运行。

## 4. 原生资产与向量身份

两种资产入口：

1. `assets --spec ...`：从明确列出的七份 JSON 表生成资产，格式见 `examples/native_tag/assets.template.json`。
2. `import-cache --spec ...`：直接接收原生 JSONL rewrite/keyword 缓存与原生 embedding pickle。pickle 仅允许 NumPy 数组类型，不执行任意对象。原生缓存通常只有向量和 ID，没有模型版本；必须提供真实来源依据，不能仅凭维度或文件名作身份认定。

`import-cache` spec 字段为 `runtime, owner, snapshot, raw_text, rewrites, keywords, embedding_cache, embedding_identity, provenance`；`raw_text` 是原生 `D会话→D会话:行→speaker:text` 对象。`provenance` 必须包含 `identity_verified=true, input_texts_verified=true, identity_evidence`。输入身份必须与 rewrite 文本、event ID、topic ID 顺序全部一致。

如果身份不可确认，使用 `reembed --spec ...` 对所有事件和 topic 统一编码。spec 用 `embedding` 代替 `embedding_identity/embedding_cache`，结构与 development 模板一致。它不重新抽取事实、不改变图边；输出独立 embedding receipt，在每个 attempt 开始和结束增量落盘 `<asset>.embedding-receipt.json`，中断或失败也保留已发生调用及 CPU/wall。`assets`/`import-cache` 同样生成复用向量和本地索引检查收据。向量构建与 topic 输入仍使用原生重写文本。两臂使用同一新资产，不能混用旧 matrix embedding。

```bash
native-tag import-cache --spec /absolute/path/to/native-cache-spec.json --out /absolute/path/to/asset.json
# 或仅在需要统一补齐时运行；此命令可能调用付费 embedding 服务。
native-tag reembed --spec /absolute/path/to/reembed-spec.json --out /absolute/path/to/new-asset.json
```

## 5. Stage A / B

development spec 见 `examples/native_tag/development.template.json`。模型密钥只从命名环境变量读入；不写入协议。填写已使用开发题和可信资产路径后：

```bash
native-tag freeze --spec /absolute/path/to/development-spec.json --out /absolute/path/to/dev-protocol.json
native-tag preflight --protocol /absolute/path/to/dev-protocol.json --out /absolute/path/to/dev-preflight.json
native-tag run --protocol /absolute/path/to/dev-protocol.json --out /absolute/path/to/dev-runs
native-tag stage-a --runs /absolute/path/to/dev-runs --out /absolute/path/to/stage-a.json
native-tag calibrate --runs /absolute/path/to/dev-runs --out /absolute/path/to/calibration.json
```

开发运行在原生轨迹上收集教师请求、响应及 raw 分数。校准编码的额外支出不冒充原生基线费用。`calibrate` 输出映射与逐调用 records；无效或不完整教师响应不计支持。未知价格/usage 保持未知，tag 费用占比达不到 10% 停止，无法可靠计算则不确定。

校准完成后，在同一固定开发题 spec 中加入 `calibration` 对象及 `shadow=true`，冻结为新协议运行 shadow。不能用新题补齐样本，不能用主评估分数重新拟合。Stage B 的 `check` 检查接口行为，真实开发 shadow 检查选择差异；二者均不能证明端到端收益。shadow 运行后执行：

```bash
native-tag shadow-check --runs /absolute/path/to/shadow-runs --out /absolute/path/to/shadow-acceptance.json
```

主实验 freeze 要求该收据通过，并与当前 calibration、开发题、执行配置一致。`calibrate` 和 `shadow-check` 分别输出 `.cost-receipt.json`；开发完整请求费用按一次采集列账，包括 Stage A 共用的采集，不能再重复相加。开发 native 即使装入 calibration，也始终返回并保留 teacher_response；只在显式 shadow 下另存 shadow_response。

### 顺序工作流与失败记录

准备目录必须包含 `development-spec.json`、已冻结执行的 `development-runs/`、`development-labels.json`、`model.json`；进入主比较还需要 `questions.json`、`labels.json`、`sample-manifest.json`、`checks/checks.json`、`rubric-acceptance.json` 及各资产的 `.embedding-receipt.json`。所有协议与收据必须属于当前实现和相同进程种子。凭据由启动环境提供，目录不存密钥。

```bash
native-tag workflow --directory /absolute/path/to/prepared-diagnostic
```

工作流等待开发 manifest（最多一小时），依次检查/校准、开发判分、真实 shadow、N/S 主运行、判分、费用与来源报告。未通过门槛停止；子命令出错、适配中止、超时或可捕获的键盘中断会写入 `workflow-status.json` 的终止状态及失败阶段，不再调用依赖阶段。SIGKILL 或断电无法保证终止状态更新，应以已持久化账本判断未完成工作。已有工作流状态或产物的目录禁止隐式重跑，历史付费请求不能被覆盖。

`run` 的 `journal.json` 在批次结束时写为 `complete` 或 `aborted`；首次适配错误原因保持不变，后续未执行题继续保留在分母中且没有调用。适配中止时 CLI 返回非零退出码。原生模型输出候选之外的 tag 时，影子可达性统计按原生读取语义返回空边并记录 `native_unlinked_selected_tags`；不会把模型新增 tag 注入图或放宽评分输入的 scope 校验。

配对比较中又发现工具成功返回的普通文本含 `terror` 时会被旧的 `error` 子串检查误报。所有付费运行和冻结分析结束后，检测已改为实际记录的异常或工具返回顶层字典的 `error` 字段；解析采用 JSON / `ast.literal_eval`，不会执行工具文本。历史结果不重写，比较报告并列给出两臂原始失败标记和同一结构化规则核查结果。执行源码保存在实验目录 `execution-source/`，后续修复不能冒充历史执行版本。

## 6. 样本、主比较与确认轮

`history --spec` 接收 `{ "paths": [绝对 JSON/JSONL 路径...] }`，记录全部搜索文件 hash，并提取历史 question ID 与问句 hash。必须覆盖已知旧 matrix16/32、校准和其他已使用记录；搜索边界显式保留，不能声称零历史暴露。

`sample --spec` 字段为 `pool, history, development_ids, development_questions, owners, salt, scopes`。pool 每题包含正常 question 字段及 `reference,evidence`，scopes 为 `{owner,snapshot,source_ids}` 列表。只按非空答案、来源范围和 hash 抽样；输出 `questions.json`、`labels.json`、`sample_manifest.json`，运行器不能接收 reference/evidence。任何分层不足直接失败。若需在看结果前调整，sample spec 可提供 `design={"rationale":"调整依据", "strata":[{"owner":"...","category":1,"count":1}, ...]}`：完整列出4个owner×4类的16个分层，计数允许为0，总题量必须仍为32、4个owner均有题。该设计与候选池、hash选择结果一起进入 sample manifest；freeze 按声明设计验证，确认轮保留同样的分层数量结构。

evaluation spec 基于 development spec，替换为 32 道冻结题，`phase=evaluation, repeats=2`，加入完整 `calibration, sample_manifest, stage_a, shadow_acceptance, checks, judge, rubric_acceptance` 对象和真实 `prices`。只有 N/S，配对试次内随机臂顺序，题目组随机交错，128 次完整运行。`phase=confirmation` 另需通过的 `stage_c` 分析及其完整 `previous_protocol`，并排除其全部题目。代码逐字段核对模型、校准器、源码/实现、rubric、价格、预算、重复、随机种子、缓存和测量政策，不能用新校准器做“确认”。新owner/snapshot的资产允许变化。

若预留归因诊断，evaluation spec 必须提前包含 `diagnostic_questions`（固定开发子集，最多16题）。后续 `phase=diagnostic` 必须引用 `previous_protocol` 和真实主结果 `stage_c`，严格使用该子集，并且主结果确有 S 严格正确率下降。布尔 `diagnostic_preregistered=true` 不再代替预注册证据。

每次请求新建原生 agent/controller，重新执行解析、初始检索、动态工具到回答。每次独立加载图与 encoder；主 wall time 如实包含这些初始化，不声称是暖部署的速度。无跨请求查询/聊天缓存，仅复用同一次请求中逐字相同上下文的已有向量。服务端 prompt cache 用量单独记录。模型失败、解析失败、工具失败、初始化失败均保留计划分母；完整计划缺行、重复或 hash 不符不能汇总。每请求的 `ledgers/NNNN.json` 在调用前后原子更新；后处理异常保留调用，畸形 usage 为 unknown。图/向量身份等适配错误会中止后续调用，剩余计划行明确标为未执行/aborted，分析只能给不确定结论。

## 7. 计费、判分、复核和分析

价格表绑定 provider/model/endpoint/billing kind/币种/日期/来源，见 `prices.template.json`；模板 null 金额不能用于正式比较。local embedding 的 API 成本显式为 0，CPU 时间另列，不声称机器计算免费。全请求中任一调用费用未知则均价未知，已知小计不冒充完整账单。当前采用保守策略：缺少可靠上界时不宣布经济门槛通过。

```bash
native-tag rubric-check --runtime /absolute/path/to/runtime --judge /absolute/path/to/judge.json \
  --cases examples/native_tag/rubric-development-cases.json --out /absolute/path/to/rubric-check.json
native-tag grade --runs /absolute/path/to/main-runs --labels /absolute/path/to/labels.json \
  --judge /absolute/path/to/frozen-judge.json --out /absolute/path/to/grades
native-tag adjudicate --grades /absolute/path/to/grades/grades.json \
  --mapping /absolute/path/to/grades/adjudication/private-mapping.json \
  --reviewed /absolute/path/to/completed-blind-review.json --reviewer 'independent reviewer' \
  --out /absolute/path/to/adjudicated-grades.json
native-tag source-export --runs /absolute/path/to/main-runs --out /absolute/path/to/source-review
# 独立审阅者填写 blind.json 的 verdict/reason，不能把自动重判冒充人工审阅。
native-tag source-review --bundle /absolute/path/to/source-review/private.json \
  --reviewed /absolute/path/to/completed-source-review.json --reviewer 'independent reviewer' \
  --out /absolute/path/to/source-assessments.json
native-tag offline-costs --spec /absolute/path/to/offline-costs-spec.json --out /absolute/path/to/offline-costs.json
native-tag analyze --runs /absolute/path/to/main-runs --grades /absolute/path/to/adjudicated-grades.json \
  --offline /absolute/path/to/offline-costs.json --source-review /absolute/path/to/source-assessments.json \
  --out /absolute/path/to/analysis.json
```

strict 要求全部项目、日期和必要限定；partial 不给分。lenient 为原 MRAgent 宽松口径。judge 只看 question/reference/answer，同文本同 rubric 和 model 复用一次判断，保留原始结果。盲审包不含方法、费用或轨迹，private mapping 独立存放。人工复核必须另写新产物；自动重判不能标记为人工复核。未决相同文本共享一个未知判断，质量差值上下界按两臂贡献系数对称计算。分批复核以 cache_key 为整体更新，保留首次 raw_score 与复核历史；第二批若与已有裁定冲突会拒绝。analyze 接收整个判分产物并验证协议、模型、rubric、完整两种评分及标签，不再接收裸 grades 列表。

来源支持通过独立 source-review 流程评估：盲包只含题目、答案和引用原文，裁定为 supported/unsupported/unresolved，同文本来源组合共享判决。它与答案正确率和来源 ID 有效性分别汇总；未复核的项保持 unresolved，不填造支持结论。

判分 attempt 账写到输出目录旁的 `.judge-calls.json`；rubric-check 写 `.calls.json`。调用开始即落盘，即便返回非法评分、发生异常或中断也保留账。

分析按题内两次平均后跨题汇总，包含逐题胜平负、owner/题型、触发子集、失败、独立延迟、费用分阶段和最高成本题敏感性。继续条件固定为完整平均费用下降 ≥15%、严格质量不降、至少 3/4 owner 降本、失败率不升、fallback ≤10%、缺失比例 ≤5%，且费用可完整核算、未决复核不反转结论。它是资源投入门槛，不是显著性或非劣效检验。

正式分析必须提供 receipt 汇总的 `--offline`。配置示例见 `examples/native_tag/offline-costs.template.json`：列出所有 embedding/index、calibration、shadow 收据（包括失败重试批次），可另列 judge/rubric_check 收据。按 ledger_id 去重拒绝重复账，缺失 usage 仍为 unknown；核对 asset/calibration 身份及开发、shadow账。复用向量也需对应的零新增embedding收据，而不是省略该项。

新增向量、索引与教师校准/影子运行 API 成本汇总为 B，计算 Decimal 向上取整回本量；无正节省则无正向回本。judge/rubric费用独立列账。CPU/wall与索引检查阶段时间另列；本地机器费用不伪装成服务API费用。必须声明快照重建 trigger、period、expected_builds 与 queries_per_period；未知可为 null，此时不计算长期摊销。当前重建预测明确针对整份增量收据组合。分析同时生成 `.json` 和 `.md` 报告，保留触发子集成本、owner质量/重复分歧、来源召回诊断、重复返回、各类工具、旧shadow状态/新状态及阶段耗时。阶段CPU/wall为包含子调用的区间，不可相加当总延迟。

## 8. 文件划分

- `core.py`：作用域、表示、PAVA、native/cheap_initial/cheap_all、shadow 与 fallback。
- `runtime.py`：源码快照、两处接线、完整原生类加载、HTTP attempt 记录、工具/读状态追踪。
- `assets.py`：原生缓存导入与统一重新编码。
- `transport.py`：在线编码、模型身份、分服务价格与未知费用处理。
- `experiment.py`：历史审计、分层采样、冻结/预检、独立 runner、Stage A 和配对分析。
- `grading.py`：完整性 rubric、盲评缓存、人工复核及对称敏感性。
- `reporting.py`：离线收据归集、重建摊销、机制诊断和可读报告。
- `cli.py`：上述可执行工作流。

所有真实资产与运行结果继续放在 gitignored `artifacts/`；测试合成数据不是实验效果证据。
