# 原生 tag 选择器 v1：方案覆盖审计

> 这是修复前的历史审计。下列问题已在后续代码修复中处理；当前状态与复验见[审计修复报告](native_tag_selector_repair_20261122.md)。原复现脚本保留历史用途，不是修复版验收命令。

审计日期：2026-11-22。对象：`main_lab/eicr_v1/src/native_tag/` 当前工作区版本；对照 Research 中《20261122：保留原生图检索的廉价 tag 选择器｜方法边界与实验设计 v1》的已保存全文。

**结论：尚未覆盖全部方案，不应把当前版本视为已满足正式 Stage C 的代码验收条件。** 核心评分算法、两个原生接线点和端到端 runner 已实现，但发现 7 个可离线复现的问题，以及若干未完成的报告和验收环节。此前 332 项通过的结果成立，但不能据此推断方案全部要求已实现。

本次只添加审计报告和复现材料，没有修改生产实现或原生 MRAgent。所有复现均使用本地冻结原生代码和模拟外部响应，没有真实 provider 调用、实际费用或真实效果结论。

## 复核证据

- 重新运行全仓测试：**332 passed，0 skipped，11.60 秒**。
- 独立审计脚本：**7 个问题全部复现**；其中执行了既有的 32 题 × 2 臂 × 2 次独立完整运行模拟。
- 原生参考快照：`artifacts/native_tag_selector_v1_20261122/runtime-final/`；仍是带已披露本地修改的冻结版本，不是干净上游复现。
- [复现脚本](../artifacts/native_tag_selector_v1_20261122/audit/reproduce.py)、[结构化结果](../artifacts/native_tag_selector_v1_20261122/audit/reproductions.json)、[全仓日志](../artifacts/native_tag_selector_v1_20261122/audit/full-suite.log)、[JUnit](../artifacts/native_tag_selector_v1_20261122/audit/full-suite.xml)。这些产物所在的 artifacts 目录被 Git 忽略，应随审计证据单独保留。

复现命令（在 eicr_v1 目录运行）：

```sh
.venv/bin/python artifacts/native_tag_selector_v1_20261122/audit/reproduce.py
```

脚本中的断言确认的是“缺陷仍然存在”，不是产品验收通过。修复后应把相应场景转成期望正确行为的回归测试。

## 已复现的问题

### F1 · P1：native 开发运行可能返回 cheap 分数，并覆盖教师记录

位置：`src/native_tag/core.py:221`、`:234`、`:241`；调用入口 `src/native_tag/experiment.py:206`，协议检查 `:152`。

触发条件：development 配置包含已拟合 calibration，且 shadow 为默认 false。freeze 接受该配置；run_one 对开发阶段自动启用 collect_raw。Selector 先调用 native 得到教师响应，随后计算映射，并在非 shadow 分支把返回值和记录中的 response 都替换成 cheap 分数。

复现：教师给 `t0=0.0`，返回值及日志变成 `t0=0.8656090991705352`，记录 mode 仍是 native。该变化跨过原生 0.7 提前读取边界。`calibration_records` 又把 response 当 teacher，因此再次校准可能学到选择器自己的预测。

对应方案 §3、§4.3、Stage B：native 必须保持原评分行为，校准标签必须来自原生教师，同状态 shadow 不得改变轨迹。

修复要求：原生响应、raw、shadow 响应分别存储；只有 cheap=True 才能替换返回响应。补齐 native × collect_raw × calibration × shadow 的模式组合测试。

### F2 · P1：运行后处理异常会抹去已发生调用，并制造完整零费用记录

位置：`src/native_tag/experiment.py:223`、`:245`、`:274`；具体异常来源 `src/native_tag/transport.py:96`。

run_one 只保护原生回答部分，计费和其他后处理在 try 外。run_protocol 却把所有逃逸异常都解释为“首次模型调用前初始化失败”，重新构造 calls=[]、wall_seconds=0、billing=0 的行。

复现：模拟 provider 完成 **6 次聊天调用**，usage 中 `prompt_tokens_details` 是意外的字符串。计费读取 nested metadata 时抛 AttributeError；落盘结果却是 **0 次调用、amount=0、complete=true、phase=setup**。没有通过 monkeypatch 强制后处理抛错；实际走了当前 transport 与计费代码。

对应方案 §4.4、§6.2、§7：所有尝试和失败都需完整计费，无法判价必须 unknown，不能当零。该问题可能让成本缺失率和失败成本失真。

修复要求：从请求开始保有可恢复的 trace；后处理失败保留原始调用和耗时；对畸形 usage 返回 unknown；只有确认尚无请求发生才能记录零调用初始化失败。

### F3 · P1：Stage A 和校准器没有绑定实际教师/实验模型

位置：`src/native_tag/experiment.py:119`、`:157`、`:305`；`src/native_tag/core.py:176`。

Stage A 门槛仅与 native runtime hash 关联；校准器只验证 embedding 身份等信息，没有强制绑定教师 chat provider/model/endpoint、开发协议和运行配置。保留原来的 Stage A 与 calibration，更换后续聊天模型仍能 freeze 成功。费用表也没有在 freeze 时验证新模型确实有对应费率。

复现：把原本可通过的 evaluation spec 的模型从 fixture 改成 different-chat-model，旧 Stage A 和旧 calibration 原封不动，freeze 仍接受。

对应方案 §4.3、Stage A/B、§7：原评分尺度和经济空间依赖实际教师模型/配置，不能用其他配置的通过记录代替。

修复要求：形成可校验的开发 protocol→原生 runs→Stage A/calibration→主协议身份链。明确哪些资产变化允许迁移，哪些模型/提示/价格变化必须重新做门槛或校准，不仅校验文件自身 checksum。

### F4 · P1：分析入口丢弃判分模型和 rubric 身份，能接受未冻结 judge 的成绩

位置：`src/native_tag/cli.py:137`、`src/native_tag/experiment.py:351`；判分文件元信息存在于 `src/native_tag/grading.py:83`。

grade CLI 确实检查了 judge model；但 analyze CLI 只校验 grades 文件自身 hash，然后传入 grades['grades']。模型、rubrics、labels/protocol 关联没有传递或核验。自洽文件 hash 不能证明其符合本次冻结协议。

复现：对同一批运行用 unfrozen-judge 生成有效判分文件，按 analyze CLI 的路径读取并分析，得到 decision=continue；协议冻结的 judge 是 fixture。

另一个静态缺口：implementation_hashes 只覆盖 native_tag/*.py，而宽松 rubric 从 eitr.evaluate 导入；grade 入口也未核对当前 rubric hash 是否等于协议中通过验收的 rubric。外部 rubric 变化不一定被现有预检识别。

对应方案 §6.1、§7：必须使用冻结的同一 judge 和严格/宽松口径，不得混入另一配置的分数。

修复要求：analyze 接收整个判分产物，验证 model、rubric hash、protocol/labels/run 集合和复核来源；缺少任一强制身份就拒绝正式结论。冻结真实执行依赖中的 rubric 内容。

### F5 · P1：确认轮允许更换模型和校准器

位置：`src/native_tag/experiment.py:128`。

confirmation 分支只验证旧 Stage C 的 decision=continue 和题目 ID/文本不重用，没有载入旧协议并比较实现、校准器、rubric、模型及关键配置。

复现：先用冻结 judge 为合成 Stage C 生成正常通过结果；使用新的一批 hash 采样题，更换 chat model，再用另一组教师分数重新拟合 calibration。confirmation freeze 与 preflight 均通过。

对应方案 §8：确认轮必须冻结全部实现、校准器、rubric 和配置。当前检查仅保证新题，不能保证是同一方法的确认。

修复要求：确认协议必须引用并验证 Stage C 原协议，比较预先规定的不可变字段；只允许更换题目/相应新图资产等方案允许的字段，且记录共享图与新会话情况。

### F6 · P1：分批人工复核可破坏相同答案的对称处理

位置：`src/native_tag/grading.py:90`，尤其 `:93`、`:99`、`:104`。

apply_adjudication 的 scores 字典只覆盖当前导入批次，既不检查此前已裁定的同 cache_key 项，也不把同一裁定同步至所有相同文本。

复现：两个相同 question/reference/answer 的 strict 记录，首次导入只把第一条改为 1，第二次只把另一条改为 0。两次均成功；最终 cache_key 相同、分数为 [1,0]、unresolved 全为 false。对称上下界因此无法再发现这组人工矛盾。

对应方案 §6.1、§7：相同文本同 rubric 必须复用判断，未决分歧须对称处理。

修复要求：以 cache_key 为全局裁定单位；跨批校验已有判决，对冲突拒绝或重新标为整组未决；一致裁定同步整组，同时保留原始分数与复核历史。

### F7 · P2：judge 失败时没有持久化调用账；离线重建存在同类缺口

位置：`src/native_tag/cli.py:122`、`:134`；离线资产路径 `src/native_tag/assets.py:48`、`:53`、`:79` 与 `src/native_tag/cli.py:89`。

复现 grade CLI：strict judge 成功、lenient judge 返回非法 label。已发生 **2 次模拟聊天调用**，已有 1 个判断缓存；命令报错退出，但 judge-calls.json 不存在。费用收据仅在整个 grade_runs 成功后写入。

静态确认的同类问题：prepare_assets 将 embedding trace 留在内存，仅在全部编码、图验证成功后返回；reembed 中途失败时，CLI 不会写 embedding receipt。rubric-check 也在全部完成后才写 calls。

这些费用不应混入在线 N/S 每请求费用，但必须作为独立实验/新增离线支出完整保留。当前不能声明覆盖了所有失败路径的调用账。

修复要求：按 attempt 增量落盘，并在异常/finally 路径保存带失败状态的账单；支持明确的中断产物识别，避免重新跑时漏算已有支出。

## 方案逐项覆盖

“已覆盖”指已具备相应代码和离线测试，不代表真实数据实验已验收。

| 方案要求 | 当前状态 | 证据与剩余工作 |
|---|---|---|
| §2–3：冻结本地原生，只替换两个明确评分点 | 核心已覆盖，验证仍有限 | AST 明确接线，参考/插桩快照、diff、native/透传回放具备；仍受 F1 影响。原生长候选细选、事件重排、时间专用绕行等需更丰富的等价回放。 |
| §4.1：scope、全局事件表、完整来源集合去重 | 已覆盖 | EvidenceView 验证 owner/snapshot/graph/key/tag；仅评分视图去重；模型、维度、输入身份检查及统一重编码路径存在。 |
| §4.2：固定 top-3、非负 cosine、候选局部计算 | 已覆盖 | 不扫描所有 tag 的评分；共享候选事件点积；固定 top-3；精确上下文的本请求 query 向量复用。 |
| §4.3：请求等权 PAVA、阶段20次支持、pooled后备、≤16开发题 | 算法已覆盖，身份链不完整 | 支持条件、端点、插值和 ties 有实现/测试；F1、F3 会污染教师或允许跨配置使用。没有真实开发校准验收产物。 |
| §4.4：编码不可用 fallback、scope错误拒绝、重试账 | 部分覆盖 | 只对 EncoderUnavailable fallback，计次及正常重试有测试；F2/F7 使异常账不完整。在线才显露的向量错误会记为单题失败，未形成实现身份错误后终止整个实验的独立策略。 |
| Stage A：总体/触发子集费用占比和≥10%门槛 | 部分覆盖 | 费用门槛及总体/触发子集存在；未绑定模型配置（F3）。输出 tag HTTP 秒但未完整汇总延迟占比、下游探索费用与失败率；开发运行 wall 含 raw 计算，不能直接当纯 native 延迟分母。 |
| Stage B：native恒等、透传、隔离、工具、shadow | 部分覆盖 | 合成回放与工具测试存在；shadow 可执行，但真实开发 shadow 记录不是主实验 freeze 的强制证据。现有8项 checks 由24项离线测试生成，不能代表所有原生分支和错误路径均验收。 |
| Stage C：32题分层、128次完整执行、随机配对、状态重置 | 标准路径已覆盖 | 已重跑128次合成完整运行；历史/开发排除和固定hash采样存在。分层不足时报错，但尚无机器可读替代分层设计入口。历史检索范围靠显式输入，不能声称自动发现全部暴露。 |
| Stage C：cheap_initial 仅事先固定诊断子集 | 部分覆盖 | 只有 diagnostic_preregistered=True 布尔要求；未绑定主结果出现前已冻结的子集，也不验证是预定条件触发后的诊断。 |
| §6.1：严格/宽松盲评、去重、复核、对称界 | 部分覆盖 | 主要框架存在；F4/F6 会破坏判分身份和对称裁定。会话成本有汇总，缺少会话级质量汇总；逐题 repeat_scores 留存但未完整汇总重复波动。 |
| §6.1：答案正确、来源有效、来源支持分开 | 缺一项完整流程 | 答案成绩和 sources_valid 有字段；source_supports_answer 始终为 None（experiment.py:241），没有来源支持盲评输入、裁定和汇总流程。不能算已实现第三项。 |
| §6.2：费率身份、unknown、全部尝试、顺序 wall/CPU | 正常路径覆盖，错误路径不完整 | 正常 token/缓存计价及unknown阻断经济结论存在；F2/F7 是阻断性缺陷。调用HTTP/CPU与整体CPU留痕，选择器/索引CPU与阶段汇总尚不完整。 |
| §6.2：新增向量、索引、教师校准成本/回本/重建频率 | 部分覆盖 | analyze 的 offline_cost 是手工参数，CLI未暴露，也未自动接入embedding/校准收据；未完成索引耗时/成本汇总和快照重建频率。CLI默认结果为 offline_incremental_cost=null、break_even_queries=null。 |
| §6.3：完整机制诊断 | 部分覆盖 | 决策、工具事件、阶段费用、轮数、shadow集合差异留痕；尚缺逐类工具/访问来源/重复读取/来源召回汇总，及教师旧状态与cheap新状态的系统区分；触发子集报告目前主要是准确率。 |
| §7：15%降本、质量/失败/3会话/fallback/缺费门槛 | 数值逻辑已覆盖，证据前提有缺陷 | 配对按题平均、对称质量界、去最高成本题敏感性等存在；需先修复 F1–F6，才能信任进入门槛的数据和身份。 |
| §8：另一批32题确认且配置不变 | 未完整覆盖 | 新题、矩阵和旧Stage C通过要求存在，但 F5 直接违反配置不变。 |
| §8：第二baseline与tag文本消融 | 条件性后续工作 | 本轮没有实现；方案明确要求确认轮成功后再进入，不应作为当前最小首轮实现的阻断缺口。 |
| §9：交付产物 | 框架与合成验收具备，真实实验产物未产生 | scope、快照、CLI、测试存在；真实Stage A/B、校准、主实验、盲审和经济报告尚不存在。生成命令不等于已经完成实验。 |

## 测试结论的边界

现有 fixture 每个 key 18 个事件、每个 tag 1 个事件，主要覆盖 tag 筛选和常用工具路径；不能自动证明超过 RERANK_LIMIT 的事件读取、长初始事实细选、时间分支和全部故障路径均与原生一致。reference 与 instrumented 也共享同一私有加载器，因此回放证明的是“该加载器下的局部接线一致”，不能单独证明它和原生独立启动方式的所有行为等价。这是验收范围限制，不是已经证明下游代码被改坏。

332项全仓测试中，native_tag 专项为24项。新增复现没有修改这些测试，正是为了明确展示：原测试仍全绿，但未覆盖的条件可以违反方案。

## 建议的完成顺序

1. 修复 F1/F2，确保 native 教师身份与全部尝试费用可信；将七个复现逐项转成防回归测试，并补上离线编码中断落账。
2. 修复 F3–F6，建立 Stage A/B→主协议→判分/复核→分析→确认协议的身份校验链，检查跨批复核的一致性。
3. 补完来源支持评估、离线支出自动归集/重建频率、机制诊断，以及事先冻结诊断子集；扩充 Stage B 边界回放。
4. 重新生成实现hash和验收收据，再用冻结开发集做真实 Stage A/B。达到经济空间、校准支持和一致性要求后，才能进入付费 Stage C。

因此，当前准确的完成描述是：**核心选择器与实验基础设施已搭建并通过已有离线测试；方案完整覆盖审计未通过，正式实验有效性仍有待修复和验收。**
