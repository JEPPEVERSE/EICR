# 原生 tag 选择器：审计修复与复验

本报告对应 2026-11-22 的[方案覆盖审计](native_tag_selector_audit_20261122.md)。修复覆盖全部 7 个已复现问题，以及该审计列出的本轮代码/工作流缺口。研究方案、top-3 公式、PAVA 条件、原生读取/工具/回答流程和经济门槛未改。原用户 MRAgent 目录未修改。

代码验收与真实实验验收分开：本次没有调用真实 provider，没有编造 Stage A 经济空间、真实校准、质量/降本或人工盲审结论。真实 Stage A/B 的数据证据仍是进入正式 Stage C 的必要条件。

## 七个问题的关闭证据

| 编号 | 修复 | 复验 |
|---|---|---|
| F1 native 教师被覆盖 | teacher_response 独立存储；只有 cheap 分支可以替换返回值；校准只读原生教师字段 | 3组collect_raw/shadow组合；合法冻结配置下，有/无calibration的native完整工具轨迹一致 |
| F2 异常后费用归零 | 每attempt前后原子保存账；后处理失败保留原trace与耗时；畸形usage保持unknown；非初始化故障不再伪装setup | 6次模拟chat完成后畸形usage仍保留6次；强制后处理失败保留全账；未知费用阻止经济结论 |
| F3 开发门槛与教师身份未绑定 | Stage A/calibration绑定开发协议、运行hash、题目/图、源码/实现依赖、聊天模型/endpoint、embedding配置、环境；Stage A另绑价格表 | 更换chat模型、encoder配置、教师来源或shadow收据均拒绝；合法协议通过 |
| F4 未冻结judge成绩能进入分析 | analyze必须接收完整判分产物；验证protocol/model/rubric/labels/run集合/两种rubric和文本缓存身份；eitr.evaluate与eicr.contracts纳入实现hash | 裸成绩列表、换judge、换rubric、同文本矛盾评分均拒绝 |
| F5 确认轮配置可漂移 | 引用完整previous_protocol及其Stage C结果；逐字段核验模型、calibration、实现、环境、rubric、价格、预算、种子、重复和测量/缓存政策；采样结构不变 | 新题且配置一致通过；换模型、校准器或随机种子拒绝 |
| F6 分批复核不对称 | 以cache_key整体裁定，跨批检查已有判决；传播到全部相同文本；保留首次raw_score及review_history | 首批裁定整组同步，第二批冲突拒绝，一致追加不覆盖原始分数 |
| F7 judge/离线失败丢账 | grade、rubric-check、assets/import-cache/reembed均有增量持久化账，异常/finally写失败状态 | strict成功后lenient非法：2次调用仍落盘；第二次embedding失败：前后attempt和耗时均保留 |

## 审计中其余缺口

| 项目 | 当前实现 |
|---|---|
| Stage B 的真实开发shadow门槛 | `shadow-check`生成带calibration/开发题/图/执行身份的收据；主协议必须引用通过收据。实际开发数据仍需真实运行生成。 |
| 预先固定的cheap_initial诊断子集 | evaluation协议预存diagnostic_questions；diagnostic必须精确复用该子集，引用原协议/结果，并满足主臂质量损失触发条件。 |
| 分层不足时的预声明调整 | sample的design显式列出16个分层及调整原因，仍为4个owner/32题；freeze核验实际计数；确认轮核验相同计数结构。 |
| 适配身份错误中止 | 查询向量在原生读取/聊天前校验；图/向量合同错误中止后续请求；剩余计划行标aborted保留分母；结果为inconclusive。 |
| 来源是否支持答案 | source-export产生仅含题目/答案/引用原文的独立盲包，source-review导入supported/unsupported/unresolved并绑定原运行；分析与答案成绩、来源ID有效性分列。没有人工结果时明确保留unresolved。 |
| 离线费用与回本 | assets/import-cache/reembed、calibrate、shadow-check生成收据；offline-costs归集新增费用并拒绝重复ledger；analyze强制核对资产、calibration、开发/shadow收据；CLI直接输出B与回本量。judge/rubric支出另列。 |
| 重建频率 | 强制声明trigger/period；expected_builds、queries_per_period可明确为unknown，未知时不虚构长期摊销；已知时报告整份增量收据组合的重建摊销和净节省。 |
| CPU/HTTP/索引时间 | 记录整体wall/CPU、每调用HTTP/CPU、选择器额外计算、索引检查和包含子调用的阶段区间；报告明确嵌套区间不可累加。 |
| Stage A 延迟和下游费用 | 同时报HTTP占比、剔除额外raw计分时间后的native wall占比估计、下游工具循环费用和失败率。 |
| 质量/机制诊断 | 增加owner质量/重复分歧、触发子集费用/失败率、逐题费用配对差、阶段候选/选中/零分/替换/fallback、逐类工具、带scope访问来源/事件、请求内重复返回、来源召回诊断和旧shadow状态/新状态区分。 |
| 可读报告 | analyze输出JSON及同名Markdown；保持未知、未复核和研究主张限制。 |
| 原生分支验收不足 | 增加超过K2/RERANK_LIMIT的细选、重排、事件读取和时间专用路径对照；独立执行未改写的参考模块，验证私有AST加载器下的完整请求回放一致。 |

独立原生加载测试仅替换离线执行所需的进程配置、dotenv及SDK构造；原生模块导入、顶层赋值、函数默认值和类体原样执行。它提高了加载器等价性的证据强度，仍不是对所有输入的形式化证明。项目限定LoCoMo 1–4类题，未扩展到LM协议。

## 运行和迁移

参见[更新后的运行说明](native_tag_selector_implementation_20261122.md)与 `examples/native_tag/offline-costs.template.json`。

协议升级为 `native-tag-protocol-2`。旧的无来源校准器、裸grade数组和缺少新增checks的验收收据不能用于正式比较。应按当前实现重新生成check、开发校准/Stage A、shadow、rubric和协议；不应给旧收据补字段、重算hash后冒充新验收。历史实验和旧审计复现脚本保留，不作为修复版验收命令。

新增入口：`shadow-check`、`offline-costs`、`source-export`、`source-review`。`analyze`现在要求`--offline`，可提供`--source-review`；确认/诊断spec需要`previous_protocol`。调用账在`ledgers/`或输出文件旁的`.embedding-receipt.json`、`.judge-calls.json`、`.calls.json`中增量保存。

第二baseline和tag文本消融仍按方案安排在真实确认轮通过之后，未提前添加。

## 复验产物

全仓回归 **349 passed，0 skipped**；native_tag专项 **41项**。sdist/wheel构建成功，CLI和diff格式检查通过。

机器可读验收、完整测试与构建记录保存于 `artifacts/native_tag_selector_v1_20261122/repair/`。最终测试数量、实现hash及收据位置以该目录的 `manifest.json` 为准。原 `acceptance/` 和 `audit/reproductions.json` 是修复前历史证据，不能代表当前版本。
