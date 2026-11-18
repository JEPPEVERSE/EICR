# 小规模开发诊断：2026-11-18

这是开发诊断，不是论文确认实验。使用已暴露的 `entry_dataset_v41flash_20261110/frozen_v0.1`（29 个题族，每组 OG/P0/P1），不打开封闭确认集。

## 冻结边界

- 阈值选择：conv-41、conv-43、conv-44，共 16 个题族、48 条输入。
- 效果检查：其余 6 个 conversation，共 13 个题族、39 条输入。原题与扰动题始终同组。
- 只使用冻结的 MRAgent 原始 pack；所有原文来源合法，评估范围为完整开发记忆，不声称时间前缀评估。
- 运行时只读取问题和许可记忆；标签仅进入阈值选择、事后评分，不能进入抽取器和回答器。
- 主干为 BM25-only；旧 pack 的向量是重写事实向量，不冒充新原文 chunk 的向量。
- 每臂最多 6,000 个实际 Qwen tokenizer token 的渲染证据。chunk 上限 384、重叠 32，问题和系统提示不计入这个证据预算。
- 模型 qwen3.5-27b，temperature=0、enable_thinking=false；同题同证据复用答案。同一个回答在不同臂间复用评判，避免重复生成和评判的随机差异。

## 对照臂

| 臂 | 预算与处理 |
|---|---|
| original | 原问题 BM25，6,000 token |
| cue_union | 原问题保留 3,000；唯一映射的 entry cue 下游原文并集 3,000，按原问题 BM25 排序；剩余预算共享 |
| direct | 直接证据向量诊断；通过冻结策略才启用检查路由，否则完整复用 original |
| structure | 单步结构扩展，beta=0.25、lambda=0.2；其余同 direct |

固定 cue_union 是一个简单的图召回对照，既不判断错误 cue，也不使用共识分数。不能把其效果称为 EICR 共识效果。

## 阈值选择

网格：tau_delta∈{0,0.05,0.15}，tau_core∈{0.05,0.15,0.3}，tau_margin∈{0.01,0.05}，tau_pair∈{0.05,0.15,0.3}。

只在 48 条选择输入上计算：正确定位的单入口检查 + 0.25×包含扰动入口的双向检查 − 错位检查。要求 OG 零检查且目标严格为正；同分优先检查更少，再优先较严格阈值。没有合格项时禁用策略。该规则是探索性的检测代理目标，不是概率校准，也没有直接优化答案分数。

另保存一个不部署的宽阈值探针，用于统计硬门控阻塞原因。它不生成答案、不重新分配预算，也不代表正式策略。

## 输出与评价

- 严格 source recall：标注的 answer_facts 来源 ID 中，完整出现在最终证据包中的比例。它是原文来源覆盖指标，不等于事实理解或证据充分率。
- 答案评分：同模型、隐藏方法身份的临时评判。必须覆盖全部 required_answer_points；扰动题还须明确纠正或限定错误前提。仅拒答或仅指出错误前提不算回答正确。
- 逐条件统计 OG/P0/P1，报告 paired wins/losses；39 条并非 39 个独立题族。
- 单独统计输出格式失败、唯一标签映射率、完整映射覆盖、门控原因、实际路由、实际 token、成功调用及服务报告用量。
- 这些判断需结合逐题复核；同模型评判不是独立人工验收。

## 运行

先安装项目测试与 pilot 依赖，并准备已授权的本地数据、凭证文件及 tokenizer。`--tokenizer` 指向 SHA256 为 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` 的 Qwen tokenizer JSON。

```sh
python -m pip install -e '.[test,pilot]'
PYTHONPATH=src python scripts/run_development_pilot.py prepare --workspace /path/to/papers --output artifacts/new_pilot --tokenizer /path/to/tokenizer.json
```

随后以相同参数顺序执行 `preflight`、`frontend`、`calibrate`、`retrieve`、`answer`、`judge`、`summarize`。`preflight` 同时检查 API 和真实问题的入口抽取。数据文件哈希、源代码哈希、选择规则和提示词哈希保存在 protocol.json。不要覆写旧协议；更换协议应创建新输出目录。

请求有持久缓存，每个输出目录最多尝试 400 次请求，4 个线程。transport 失败使阶段失败，不能当成科学样本的失败继续评分。原始数据、缓存、回答和本地索引均写入被 Git 忽略的 artifacts/，凭证只从环境或显式本地 .env 读取，不进入日志。

## 接口诊断轮次

第一轮 v1 保留原抽取提示和强制 JSON 模式，87/87 前端失败：73 条顶层不是对象、14 条字符位置不匹配。v2 加入具体对象模板并使用确定性原文定位，仍有已完成的 46/46 顶层对象失败，随后终止。

单题抓取到服务返回内容为 JSON 字符串 `" { "`，finish_reason=stop、completion_tokens=4。关闭供应商 response_format 强制模式后，同一抽取请求正常返回对象。因此 v3 明确关闭供应商 JSON 模式，但客户端继续严格解析 JSON 对象。此观察仅针对本次模型/端点/配置，不宣称所有 Qwen 调用都有同样问题。

位置修复只接受原问题中唯一出现的完全一致文本；多次出现必须有有效位置，缺失/改写文本不猜测。scope 定义也明确限制为第一人称记忆所有者，避免把场景对象误排除在 entry 之外。共识公式、索引构建和门控在接口修复期间未改变。

## 评分器复核（事后补充，不替换冻结记录）

回答完成后发现引用序列化错误和自动评分错误，增加两份独立审计。它们只使用已有缓存答案，不重新生成回答，不改阈值，也不改变检索结果。

```sh
PYTHONPATH=src python scripts/audit_pilot_outputs.py --workspace /path/to/papers --output artifacts/new_pilot
PYTHONPATH=src python scripts/rescore_pilot.py --workspace /path/to/papers --output artifacts/new_pilot
```

第一份审计将严格输出校验与答案内容判断分开。仅当错误的 source_id 完全等于已呈现的某一引用头时，才尝试机械地还原其 source_id；随后仍调用原来的完整引用范围校验。不匹配的 ID 和越界坐标不会被猜测或修复。

第二份审计修正评分器把 OG 当成扰动题、遗漏参考会话日期的问题。使用同一模型、隐藏方法身份，分开输出每个必需答案点、事实矛盾、是否承认错误前提、是否明确给出正确替代关系。纠错判定必须给出来自候选答案的精确引文；最终是否正确由程序按这些字段计算。OG 无需纠错；回答绝对日期的问题不能只回答“昨天”或“上个月”。同题相同 answer_text 复用评判。

原始 summary.json 和 judgments/ 保留，不能把错误评分器产生的准确率当成有效实验结论。最终报告的回答文本分数标明为“修正评分器后的事后自动评分”，不是预注册确认结果，也不是人工标注。输出契约通过率仍独立报告。

最后 ES027_P0 的评分格式连续失败，按已缓存答案补充一条显式助手审定记录，保存于 score_overrides.json，并以问题 ID 与答案内容哈希绑定。原回答只给出“last month”，缺少标注要求的 September 2023 及推导日期，故记为未完成必需答案点。它不是成功的模型评分，也不是人工审核；汇总文件显式列出该题。
