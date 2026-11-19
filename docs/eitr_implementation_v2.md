# EITR v2：实现与运行说明（0.2.0）

2026-11-19。主线为正常图记忆 QA 的成本—质量优化，不做入口纠错。实现依据研究设计 v2；第一版交付遗漏及本次修复逐项记录于 [完整性验收](eitr_completion_audit_20261119.md)。此前 114 项测试和冒烟记录是历史状态，不能当作本版完整验收。

## 1. 已实现范围

| 模块 | 当前实现 |
|---|---|
| R_cue | 事实唯一 cue 均分 → 完整 origin 集合内平均 → 来源组间平均 → IDF/sqrt/L2 |
| R_fact | 对齐的事实向量单位化 → 同来源组聚合 → tag 单位化；附向量对齐命令 |
| 关联索引 | 正相似度有向 top-k，零向量不补边，主要共享坐标可追溯 |
| 入口 | 一次提取、一次有界格式重试、原问题 span 校验、廉价原生 key/tag 映射 |
| 委员分组 | 规范化表面文本去重；唯一精确/已验证别名指向同 key 时合并；歧义/模糊匹配不强行合并 |
| 候选枚举 | 分组轮转、组内最大路径、保留 depth 状态、完整邻接行计费、零输出桥接 |
| 动态选择 | 真实原文、固定邻接上下文、精确覆盖边际、整题相关性、实际渲染成本、预留回收 |
| 对照 | P0–P5、去掉支持/覆盖两项变体、strict_seed；P2/P3/P4 固定同一候选池 |
| 原文基础设施 | atom/BM25 离线缓存；原文/跨度/配置/tokenizer/实现校验；批量每快照复用 |
| 实验 | 多 owner 协议冻结、预检、逐题落盘、回答复用标记、失败调用账、独立请求计时 |
| 评测 | 在线输入与标签隔离、盲评、来源召回、阶段失败定位、配对统计与可选 owner bootstrap |
| 可复现性 | 完整候选相关性、逐步选择与备选动作、截断记录、精确调度回放、代码/环境/输入 hash |

P5 的受控定义见 §5，不能称为原生 M0。tag 文本向量、逐项表示消融、全梯子、多编码器、SwiftMem/HyperMem 的完整复现不属于本轮必要实现，按设计在初步信号后或正式相关工作对照阶段开展。

## 2. 安装与最小离线运行

```bash
python -m pip install -e '.[test,tokens,pilot]'
python -m pytest -q
python -m eitr canonicalize --snapshot examples/eitr/raw_snapshot.json --out artifacts/demo/snapshot.json
python -m eitr build --snapshot artifacts/demo/snapshot.json --config configs/eitr_v2.json --out artifacts/demo/index.json
python -m eitr prepare-corpus --index artifacts/demo/index.json --tokenizer characters --out artifacts/demo/corpus.json
python -m eitr compare --index artifacts/demo/index.json --corpus artifacts/demo/corpus.json --question 'Which herbs did I use for halloumi salad?' --tokenizer characters --out artifacts/demo/comparison.json
python -m eitr batch --index artifacts/demo/index.json --corpus artifacts/demo/corpus.json --questions examples/eitr/questions.json --tokenizer characters --out artifacts/demo/batch
```

输出路径必须不存在。`characters` 只供工程测试；CLI 禁止用它请求真实模型。生产使用与回答模型匹配的 tokenizer JSON，并固定 SHA256，或明确选择已验证兼容的 tiktoken encoding。

`build`、`prepare-corpus` 会另写 `.metrics.json`，包含输入加载、构建及序列化的 wall/CPU 和进程 RSS 高水位。内存值不是每题增量峰值。既有事实 embedding 的历史生成费用不自动视为免费。

## 3. 数据来源、日期与原生辅助图

```bash
python -m eitr import-mragent --pack /path/conv.json --manifest /path/scope.json --out artifacts/owner/snapshot.json
```

scope manifest 示例：

```json
{
  "owner_id": "conv-X",
  "snapshot_id": "frozen-snapshot",
  "allowed_source_ids": ["D1:1", "D1:2"],
  "extraction_prefix_verified": true,
  "keyword_cache_is_pre_injection": true,
  "session_datetimes": {"D1": "2024-09-15"},
  "attestation": "Verified source horizon and date provenance"
}
```

时间映射使用 raw session ID（D1），不能使用 rewrite session_1。没有核验的时间为 null，不从重写事实推测。声明来自数据提供者，程序验证一致性，不能证明虚假声明。

可从原始 LoCoMo 对话核实日期：

```bash
python -m eitr attest-dates --snapshot artifacts/owner/snapshot.json --conversation /path/original_conversation.json --out artifacts/owner/dated-snapshot.json
```

输入包含 `owner_id`、`conversation` 和可选 `raw_format`。默认精确比较 `speaker + ': ' + text`；`mragent_artifacts_v1` 精确复现现有缓存的 `speaker.strip() + ':' + text.strip()`，以及原生 `blip_caption` 的 ` and shared ...` 后缀。所有原文必须匹配，才附加真实 session date。日期改变快照 hash，之后重建两个索引及原文缓存。

原生 keyword 边与 injected_speaker 边区分；后者不进入统计特征。导入器恢复原生构图时单来源事实的 speaker 导航边；多来源或缺 speaker 的情况记录为无法按该规则核验，不猜测关联。native import 同时保留原生 topic→fact 与 person/aspect→origins，供 P5 的七种工具使用。包含范围外事实的 topic 不保留；无合法事实来源的空 topic 不作导航候选。person/aspect 读取原文，不把重写个人总结作为答案证据。原生辅助关系不参与 R_cue 向量。

旧 EICR records 可能已丢失 edge_origin，不能改 schema 后冒充原生输入；应重新导入原始缓存。

## 4. 主算法和对照

来源组是排序后的完整 origin ID 集合，不是独立事件。事实 cue 均分并聚合为 H；对有效 tag 数 M，`d_c=log((M+1)/(df(c)+1))`，`v_t=normalize(sqrt(H_t)*sqrt(d))`。同 cue 的关联支持是各入口组已发现路径中的最大值，再按组等权平均。别名合并后只占一个支持委员、一个覆盖维度；覆盖查询使用该组在问题中实际出现的表面形式并集。

种子只读实际映射 `(key,tag)` 事实并集，再次到达也不扩围；非种子可读其原生 tag 事实。core＋固定同会话前后各至多一个 atom 构成完整动作，不能在达到候选上限时截掉上下文。

```text
C_j = max_{e in selected} r_j(e)
delta_cov = mean_j max(0, max_{e in new} r_j(e) - C_j)
rel_density = sum_{e in new} text_tokens(e) * r_q(e) / max(1, actual_delta_tokens)
cov_density = 128 * delta_cov / max(1, actual_delta_tokens)
priority = [epsilon + (1-epsilon)*support] * [eta*rel_density+(1-eta)*cov_density]
```

`actual_delta_tokens` 是完整包重渲染后的增量，包含真实日期、speaker、来源头和分隔符。每次提交后重算覆盖、去重和成本。收益密度不再除第二次成本。覆盖满分不表示答案充分，也不停止后续相关证据读取。

| arm | 行为 |
|---|---|
| P0 | 原问题 flat BM25＋去重＋成本贪心，无提取模型 |
| P1 | direct＋精确种子范围，无关联图扩展 |
| P2 | 同图候选，tag 均分配额后按固定 tag 顺序回收 |
| P3 | 同候选池，整题相关性/真实成本贪心 |
| P4 | 完整支持＋覆盖委员会 |
| P4_no_support | Q=1，保留覆盖反馈 |
| P4_no_coverage | eta=1，保留支持 |
| strict_seed | 仅种子诊断，无 direct 回退 |
| P5 | 真实 MRAgent 工具循环，共用原文和回答协议 |

无映射时 P1–P4 使用 P0 的直接候选与完整预算。P2/P3/P4 及组件消融的 pool_hash 必须相同。前端网络调用的时间、用量不影响候选池身份。P0 不承担前端费用；P3 不承担不使用的 cue 查询。

R_fact 输入为 `{snapshot_hash, encoder, fact_ids, vectors}`，顺序必须与 canonical facts 对齐。可先用 `align-fact-vectors --index ... --vectors export.json --out aligned.json` 对齐，export 是 `{encoder,fact_ids,vectors}`，ID 集合必须完全一致；随后 `build --fact-vectors aligned.json`。不自动调用 embedding API，不把 fact 向量冒充原文 atom 向量。

## 5. 正式 P5 如何接入

```bash
python -m eitr pin-mragent --root /path/MRAgent --out artifacts/native-pin.json
python -m eitr compare --index artifacts/owner/index.json --corpus artifacts/owner/corpus.json --question 'Your question' --arms P0 P1 P2 P3 P4 P5 --tokenizer json --tokenizer-file /path/tokenizer.json --tokenizer-sha256 VERIFIED_SHA256 --native-root /path/MRAgent --native-manifest artifacts/native-pin.json --llm-frontend --api-base https://provider.example/v1 --model PINNED_MODEL --key-env YOUR_API_KEY --out artifacts/owner/comparison.json
```

上例会真实调用前端和探索模型；没有 `--generate` 时不调用最终回答器。正式批量回答使用 `batch --generate` 或冻结协议 `experiment`。

实现从用户的可信本地 checkout 校验并加载真实 `LLM.chat_with_tools_once`、`ToolBridge`、TOOLS、Prompts 和 JSON parser；不重写多轮控制函数，不导入会自动初始化凭证/客户端的整套应用，也不分发上游源码。校验基于文件内容，不能把含本地改动的 checkout 声称为官方干净 release。

七种原生工具均有适配：key/tag 读取、会话时间、事件关键词、事件上下文、个人信息方面、个人方面读取、topic 事件读取。网络使用原生 function tool-calling 消息，保留每轮上下文、工具参数和响应。工具内容受同一原文、邻接上下文、候选/动作/证据预算限制。

按照受控协议替换的部分明确写入每次 manifest：共用入口映射与廉价 tag/topic 初筛；原生工具的事实文本变为可追溯原文；共用预算；原生循环决定何时停止，原生生成的答案不进入最后回答器，但其模型用量仍计探索费用；最终调用共用回答器。默认 generic native category=0，不把评测分类标签传入模型，不执行原生额外 dense/LLM 初筛。**这定义 P5；原生全系统 M0 是另一项可选系统参考。**

旧简化循环保留为 `explore-proxy`/`P5_proxy`，不在实验 RUN_ARMS 中，不可用于替代 P5。

## 6. 冻结协议和运行

所有路径使用绝对路径。spec 必须显式包含以下字段：

```json
{
  "config": {},
  "arms": ["P0", "P1", "P2", "P3", "P4", "P5"],
  "tokenizer": {"kind": "json", "path": "/path/tokenizer.json", "sha256": "VERIFIED_SHA256"},
  "assets": [{"index": "/path/index.json", "corpus": "/path/corpus.json"}],
  "questions": "/path/questions.json",
  "native": {"root": "/path/MRAgent", "manifest": "/path/native-pin.json"},
  "model": {"api_base": "https://provider.example/v1", "model": "PINNED_MODEL", "key_env": "YOUR_API_KEY", "max_output_tokens": 2048},
  "frontend": "llm",
  "generate": true,
  "require_dates": true,
  "prices": null
}
```

`config:{}` 使用完整默认值；建议保存 configs/eitr_v2.json 的完整内容。prices 可为 `input_per_million`/`output_per_million` 的实际美元单价。纯检索工程测试可设 supplied、generate=false、require_dates=false，但不能把缺日期的测试称为完整时间问答实验。

questions 只允许 `question_id,owner_id,snapshot_id,question,mentions?`。supplied 模式要求逐题给 mentions（空列表也应显式）；llm 模式不允许附带 mentions。gold/reference/answer 等标签字段在运行前拒绝。

```bash
python -m eitr freeze --spec /path/spec.json --out artifacts/protocol.json
python -m eitr preflight --protocol artifacts/protocol.json
python -m eitr experiment --protocol artifacts/protocol.json --out artifacts/experiment
```

freeze/preflight 不调用模型。代码、数据、配置、依赖环境、tokenizer 或上游文件变化后必须重新冻结。experiment 按 owner/snapshot 分组运行，每个原文索引加载一次。每题检索后即落盘，每个答案完成后更新；失败保存调用账并停止，不能忽略错误拼成成功主表。完成后的题文件有 hash，评分前再次校验。

## 7. 回放、诊断与评分

`retrieve` 输出可直接用于 `replay`；批量结果先取出相应 rows 中单个 run JSON：

```bash
python -m eitr replay --index /path/index.json --corpus /path/corpus.json --config /path/config.json --tokenizer json --tokenizer-file /path/tokenizer.json --tokenizer-sha256 VERIFIED_SHA256 --run /path/run.json --out artifacts/replay.json
python -m eitr grade --batch /path/owner-batch --labels /path/labels.json --index /path/index.json --api-base https://judge.example/v1 --model PINNED_JUDGE --key-env JUDGE_KEY --out artifacts/grades
python -m eitr summarize --rows artifacts/grades/rows.json --bootstrap 2000 --seed 0 --out artifacts/summary.json
```

labels 每题包含 `question_id,owner_id,snapshot_id,references:[...],source_ids:[...]`。评分过程才读取标签；judge 只看到问题、参考与预测，看不到 arm 和成本。同题同预测可复用判分；人工复核样本和分歧仍是正式评测操作，程序不伪造人工结论。

诊断区分无映射/图不可达、图工作上限、每 tag 筛选、direct cap、全局候选 cap、预算/选择丢失，并报告各阶段来源召回。不受在线工作限额的两跳可达 oracle 仅在评分阶段计算。来源出现不等于相关答案跨度完整，因此诊断不会自动宣称“证据充分但模型错误”。

bootstrap 按 owner 重采样，保留 owner 内配对题；仅一个 owner 不生成区间，owner 很少时显式标记不稳定。重复生成不增加独立题数。程序不会自动得出 2pp 非劣效结论。

## 8. 成本与独立计时

compare/batch 为机制比较复用前端与相关性缓存；这些分臂时间不用于端到端 p50/p95。相同证据包可复用回答，但独立部署费用仍计一次回答。

```bash
python -m eitr benchmark --index /path/index.json --corpus /path/corpus.json --questions /path/questions.json --arms P3 P4 --repeats 3 --tokenizer json --tokenizer-file /path/tokenizer.json --tokenizer-sha256 VERIFIED_SHA256 --llm-frontend --generate --api-base https://provider.example/v1 --model PINNED_MODEL --key-env YOUR_API_KEY --out artifacts/benchmark.json
python -m eitr cost-report --runs artifacts/benchmark.json --prices /path/prices.json --offline /path/lifecycle.json --out artifacts/cost.json
```

benchmark 每臂每次独立执行前端/选择/回答，轮换运行顺序；逐运行落盘并保存失败 manifest。原文索引预加载，冷加载另记；supplied 前端和 retrieval-only 模式显式标记，未生成答案的时间不会冒充完整回答延迟。

成本汇总保留阶段调用数、失败数、输入/输出 token、p50/p95、CPU/进程 RSS、上下文大小。缺失 API usage 是 unknown/null，不是零。lifecycle 输入：`baseline_arm,method_arm,added_build_seconds,added_build_usd,queries_per_snapshot`，分别计算时间和货币回本点以及每次快照更新周期的净节省；不节省则没有回本点。

## 9. 验收而非效果结论

原生集成测试：

```bash
EITR_NATIVE_ROOT=/path/MRAgent python -m pytest -q
```

未提供上游 checkout 时，这些特定集成项明确 skip；普通算法/接口测试仍执行。真实缓存工程验收入口是 `scripts/eitr_acceptance.py --help`。它检查缓存 SHA、来源日期、R_cue/R_fact、P0–P5 与精确回放，P5 使用脚本化响应驱动实际原生循环，**不调用 LLM、不产生可用于论文的 QA 成绩**。

代码验收能够减少实现错误造成的混淆，不能保证不存在任何 bug，也不能代替冻结数据上的效果实验。完成首轮协议后优先比较 P3/P4，并保留 P0/P1/P2/P5；不扩展方法边界。
