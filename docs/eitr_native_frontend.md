# MRAgent 风格前端适配与静态标签重用

本模块用于隔离“入口抽取/映射是否造成损失”这一因素。它把问题转成关键词及检索变体，映射到已有 key，再由整题条件下的标签评分选择已有 tag；不重新抽取离线图，不制造新 tag，不将动态检索循环视为已经复现。

## 来源与适配范围

明确借鉴 [MRAgent](https://github.com/Ji-shuo/MRAgent)，本地核对版本为 `7441506db984b7c4da32e8dbeb2527f2e351270a`：

- `prompts/prompts.py`：`QUESTION_KEY_SYSTEM_PROMPT` 的关键词、alternatives、question_time 输出思路。
- `memory/controller.py`：`match_query_key_to_graph_keys` 的归一化、精确相等、词集合包含、Jaccard 匹配规则。
- `agent/agent.py`：`select_key_tag` 的问题条件化标签相关性评分。
- `prompts/prompts.py`：离线 `REWRITE_SYSTEM_PROMPT` 生成的标签已经包含在当前导入资产内；本次只重用这些标签。

实现增加严格 JSON 合同、合法 ID 校验、确定性候选上限、批量评分、错误追踪。它不是上游前端逐行复制：关键词数量有限，标签评分合并为一次调用，上游只在特定条件下评分而本模块对所有已提供标签评分；也未复现上游全部时间处理及后续 agent 动作。实验需标记为 `native_style_v1` / MRAgent 风格适配，不可称为“原生 MRAgent 结果”。

## 执行路径

1. `static_tag_catalog(index)` 从冻结 snapshot 读取原有 cue、tag、fact 和三者的合法关系。保留原 ID；只排除被显式标为 `scope_role="scope"` 的 key。人物导航边仍沿用导入图，未作为新增 cue 向量特征。
2. 一次 LLM 提取：`{"question_time":"", "keywords":[{"id":"...", "alternatives":["..."]}]}`。最多 `Config.max_query_cues` 个关键词。若模型超限或结构错误，明确失败，不截断后假装成功。
3. 按原词精确匹配、变体精确匹配、词集合包含、Jaccard ≥ 0.6 的顺序匹配已有 key。也允许图中已有且有来源的显式 alias。默认每关键词最多 3 个 key，省略项记录于 `omitted_key_ids`。
4. 收集这些 key 已连接的 tag。整题 BM25 只用于有界候选呈现顺序；默认每组最多 64 个、整题最多 64 个 key-group/tag 对，组间轮转分配机会。这里仍有候选截断风险，日志保留被截断 tag。
5. 一次 LLM 对全部已提供 group/tag 打 `[0,1]` 分。要求回复恰好覆盖提供的 ID，拒绝新增 ID、缺失 ID、重复组、布尔数值、非有限数值和越界分数。分数表示检索相关性，不表示事实为真或“同一事件”的概率。
6. 按评分降序、tag ID 稳定打破平局，丢弃零分 tag。之后才应用每 key 的输出 tag 上限。`facts_by_tag` 只保留最终获准的 key/tag 对对应的事实，避免通过已满额 key 混入其他事实。全局 seed 上限采用组间轮转；保存截断记录。

有命中的常规请求最多两次模型调用；完全无可用 key/tag 时只有提取调用。`question_time` 仅保存在元数据中，`time_used_for_filtering=False`，不根据模型推断日期过滤记忆。

## API

```python
from eitr.native_frontend import map_native_groups, NativeFrontendOptions

query = map_native_groups(index, question, config, backend=backend)
# backend 必须提供 chat_json(system, user) 以及 calls 记录。
# query 可直接作为 retrieve(..., query=query) 的共享查询输入。
```

在线与手工/缓存响应不可混用。离线检查可提供两段原始响应：

```python
query = map_native_groups(
    index, question, config,
    keyword_response=recorded_keywords,
    tag_response=recorded_tag_scores,
)
```

离线调用不会联网；没有候选时不要提供 `tag_response`。`NativeFrontendOptions` 可控制变体数量、候选数量、Jaccard 阈值和提示长度。提示长度上限是字符数，不能当作 token 费用；费用仍以 backend 的真实 usage 为准。

返回 `eitr-query-1`，兼容现有 `groups`、`seed_tags`、`facts_by_tag`、`mapping_config_hash` 合同，并增加 `frontend_request` 中的模式、候选对数、提示/目录 hash 与来源。关键词不要求逐字出现在问题中，因此不伪造 literal mention span。不同抽取关键词暂不强制合并成同一身份；其 alternatives 也不是经证实的实体别名。

失败抛出 `NativeFrontendError`，携带 `stage`、`frontend_calls`、`frontend_outputs`。本模块不重试、不悄悄转成无入口检索。成功但没有 seed 时返回 `status="no_mapped_seeds"`；下游若使用 flat fallback，需按下游原有运行模式单独记录。后续比较必须把这两次前端调用、失败调用的费用纳入独立部署成本。

## 检查

`tests/eitr/test_native_frontend.py` 的 20 项测试通过，覆盖合法映射与证据来源、上限、恶意/缺失输出、调用追踪、无静默降级、静态资产不变、离线重放与 `retrieve` 集成。测试使用 fake backend，不能证明真实 LLM 的映射准确率；需在相同问题、冻结图、相同后端和预算上实测前端候选/最终证据召回变化。
