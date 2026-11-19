# 显式评估口径与引用有效性

2026-11-19。本次核对的是本地 MRAgent checkout 的评估代码和其 git HEAD `7441506db984b7c4da32e8dbeb2527f2e351270a`，不是根据论文文字推测的评分标准。

`eval/judge.py` 的 `ACCURACY_PROMPT` 原文包含：

> The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

日期题另要求指向相同日期或时间段，可以使用不同格式。`eval/evaluate_reasoning.py` 对非 category 5 问题直接调用该 judge；category 5 在该上游程序中跳过 LLM judge，采用独立字符串匹配。上游还计算 F1。因此只复制 judge rubric 不能声称复现上游完整评分程序，更不能保证复现论文数字。

本地两个文件有修改，主要涉及 provider wiring/defaults。经过 AST 提取，`ACCURACY_PROMPT` 与该 HEAD 版本逐字一致。`judge_profile_metadata('mragent_lenient')` 记录上游仓库、commit、两个原始文件 SHA256、两个本地审计 SHA256、rubric 的 SHA256 与本适配器 system prompt SHA256。评分 manifest 与每行 profile hash 固定评分口径。

## 接口

```python
from eitr.evaluate import grade_batch, judge_profile_metadata

# 原来的严格 semantic rubric 默认不变。
strict = grade_batch(batch, labels, index, backend, new_output_dir,
                     judge_profile='semantic_strict')

# 单独新目录产生宽松口径结果；每题要求单一 reference，匹配上游格式。
lenient = grade_batch(batch, labels, index, backend, another_new_output_dir,
                      judge_profile='mragent_lenient')
```

两种口径都用相同 `backend.chat_json(system, user)` 合同。严格口径继续发送原有 system prompt 和 `{question, references, candidate}` JSON。宽松口径将完整上游模板格式化后放入 user message，新增 system message 只限定任务数据不构成指令、要求盲评和 JSON。宽松输出解析 `label` 的 `CORRECT` / `WRONG`，不假装模型给出了上游模板并未强制返回的推理。

所有臂使用同一个 profile。同题、同文本、同 profile 的语义判分可以复用，引用合法性逐个 arm 独立审计。不同 profile 或不同 profile hash 的结果禁止混合汇总。已有输出目录拒绝覆盖，历史数据不自动重评。

## 分开的结果字段

- `correct` 和 `judge_correct`：答案文本在所选 rubric 下的判定。
- `citations_valid`：给出的引用是否形状正确、属于合法原文、落在证据包可见跨度内。空引用集合本身不违反跨度规则。
- `answer_contract_valid`：在引用检查之外，还检查结构、resolution status、unsupported premises、supported 答案的非空内容和引用要求。
- `valid_and_correct`：同时通过文本判分和回答合同。summary 独立给出该准确率及合同有效题数。
- `citation_errors` 与 `answer_contract_errors`：具体失败原因。

即使宽松 judge 判 `CORRECT`，越界引用也仍然是 invalid。引用范围合法仅说明文本可追溯，不证明引用内容蕴含答案。没有 `answer_text` 的原始失败调用无法用此 grader 判语义正确性，应继续在失败账中独立计数。
