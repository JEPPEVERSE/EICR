# 原生与答案修正版：开发集重复验证

日期：2026-11-23。沿用相同的 16 道开发题、32 个原生最终上下文，比较原生提示与 `evidence_complete_v2`。本轮重新调用服务商回答并评分，没有复用上轮答案。

本轮自动严格正确率：原生 **26/32**，修正版 **24/32**，差值 **-6.250 个百分点**。输入 token 增加 **18.76%**。该结果属于固定上下文开发诊断，不能视为端到端提升或独立确认。

## 同等标准与结果

代码、模型、两种提示版本、问题、上下文、随机执行顺序、每次 1,024 输出 token 上限和 strict/lenient 标准均与上一轮 v2 验证相同。模型温度为 0。两臂配对随机交错且串行；评分隐藏方法名，同轮精确相同文本复用结果。每题的两份既有上下文分别重放，未重新运行检索、tag 选择或完整 agent。

| 指标 | 上轮原生 | 上轮修正版 v2 | 本轮原生 | 本轮修正版 v2 |
|---|---:|---:|---:|---:|
| 自动 strict | 24/32 | 25/32 | 26/32 | 24/32 |
| 自动 lenient | 24/32 | 25/32 | 26/32 | 24/32 |
| 合法引用 ID | 32/32 | 32/32 | 31/32 | 31/32 |
| 调用／回答格式失败 | 0 | 0 | 0 | 1 |
| strict 待复核运行 | 8 | 8 | 8 | 8 |
| 输入 token | 92,616 | 109,992 | 92,616 | 109,992 |
| 输出 token | 2,299 | 2,240 | 2,389 | 2,211 |

按 32 个相同上下文比较，修正版胜／平／负为 **1/28/3**；先在每题内平均后，16 道题的胜／平／负为 **1/13/2**。重复上下文与跨轮重放不是新的独立题目。

修正版在 `conv-43-qa020` 的一次响应虽然包含 Seattle，但缺少原生合同要求的 `mode: "answer"`，因此按冻结规则计为格式失败。原生提示在 `conv-41-qa014` 的一次响应把 ID 与正文混在 supports 中，引用不合法；语义评分独立进行。两种问题都保留在分母和账本中，没有补跑替换。“合法引用 ID”的分母包含无有效答案的请求，所以修正版少一条合法记录是这次格式失败所致。

## 逐题严格分数

| 问题 ID | 原生（两份上下文平均） | 修正版（两份上下文平均） | 修正版结果 |
|---|---:|---:|---|
| conv-26-qa052 | 1.00 | 1.00 | tie |
| conv-26-qa064 | 1.00 | 1.00 | tie |
| conv-26-qa073 | 1.00 | 1.00 | tie |
| conv-26-qa101 | 1.00 | 1.00 | tie |
| conv-41-qa014 | 1.00 | 1.00 | tie |
| conv-41-qa029 | 1.00 | 1.00 | tie |
| conv-41-qa058 | 0.50 | 1.00 | win |
| conv-41-qa140 | 1.00 | 1.00 | tie |
| conv-42-qa072 | 0.50 | 0.50 | tie |
| conv-42-qa078 | 0.00 | 0.00 | tie |
| conv-42-qa084 | 1.00 | 0.50 | loss |
| conv-42-qa127 | 1.00 | 1.00 | tie |
| conv-43-qa011 | 1.00 | 1.00 | tie |
| conv-43-qa020 | 1.00 | 0.00 | loss |
| conv-43-qa051 | 0.00 | 0.00 | tie |
| conv-43-qa149 | 1.00 | 1.00 | tie |

## 稳定性、成本与判断

与上轮 v2 验证相比，原生 2/32 个答案文本变化、2/32 个 strict 分数变化；修正版分别为 6/32 和 1/32。温度为 0 并不保证服务端输出逐次相同。

跨轮共有 48 个精确相同评分缓存键，其中 0 个发生自动判分不一致；明细保留在 repeat-comparison.json。该检查区分生成变化与判分波动，没有事后覆盖任何成绩。人工裁定仍未完成。

本轮修正版没有复现上轮相对原生的一次得分优势，反而少判对两次。主要差异如下：

- `conv-43-qa020`（Chicago 之前的城市）：原生两次回答 Seattle；修正版一次认为证据不足，一次虽输出 Seattle 却缺 mode 字段，两次都按原规则计错。
- `conv-42-qa084`：一个题次原生回答 No、修正版回答 Yes，自动评分只接受原生；该分歧在前两轮也出现过，需结合推断口径复核。
- `conv-41-qa058`：一个题次修正版日期答案获判对、原生未获判对；时间精度仍是既有复核事项，不能据此认定日期能力已稳定提升。

相同文本的跨轮评分在本次检查中一致，说明这些已观察变化更指向生成和输出合同的稳定性；这不排除不同措辞的自动判分仍有噪声。下一步应优先处理漏 mode 的结构化输出，以及归纳规则是否导致有证据却过度拒答。此处仅定位问题，本轮未修改代码或再次调提示。

本轮新增 64 次回答和 54 次自动评分，合计 118 次服务商调用。逐次 usage 完整性与账本已保存；两组答案输入／输出 token 和 API 等待时间在分析 JSON 中分开记录。未重新核实当天实际费率，因此实际货币费用保持未知；历史周日价重计不是当天账单。这里的成本仅为回答阶段及评分投入，不含重新检索，不能推算完整请求降本比例。

**当前保持原生默认。** 判读重点是配对结果、跨轮稳定性、引用合法率与新增输入成本；开发题和自动评分的局限没有因重复运行消失。本轮没有改提示、调阈值、挑选有利题次，也没有启动新题确认轮。

## 可复核产物

- [本轮自动分析](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/evaluation/replay-analysis.json)
- [跨轮及逐题比较](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/repeat-comparison.json)
- [冻结重放计划](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/plan.json)
- [全部回答与调用账本清单](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/runs/manifest.json)
- [待人工复核答案](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/evaluation/replay-grades/adjudication/blind.json)
- [运行前记录与历史文件指纹](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_development_repeat_20261123/preregistration.json)
- [上一轮修复与优化报告](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/docs/native_tag_optimization_20261123.md)
