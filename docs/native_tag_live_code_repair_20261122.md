# 真实诊断暴露问题的代码修复

对应 [2026-11-22 真实诊断](native_tag_diagnostic_report_20261122.md)。本次任务修复代码并做离线验证，没有发起新的 DeepSeek 请求，也没有将历史失败重新标记为验收通过。

## 修复范围

| 问题 | 原因与修复 | 验证 |
|---|---|---|
| 影子统计遇到 `(Maria, Plan)` 崩溃 | 原生模型可能返回候选之外的 tag；原生读取器对此返回空边，而统计代码曾直接索引字典。统计层现在采用空边语义，保留原响应/选择，并记录 `native_unlinked_selected_tags`。合法候选和 scope 预检仍然严格。 | 初始、扩展阶段均注入未知 tag；修复后原生与影子的工具结果、状态和答案相同，不完整教师响应仍不进入校准。 |
| 真实响应回放引入额外空字段 | 离线 Message 曾忽略 SDK 的 `exclude_none=True`，将 null 字段带入下一轮请求。现在对消息及嵌套工具调用按该选项序列化，并保持原始保存响应不变。 | 带 `content/refusal/index=null` 的真实 SDK 形状测试；5 个真实响应案例的离线差分验证。 |
| 不同进程的原生并列项顺序变化 | 原生代码以 set 聚合候选，排序键相同的元素保留集合遍历顺序；Python 哈希随机化会改变顺序。CLI 在解释器启动时固定默认 `PYTHONHASHSEED=0`，保留显式固定种子；协议同时记录配置值和实际 hash 探针。 | 两个独立进程冻结出的协议完全相同，两个独立进程执行的完整原生轨迹相同；改成种子 1 时，preflight 在执行前拒绝。 |
| 中止原因反复嵌套，批次 journal 停在 running | 只记录首次适配错误，后续未执行行引用同一原因并保持零调用；批次结束写入 complete/aborted 的 journal；适配中止时 `run` 返回非零退出码。 | 3 行批次只执行第 1 行，首错不变，其余行保留在分母；成功和中止两种终态均测试。 |
| 工作流异常后仍显示运行中 | 原临时脚本缺少终止状态捕获。新增可复用的 `native-tag workflow --directory ...`，在子命令异常、适配中止、超时或可捕获的键盘中断时记录失败阶段、命令与异常，并停止依赖阶段。 | 模拟 run 失败、适配中止、shadow-check 失败、键盘中断和等待超时；均未冻结主实验，已有账本保持不变；重复启动同一目录被拒绝。 |

前两项在诊断收尾时已有修复，本次重新验证并补齐了后三项。原生检索的集合构造、排序键、图边、tag 阈值、工具和回答流程均未修改；对跨进程顺序的控制属于两臂共同的执行环境。没有用排序后的新检索策略替换原生行为。

## 验证结果

- 全仓测试首次运行：**355 passed，6 skipped**。6 个跳过项均因未设置旧 EITR 的 `EITR_NATIVE_ROOT`。
- 指定本机 MRAgent 源码后，上游测试组 **8 passed**，覆盖上述全部 6 个跳过项，另含 2 个已通过的同文件测试。合计 **361 个不同测试均已通过**。
- 当前源码的 native-tag 机器验收：**53/53 测试通过、12/12 类检查通过，无跳过**。新增进程身份、异常终止和回放测试纳入验收证据表。
- 将已保存的 5 个真实响应案例作为离线 fixture，在同一进程内生成原生参考，再做严格影子请求回放：**5/5 一致**，包括新增 `Plan` 的历史失败案例，新增 provider 调用为 **0**。
- CLI 帮助和 diff 格式检查通过。原生冻结快照、原始开发/影子账本及其 manifest 仍可按原 hash 验证。

跨进程固定种子的测试使用当前实现重新生成的离线协议。历史付费进程未固定哈希种子，无法追溯补齐其真实随机状态；真实响应 fixture 的新输入不冒充历史付费输入。旧协议应保留为历史证据，不能补字段、重新签名后当成新一轮验收。

## 使用与产物

推荐从项目目录使用 `PYTHONPATH=src .venv/bin/python -m native_tag.cli ...`；重新安装项目后，`native-tag` console entrypoint 也使用新的进程启动路径。直接调用自定义准备脚本时，应在启动前设置 `PYTHONHASHSEED=0`。完整说明见[实现与运行说明](native_tag_selector_implementation_20261122.md)。

真实诊断目录中的旧 `continue.py` 已归档为 `pre-workflow-repair/continue.py`，当前入口转发到包内工作流。旧目录有终止状态及付费产物，启动会被拒绝。下一轮应使用新目录并重新冻结实现、环境、开发及影子证据。

修复与验收产物位于 [native_tag_live_code_repair_20261122](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122)：

- [机器验收收据](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122/checks/checks.json)
- [全仓测试日志](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122/full-tests.log)
- [补充上游测试日志](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122/upstream-tests.log)
- [真实响应离线验证](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122/recorded-response-replay.json)
- [最终修复清单及源码身份](/Users/chenshu/Desktop/papers/main_lab/eicr_v1/artifacts/native_tag_live_code_repair_20261122/manifest.json)

这些证据说明已修复本轮复现的执行问题，不代表廉价选择器的真实质量/成本门槛已通过。
