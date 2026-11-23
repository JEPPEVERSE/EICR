# 原生 tag 选择器 v1：修复后代码验收

已核对的 Notion 方案和原生替换范围不变。2026-11-22 方案覆盖审计发现的7个问题及本轮功能缺口已经修复，详见[修复报告](native_tag_selector_repair_20261122.md)与[运行说明](native_tag_selector_implementation_20261122.md)。原审计保留为修复前历史记录。

修复后全仓回归 **349 passed，0 skipped**；native_tag专项 **41项**，比原24项增加17个执行用例。重点新增：

- native携带calibration时仍保持教师分数和完整轨迹；shadow独立记录。
- 畸形usage、后处理失败、judge失败、embedding失败的attempt持久化与unknown计费。
- Stage A/calibration/shadow身份链、冻结judge/rubric校验、确认轮配置一致和预注册诊断子集。
- 分批复核的全局对称性与原始成绩保留。
- 来源支持盲审、离线收据与重建摊销、完整分析CLI和可读报告。
- 长候选事实细选、事件重排、时间专用路径，以及独立执行未改写原生模块的对照。

包含原有的完整native/透传回放、作用域和向量身份、PAVA、阈值、小列表、全部原生工具、预算，以及32题×两臂×两次的128次独立完整流水线模拟。

sdist/wheel构建成功，CLI入口与diff格式检查通过。最终代码hash、测试和构建证据在 `artifacts/native_tag_selector_v1_20261122/repair/manifest.json`，验收收据在 `repair/checks/checks.json`；原生快照继续使用 `runtime-final/`。旧 `acceptance/checks.json` 只适用于修复前代码。

未调用真实provider，没有真实准确率、服务费用、降本或非劣结论。源码验收通过不替代真实开发集Stage A/B、校准支持、人工盲审或确认轮；新协议会强制检查相关收据。人工来源支持评估未完成时保持unresolved。
