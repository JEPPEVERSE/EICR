# v3 完整实现与验收报告

日期：2026-11-21。对应最新收窄主线：**一次问题需求解析 → 下游事实/tag 矩阵匹配 → 固定预算候选 → 成熟重排器 → 原文回答**。

## 交付结果

已实现新主线的完整执行链路，而非只有数学函数或未接入实验的独立原型。新增 7 个生产模块，接入现有 `eitr` CLI、成本账和评分模块，版本更新为 0.3.0。详细公式、执行规则、限制和命令见 [实现说明](matrix_v3_implementation_20261121.md)。

| 模块 | 当前状态 |
|---|---|
| 来源绑定、合法范围检查与 MRAgent canonical snapshot 复用 | 完成 |
| 真实 OpenAI-compatible embedding 调用、query/document 身份校验、缓存和账单 | 完成；本次未发起真实模型调用 |
| X 事实矩阵、B 稀疏成员矩阵、来源平衡权重 | 完成 |
| tag–tag 平均关联、双向覆盖、最大桥接及 witness | 完成；辅助图候选边采用近似 shortlist |
| 单次 LLM 需求解析、交集/分主体等组合、grounding 与显式 fallback | 完成 |
| 原问＋各需求的 query–fact 与 query–tag 计算及 witness | 完成 |
| 固定轮转候选、直接召回保底、可选有界图扩展 | 完成 |
| 原文 gte-rerank-v2、单请求限制与完整上下文 | 完成，复用已有成熟接口 |
| 固定预算选择、精确渲染成本、需求覆盖诊断 | 完成 |
| 组合要求进入回答提示、原文引用校验、失败原始响应保存 | 完成 |
| 冻结/预检、多 owner 批处理、回放、strict/lenient 评分、汇总 | 完成 |
| 生成失败保留分母、聊天/embedding/rerank 分类成本 | 完成 |
| 可运行模板、合成 smoke、数学与端到端反例测试 | 完成 |

未列入当前方案的训练式 attention、分层索引/剪枝、额外 judge 和新委员会没有加入。历史 P*/C* 保留复现，当前主线使用明确的 `matrix-*` 命令，不会隐式调用旧的委员会。

## 验证记录

1. 新增 `tests/eitr/test_matrix.py`：**25 passed**。覆盖分块关系与密集参考计算一致、方向不对称、来源重复、多模态 witness、索引读写、零/非法向量、缓存顺序/owner/encoder 错配、交集与分别、前端失败降级、四臂回放、实际预算、上下文不可分割、无 tag 直接召回、长来源末尾证据、真实 HTTP 接口响应顺序校验、协议冻结、离线批处理、隔离评分、无答案失败计错等。
2. 全仓测试：**271 passed, 6 skipped**。6 项跳过项全部是未设置路径的旧 MRAgent 集成检查。随后定位本地上游 checkout，设置 `EITR_NATIVE_ROOT`，针对相关两份测试文件运行：**16 passed**，覆盖全部 6 项原跳过项。合计 277 个不同测试均通过，没有遗留未验证的跳过项。
3. `python -m build --no-isolation`：成功构建 `eicr_memory-0.3.0.tar.gz` 和 wheel；检查 wheel 确认包含全部 7 个矩阵模块。项目虚拟环境补装缺少的 setuptools/wheel 后构建通过。
4. `pip check`：无损坏依赖。CLI 帮助与新命令可加载，`git diff --check` 通过。
5. 合成 smoke：四个 arm 均完成候选、装包和回放；没有模型 API 调用。使用的 synthetic embedding/reranker 和字符计数都明确标注，不作为有效性证据。

## 真实资产兼容检查

对已有 `eitr_round2_20261119` 两份资产进行 canonical snapshot 校验和当前配置的 Corpus.load：

| 资产 | 事实 | tag | 原文来源/atoms | 最大 tag 事实数 | 无事实来源 | 复用原文 corpus |
|---|---:|---:|---:|---:|---:|---|
| conv-41 | 1,844 | 320 | 663 | 107 | 0 | 通过 |
| conv-42 | 1,354 | 281 | 629 | 53 | 0 | 通过 |

共 3,198 条事实、601 个 tag。它们无需重新抽取即可进入新建图流程。新矩阵 embedding 必须使用已声明模型身份重新准备；没有把旧的稀疏 cue 表示或身份不明的缓存冒充新向量。

本地审计输出：`artifacts/matrix_v3_20261121/asset_audit.json`；合成输出：`artifacts/matrix_v3_20261121/synthetic_smoke/`。这些运行产物受 gitignore 管理，不上传原始记忆内容。

## 实现复核中修正的关键问题

- 直接检索控制臂跳过 tag 计算，使用完整候选额度，避免用过弱 baseline 衬托矩阵方法。
- 每个需求按自己的事实相似度构建队列，不因为另一需求相关就混入本需求的零相关事实。
- source-level provenance 映射到全部原文 atoms；不假装拥有事实级字符对齐，也不只取长来源开头。
- 候选上限按上下文闭包检查，不能截断一个已接受动作的上下文。
- 回放重新计算矩阵得分、witness、路由与选择，不仅比较日志里的 hash。
- 前端缓存身份包括模型、endpoint、生成参数和 token 上限；查询向量还绑定问题、计划和索引。
- 完全无答案的生成失败按失败计错；有原始文本但引用合同失败时可单独盲评其语义，合同有效性仍为 false。
- 成本账按 billing kind 拆开，拒绝用聊天价格套 embedding/rerank token；离线构建成本单独记录。

## 结论与下一次实验条件

**当前收窄方案在代码、CLI、测试和实验接口层面已闭环。** 本轮未进行真实 embedding、LLM 或 reranker 效果实验，不能宣称准确率提升、tag 图有效或成本下降。

下一次真实小规模验证可以直接使用模板冻结：`direct_rerank / matrix_rerank / matrix_graph_rerank`，必要时加 `matrix_embedding`。应先记录候选召回、证据包召回、最终质量和分阶段成本；最大事实匹配的 tag 大小偏差、自由改写误差、图 shortlist 的漏边和全事实扫描成本仍是需要实测的研究风险，不是已经由测试证明解决的问题。
