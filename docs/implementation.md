# 实现说明

实现版本：0.1.0，2026-11-18。详细规格保留为历史设计；此文说明实际运行合同。

## 索引与接口

主要 Python 接口：

```python
from eicr import IndexConfig, build_index, save_index, load_index
from eicr.pipeline import run
from eicr.packer import CharacterTokenizer

index = build_index(snapshot, IndexConfig())
save_index(index, "artifacts/new-index")
index = load_index("artifacts/new-index")
result = run(index, question, CharacterTokenizer())  # 仅示例字符计数
```

`Index.similarity(a,b)` 返回 direct/expanded/combined 三个分数，零向量均为 null；`coverage`、`joint`、`facet_profile` 与 `explain` 分别提供方向覆盖、集合共同支持、tag 质量分布和证据坐标贡献。expanded 贡献始终标为派生特征，不是独立见证。

索引包含 `manifest.json`、标准化 `records.json`、`neighbors.json`、`build_report.json`、R/X/Y/P/T 稀疏 NPZ 和 `weights.npy`。记录内保存原文、来源根、来源族、事实原 ID、直接边与元数据。文件布局合并了设计中的多张 JSONL 表，但保留其审计字段。每次加载验证文件哈希，并由冻结记录重算矩阵、权重和报告；不信任不匹配的缓存。此严格加载会有重建成本，暂不提供跳过审计的快速模式。

图和元数据只属于一个 owner/snapshot；不实现增量追加。配置 `weight_mode="uniform"` 提供等权直接证据对照，默认 `source_family`。没有来源事实的原始 source 不占据向量质量，但仍可被原文检索。

分面：当前 importer 使用 fact 上已有 tags，均分多 tag 质量；保留的原生结构是每条 cue–fact 关联与其 fact tags。自定义每条三元边分配权重尚未提供输入接口；不要声称已支持任意 cue 特定的 facet 权重。topic 与 tag 分离，tag 不进入默认邻域。

结构候选有确定上限。P/T 与保留 g/q/r、边型、topic/link 来源均保存；阈值以下候选记录聚合剪枝数量，top-k 未入选但达阈值的边保留明细。不会为每条弱到零的候选写出巨量日志。

## 前端与策略

`map_query` 接受短语跨度或运行字面扫描；它只映射现有节点。字面扫描会漏掉改写表达和关系词，映射率不是语义完备率。LLM 前端由 `pipeline.run(..., frontend_backend=...)` 显式启用；格式和 offset 无法通过两次检查时回到原问。

`diagnose` 输出小规模 pairwise、直接来源贡献、coverage/J、相同行分组与 Delta。owner/scope、零表示、覆盖全库的零信息节点不参与。相同行分组不是同义词合并，不改变离线图。

`freeze_policy` 检查阈值范围及选择记录，再绑定确切 index hash。它是冻结工具，不是开发集网格搜索器。未提供标签、分组数据和调参目标时，自动选择一套阈值会违背设计，因此未启用。概率校准、自动替换 cue 和多跳取证均未实现。

## Dense bundle

Hybrid 接口接受已有模型的向量，不负责向外部服务批量编码。先用**相同 tokenizer、384 上限、32 重叠的冻结分块**导出顺序，按该顺序编码 chunk，并对当前原问题编码：

```python
import hashlib
import numpy as np
from eicr.contracts import digest, write_json
from eicr.packer import chunk_sources

chunks = chunk_sources(index.records["sources"], tokenizer)
# vectors = encoder([c["text"] for c in chunks])
# qvector = encoder([question])[0]
np.savez_compressed("bundle.npz", chunk_vectors=vectors, query_vector=qvector)
write_json("bundle.json", {
    "npz_sha256": hashlib.sha256(open("bundle.npz", "rb").read()).hexdigest(),
    "index_hash": index.index_hash,
    "tokenizer": tokenizer.identity,
    "chunk_order_hash": digest([c["span_id"] for c in chunks]),
    "question_hash": digest(question),
    "dense_identity": "your-frozen-model-and-preprocessing-identity"
})
```

CLI 验证 NPZ 哈希；pipeline 验证索引、tokenizer、顺序、问题与编码器身份。向量非有限、零向量或维度不匹配均报错。固定 RRF k=60。没有 dense bundle 时只运行 BM25，并如实记录 backend。

## 预算、回答与费用

分块按原文字符 offset 保存，实际大小由指定计数器检查。优先句界，过长句可切成 token 限制内的文本跨度。候选上限256，按队列合同排序，独立队列按来源族轮转。

装包器只对选中区间做增量合并。预算按来源标签、文本和间隔的最终整体渲染计数，≤配额之和。重叠合并有时会减少标签/BPE token，账本保留负增量；每队列 `charged_positive_increments` 不是最终总 token 的可加分解，最终 `used_tokens` 才是权威总量。问题、系统提示和模型输出另记。每条入选跨度保存全部队列归因，未入包记录原因。

OpenAI-compatible backend 接受显式 HTTPS endpoint（本地 localhost 可用 HTTP）、model、凭证环境变量、timeout 和输出上限。默认输出上限2048。无隐式网络调用；此接口只执行前端与最终回答，不自动调用 embedding 或评价服务。调用 usage 由 provider 返回并脱敏保存；若 provider 没有提供 usage，不伪造实际费用。

最终回答引用必须位于实际可见原文跨度内。引用合法只证明引用范围合法，不能证明回答语义正确；需要独立评价。当前回答格式错误即失败，不自动多次重答。

## 评价接口

`eicr audit-splits --rows splits.json` 的每行含 memory_id、question_family_id、split；检查同一记忆和原题族不跨 split。

`eicr evaluate --rows scores.json --out outputs/summary.json` 的每行含 example_id、memory_id、method、condition、correct（外部审定布尔值）。它输出分层准确率与样本量，不做无依据显著性/概率校准。分组 bootstrap、完整实验矩阵、候选公平性复核与自动开发选择留待正式实验实施，不能把当前描述统计当作这些步骤已经完成。
