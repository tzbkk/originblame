# CIKM'26 Camera-Ready 修改清单

**Submission 3519** · OriginBlame · Demo Track · **Accept**
论文源文件：`paper/cikm-2026.tex`（4 页预算，章节在 `sections-cikm/`）
配套全文（可引用转嫁细节）：arXiv:2607.13037，源文件 `paper/originblame.tex`

---

## 评审分数

| Reviewer | Relevance | Overall | 倾向 |
|---|---|---|---|
| R1 | 5 excellent | weak accept | 偏 demo 重心、弱化 novelty |
| R2 | 4 good | weak reject | 存储/性能/不可学 gap |
| R3 | 5 excellent | accept | unlearning gap、author 语义、hash 局限 |
| R4 | 5 excellent | accept | novelty 定位、对比基线、扩展性 |

---

## 总体策略

1. **再平衡，而非纯加法**：4 页 demo paper，版面零和。压缩 MU 实验细节 → 腾给 demo 交互 / token 可视化 / 扩展性讨论。
2. **细节转嫁全文**：MU 详表、压力测试、存储详表都已在 arXiv 全文，demo paper 用一句话 + `\cite` 指过去即可。
3. **安全修改**：已 accept，camera-ready 只做澄清 / 重新定位 / 补限制说明，不引入需重审的新主张。

---

## 弱点 → 主题 交叉引用（确保无遗漏）

| 主题 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| A 相关工作 & novelty 定位 | W2 W7 | | | W1 W5 |
| B 回归 demo 重心 | W1 W5 | | | |
| C 交互场景 & token 可视化 | W3 W6 | | | |
| D 扩展性 / 压力测试 | W4 | | | W5 |
| E 存储开销讨论 | | W1 | | W4 |
| F track() 日志瓶颈 | | W2 | | |
| G author vs contributor 语义 | | | W2 | W2 |
| H transformation gap 定位 | | W3 | W1 | W3 |
| I 哈希管理局限 | | | W3 | |

---

## 主题 A — 相关工作对比表 & novelty 定位
**来源**：R1-W2, R1-W7, R4-W1, R4-W5　**优先级 P0**　**文件**：`intro.tex`（Table 1, L7-30; L32 主张）、`references.bib`

R1/R4 一致指出：对比表漏掉数据血缘 / 数据治理类系统，且 "first deterministic method" 主张过强；novelty 实为系统集成而非新理论。

- [ ] **A1 扩充 Table 1**（`tab:provenance-comparison`）：至少补 2-3 类被漏的系统
  - 数据血缘：OpenLineage、Apache Atlas、Spline（任选 2）
  - 数据治理/目录：DataHub 或 Amundsen 或 HF Data Cards
  - 行级版本：Dolt 已有，补一句区别（无 author 归属）
  - 新增一列 **"Author attribution"** 或 **"Pipeline integration"**，显式区分维度
  - `references.bib` 新增对应条目（openlineage / atlas / spline / datahub）
- [ ] **A2 修正过强主张**（`intro.tex:32`）
  - 现：`the first deterministic method to construct them`
  - 改：`a deterministic, model-free method to construct forget sets`（删 "first"）
- [ ] **A3 明确 novelty 定位**（intro 段落 + system 开头）
  - 写清贡献是「把 record/token 级 provenance 内嵌进 HF/Datatrove pipeline + 可撤销」，不是新 provenance 理论
  - 一句话区分「provenance theory（W3C PROV、OpenLineage）vs 我们做的 pipeline-embedded authorship」
- [ ] **A4 对比基线强化**（R4-W5）：在 evaluation 或 intro 加一句「与 DVC/LakeFS 的 file-level 撤销对比，ob 的 record-level 把 over-deletion 从 X× 降到 ~1×」

---

## 主题 B — 回归 demo 重心
**来源**：R1-W1, R1-W5　**优先级 P0**　**文件**：`evaluation.tex`（§Machine Unlearning, L16-22）、`demo.tex`

R1：demo paper 应以现场体验为主，当前 MU 实验占比过大；缺可用性/UX 评估。

- [ ] **B1 压缩 MU 段**：现 Table 2（`tab:mu-results`）18 行 → 缩成 3-4 行摘要（仅 NPO line vs rand 的 retain PPL 对比 + MIA 区分度），详表移全文 arXiv 引用
- [ ] **B2 MU 文字精简**：evaluation.tex L16-20 改成 2-3 句，核心保留「provenance forget-set 在 retain PPL 上优于 random 5-20%」+ transformation gap 一句
- [ ] **B3 腾出版面给 demo**：省下的 ~0.3-0.4p 给主题 C（交互场景 + token 可视化）
- [ ] **B4 加一句非准确性评估**（R1-W5）：demo.tex 加「现场 demo 观察项」或简短可用性说明（informal，无需正式用户研究），例如「attendee 可在 <30s 内完成一次 GDPR 删除流程」

---

## 主题 C — 交互场景 & token-level 可视化
**来源**：R1-W3, R1-W6　**优先级 P0**　**文件**：`demo.tex`、`figures-cikm/`

R1：七个 view 只列了浏览/撤销，缺交互剧本；token-level provenance 在 system 描述了但 demo 里无可视化。

- [ ] **C1 增 2-3 个具体 demo 剧本**（walkthrough），每段 2-3 句
  - 剧本1：GDPR right-to-erasure 全流程（查询作者 → 评估影响 → 三级撤销 → 审计日志）
  - 剧本2：license 审计（按 license 筛选 records → 导出合规报告）
  - 剧本3：导出 unlearning forget-set（revoke → `ob generate-set` → 喂给 NPO/RMU）
- [ ] **C2 token-level 可视化**（R1-W6，核心缺口）
  - 选项 a（推荐）：webapp 加一个 token-level 视图（record 展开后按 token 高亮归属作者），截图放 `figures-cikm/`，demo.tex 配图说明
  - 选项 b：用一个 TikTok/示意图展示 packed sequence 里 token range → source 的映射
  - 至少要有一张图 + 一段文字体现 token-level 交互
- [ ] **C3 现有 7 view 描述加交互动词**：把「shows / displays」改成「click → trace → revoke → undo」式动作链

---

## 主题 D — 扩展性 / 压力测试
**来源**：R1-W4, R4-W5　**优先级 P1**　**文件**：`evaluation.tex`

R1/R4：只报单用户查询延迟，无并发用户、企业级、failure-prone pipeline 讨论。

- [ ] **D1 加「并发与企业部署」讨论段**（~3-4 句）
  - 架构天然可水平扩展：无状态、文件系统存储、PID 隔离多进程写、`ob clean` 合并
  - 只读查询无锁，多并发读安全；写经 WAL + PID 隔离
  - 企业级瓶颈在文件系统 inode / 大目录，256-bucket 分片缓解
- [ ] **D2 引用全文扩展性数据**：full paper 有 3-run avg scalability 表（1k-220k），一句「详见 arXiv:2607.13037 Table 5」即可
- [ ] **D3 failure-prone pipeline**（R4-W5）：system.tex 已提 WAL crash recovery，evaluation 强调一句「track() 在 WAL 下可从 pipeline 中断恢复，无 provenance 丢失」
- [ ] **D4（可选）并发 webapp 小测**：若有余力，跑个 wrk/k6 测 QPS，即使一句「single-node FastAPI 支撑 N 并发查询」也回应了 R1-W4；不做也可接受（demo paper）

---

## 主题 E — 存储开销讨论
**来源**：R2-W1, R4-W4　**优先级 P1**　**文件**：`evaluation.tex`（L14）

R2/R4：JSONL 元数据比最高 2.7×，需讨论何时可接受 / 替代方案。

- [ ] **E1 存储开销扩成一段**（现仅一句）
  - 说明 trade-off：JSONL 选型为 human-readable + Git-trackable + 无 DB 依赖（合规审计友好）
  - 何时可接受：中小规模 + 合规场景（审计可读性 > 存储成本）
  - 何时需替代：web-scale → streaming mode（system.tex:25 已述，token-index 流式不产 JSONL）或 binary index
  - 引用全文 storage 表（full paper 0.22×-0.32× at 220k，对比 demo paper 的 1.6×-2.7×，差异来自测量口径）
- [ ] **E2 明确测量口径**：demo paper 测 JSONL 全量；full paper 测 streaming（无 JSONL）。一句话澄清「2.7× 是 file-mode JSONL 上界，streaming mode 降至 ~0.3×」

---

## 主题 F — track() 日志瓶颈
**来源**：R2-W2　**优先级 P1**　**文件**：`system.tex`（L29）、`evaluation.tex`

R2：track logging 引入执行瓶颈。

- [ ] **F1 system.tex 明确 track() 成本构成**：SHA-256 计算 + WAL 同步写；说明 PID 隔离避免多进程锁争用
- [ ] **F2 evaluation 已报 1-21% pipeline 开销 = track 全部成本**，加一句明确指向「§Computational Overhead 的 1-21% 即 track() 端到端开销，无可隐藏瓶颈」
- [ ] **F3 讨论优化空间**：WAL 当前同步写可改异步批量（future work），或 streaming mode 绕过 JSONL

---

## 主题 G — author vs contributor 法律语义
**来源**：R3-W2, R4-W2　**优先级 P1**　**文件**：`system.tex`（L7）、`intro.tex`

R3/R4：author/contributor 区分对合规过于干净；协作 / 转化内容里 contributor 可能仍有权益；无法删历史编辑是局限。

- [ ] **G1 system.tex L7 强化法律语境**：承认协作 / 派生内容里 contributor（被覆盖编辑者）在版权上可能仍有派生作品权益
- [ ] **G2 明确「不暴露 contributor revoke」是当前设计选择**，而非能力缺失：「ob 记录 contributor 但当前不暴露 `revoke --contributor`，因被覆盖内容已无可见副本可撤；未来工作可支持历史版本撤销」
- [ ] **G3 在 demo/evaluation 加一句承认局限**：协作数据集（如 Wikipedia）的转化贡献，版权语义比系统模型复杂

---

## 主题 H — transformation gap / unlearning 定位
**来源**：R2-W3, R3-W1, R4-W3　**优先级 P1**　**文件**：`evaluation.tex`（L20）、`abstract`、`conclusion`

R2/R3/R4（三人共指）：provenance 找准了 source record，但模型已把内容转化/泛化，unlearning 后 extraction 仍高。当前已三处诚实承认。

- [ ] **H1 强化「必要不充分」定位**：明确「ob 解决 forget-set 构造（deterministic、model-free、可审计），unlearning 效果是正交独立问题」
- [ ] **H2 加一句合规价值**：即使 unlearning 不完美，provenance 让 forget-set 可复现 / 可审计，本身满足 GDPR 的问责要求
- [ ] **H3 三处措辞统一**：abstract / evaluation / conclusion 的 transformation gap 表述对齐，避免一处重一处轻

---

## 主题 I — 哈希管理局限
**来源**：R3-W3　**优先级 P2**　**文件**：新增 limitations 段（`system.tex` 末或 conclusion 前）

R3：hash 管理的固有局限——版本控制不足、数据损坏/丢失、pipeline 外篡改。

- [ ] **I1 新增一段 Limitations / Discussion**（4-5 句）：
  - **pipeline 外篡改不可见**：ob 只跟踪经 `track()` 的记录，pipeline 外手改数据文件无法捕获（与所有 provenance 系统同）
  - **数据损坏 / 丢失**：`ob reconcile` 用 hash（Pass 1）+ semantic（Pass 2）重链，full paper 报 96.9%-98.2% 恢复率，引用之
  - **hash 碰撞**：SHA-256 碰撞概率 ~2^-128，引用 `nist2015fips1804` / `katz2020moderncrypto`，工程上可忽略
  - **版本历史**：ob 存当前快照而非全编辑史（contributor 记录部分缓解），完整历史需配合 Git
- [ ] **I2 conclusion 加一句 future work**：完整编辑史撤销、semantic-aware 篡改检测

---

## 执行顺序建议

1. **先做版面再平衡（B1-B3）**：先把 MU 表压成摘要，腾出版面，否则后续都是空中楼阁
2. **主题 A（对比表 + novelty）**：技术含量高、改动 intro 骨架，优先
3. **主题 C（交互 + token 可视化）**：可能需要 webapp 改动 + 截图，耗时最长，尽早启动
4. **主题 E/F/G/H/I**：多为加段落，可并行
5. **主题 D**：若无新实验，纯讨论段最后补

---

## 全弱点 Checklist（逐条核销）

| # | 弱点 | 落点主题 | 状态 |
|---|---|---|---|
| R1-W1 | MU 实验占比过大 | B1 | ☐ |
| R1-W2 | 对比表漏数据血缘/治理 | A1 | ☐ |
| R1-W3 | 交互场景不足 | C1 C3 | ☐ |
| R1-W4 | 无并发/企业级扩展性 | D1 D2 | ☐ |
| R1-W5 | 缺可用性/UX 评估 | B4 | ☐ |
| R1-W6 | token-level 可视化缺失 | C2 | ☐ |
| R1-W7 | "first deterministic" 过强 | A2 | ☐ |
| R2-W1 | JSONL 存储开销 2.7× | E1 E2 | ☐ |
| R2-W2 | track 日志瓶颈 | F1 F2 | ☐ |
| R2-W3 | unlearning 后 extraction 仍高 | H1 H2 | ☐ |
| R3-W1 | 模型存储转化知识 | H1 | ☐ |
| R3-W2 | author/contributor 语义 + 历史编辑 | G1 G2 | ☐ |
| R3-W3 | hash 管理局限 | I1 | ☐ |
| R4-W1 | novelty 是集成非新理论 | A3 | ☐ |
| R4-W2 | author/contributor 法律 nuance | G1 G3 | ☐ |
| R4-W3 | unlearning 验证 inconclusive | H1 H3 | ☐ |
| R4-W4 | 存储 1.6-2.7× 何时可接受 | E1 | ☐ |
| R4-W5 | 对比基线 + 压力测试 | A4 D3 | ☐ |

---

## 页面预算跟踪（4 页 demo）

| 章节 | 现预算 | 调整 |
|---|---|---|
| Intro | 0.70p | +0.10（A1 表扩充） |
| System | 1.30p | +0.10（F1/G1/I1 段落） |
| Demo | 0.20p | **+0.40**（C1 剧本 + C2 token 图）← 核心增量 |
| Evaluation | 1.30p | **-0.40**（B1 MU 压缩）+0.10（D1/E1 段） |
| 其他 | 0.50p | 不变 |

净变化 ≈ 0，靠 MU 压缩腾给 demo。若 camera-ready 允许 +1p，则 D/E/I 讨论段有空间。
