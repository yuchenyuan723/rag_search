# 公司财报检索问答知识库

一个财报混合检索问答系统。语料是 `公司财报/` 目录下 10 家公司的
10 份财报 PDF（9 份 2026 年半年报 + 长城汽车 2025 年报），切块后共 3903 段。
问答页直接引用财报原文句子作答，每句带公司、章节、页码出处，点击可跳转 PDF
对应页；比较类问题还会自动生成跨公司对比表，也可在顶部「跨公司对比」页签里按指标
拉排名。

## 功能

- PDF 解析（pdfplumber）：按页输出文本行，表格逐行还原成 Markdown（先识别有框线表格，无数值密度启发式兜底无框线表格），页眉页脚和目录引导线自动剔除。
- 切块与元信息：约 480 字的段落感知切块、相邻块重叠 90 字；每个切块带公司、公司全称、报告类型、报告期、四级章节路径和页码。
- 向量索引：Qwen3-Embedding-0.6B（1024 维，last-token pooling，查询带指令前缀），全量余弦扫描。
- BM25 索引：jieba 分词 + Okapi BM25，内置财经词表与同义词扩展。
- 混合检索：向量 0.6 + BM25 0.4 加权融合；界面可切换到纯向量或纯 BM25 做对比，也可按公司、报告类型过滤。
- 证据句作答：对召回切块做句子级重排，把最相关的原文句子作为答案，编号与来源面板一一对应，来源可跳转 PDF 页码。
- 可选 AI 总结（StepFun 阶跃星辰 LLM）：在证据句之上做归纳；未配置 API key 或调用失败时，自动回退为纯证据句展示，不阻塞问答。
- 跨公司对比：顶部页签「跨公司对比」，比较类问题也会在回答区自动附上对比表。指标从各公司「主要会计数据和财务指标」章节定点抽取（不碰担保条款、母公司报表附注），统一折算为 亿元 / 元每股 / %，支持升降序，每个数字都能回溯到 PDF 页码；某家公司没披露该项就进「未披露」，绝不凑数。除五个预设指标外，指标框还能直接输入财报表格行名做自定义对比（如「研发投入合计」「股东总数」）。

## 检索规则要点

下面这些规则决定了答案能不能落在正确的表格行上，改检索逻辑前最好先读一遍：

- BM25 只给问题里的实词打分。公司名、报告期词和单蹦的年份会先丢掉，否则术语表和「关联方存款」表（这些词重复几十次）会永远排在真正承载答案的表前面。
- 问题点名了公司，检索就锁死在这家公司内，即使是比较型问法：「赛力斯的营业收入构成中哪些业务占比最高」问的是赛力斯内部结构，不是跨公司比较。
- 跨公司问题（「哪家……最高」「十家公司中……」）每家公司保底出一个场：按公司各取最优段，召回上限抬到涉及的公司数（最多 12），保证十家公司都能同时出场。
- 数值型问题优先要「既命中实词、又带多个长数字」的切块，定义类、术语类短文不能靠数字密集混进来。
- 证据句必须覆盖问题里的某个实词。同一张表里没问到指标的行（流动比率、偿债比率之类）会被剔掉，不会摆在答案行旁边充数。
- 如果一个证据句都没覆盖到问题的实词，系统回答「未检索到」，而不是拿一张税率表糊弄。
- 表格行引用时保留章节标题前缀（例如「(2) 营业收入、营业成本的分解信息 | 合同分类 → 汽车整车 53942634056.20」），脱离上下文引用也读得懂。

## 目录结构

```text
app/
  config.py     # 路径、切块与检索参数；.env 在 app/__init__.py 里统一载入
  ingest.py     # PDF 解析：文本、表格还原、章节跟踪
  chunking.py   # 切块打包与元信息
  qwen3.py      # Qwen3-Embedding 封装
  bm25.py       # jieba 分词器 + Okapi BM25
  index.py      # 构建流水线（解析缓存在 data/extract/）
  retrieve.py   # 混合检索 + 证据句
  compare.py    # 跨公司指标抽取：同一指标、同一单位、页码可溯
  llm.py        # StepFun 客户端：密钥只从环境变量读
  server.py     # Flask API + 静态页面 + PDF 跳转
static/index.html   # 问答页面（原生 JS，无外部资源）：检索问答 + 跨公司对比两个页签
build_index.py       # 只构建索引
run.py               # 一键入口（索引缺失时自动构建）
fetch_reports.py     # 下载财报 PDF 的入口，等价于 python tools/fetch_reports.py
smoke_test.py        # 索引冒烟测试：直接调 KnowledgeBase 跑五道固定问题
tools/
  fetch_reports.py   # 从巨潮资讯网按股票代码下载财报 PDF 到 公司财报/
  rebuild_bm25.py    # 只重拟合 BM25（改了 app/bm25.py 词表后跑，约 10 秒）
  eval_retrieval.py  # 检索回归：10 道题 + 金标切块命中检查
  eval_compare.py    # 跨公司抽取回归：覆盖率、单位换算、排序单调性
  acceptance.py      # 验收跑批：导出十道验收题的原始召回数据
.env.example         # 配置模板（MODEL_PATH / LLM_API_KEY），复制成 .env 后填值
data/                # 构建产物（chunks / embeddings / bm25 / meta），不入库
公司财报/             # 10 家公司 10 份财报 PDF（语料），需自行下载，不入库
验收报告.md          # 十道验收题的结果与正确率统计
```

## 使用方法

### 1. 安装依赖

```powershell
# 本机无 NVIDIA 显卡，CPU 版 torch 就够
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. 配置 .env

```powershell
copy .env.example .env
notepad .env
```

`.env` 只有两项会真正影响运行：`MODEL_PATH` 指向本地 Qwen3-Embedding-0.6B 目录（留空则用项目下的 `models/Qwen3-Embedding-0.6B`，详见「模型准备」）；`LLM_API_KEY` 填 StepFun API Key，留空就只展示财报原文摘句、不生成 AI 总结。`.env` 与 `.env.*` 已被 `.gitignore` 忽略，不会进仓库，`.env.example` 才是提交上去的模板。

其他路径也都能用环境变量覆盖：`FIN_CORPUS_DIR`（财报目录）、`FIN_DATA_DIR`（索引产物目录）、`FIN_MODEL_DIR`（模型目录，兼容旧写法）。

### 3. 下载财报 PDF

运行 `python fetch_reports.py` 下载所需 PDF，再执行分析代码：

```powershell
python fetch_reports.py --list    # 只列出能匹配到的公告，不下载
python fetch_reports.py           # 把缺失的报告下载到 公司财报/
```

脚本从巨潮资讯网按证券代码检索定期报告，内置清单与仓库语料一致：上汽 600104、北汽蓝谷 600733、宇通客车 600066、广汽集团 601238、比亚迪 002594、江淮汽车 600418、赛力斯 601127、金龙汽车 600686、长城汽车 601633、长安汽车 000625。更多用法：

```powershell
python fetch_reports.py --only 上汽,比亚迪 --overwrite   # 只抓指定公司并覆盖旧文件
python fetch_reports.py --year 2024 --overwrite           # 换一年重新抓
```

注意年报按报告年度在次年 1-4 月披露，所以 `--year 2024` 找的是 2025 年初发布的那份年报。默认跳过 `公司财报/` 下已存在的同名文件，只有显式加 `--overwrite` 才覆盖。保存文件名沿用仓库现有命名，`app/ingest.py` 靠文件名反推公司 key，改名会让 key 与元信息对不上。

### 4. 构建索引并启动

```powershell
python build_index.py    # 首次运行要解析全部 PDF 并给每个切块算向量
python run.py            # 浏览器打开 http://127.0.0.1:8000
```

页面打不开、接口报 404/500、或者改了代码却看不到效果时，先确认 8000 端口上只有一个 `run.py`：Windows 允许多个进程同时监听同一端口，早先启动的旧服务会继续拿旧页面、旧代码应答，新进程反而接不到请求。用 `netstat -ano | findstr :8000` 找到旧进程并关掉，再 Ctrl+F5 强刷页面。

### 模型准备

向量模型是 **Qwen3-Embedding-0.6B**（`Qwen/Qwen3-Embedding-0.6B`，1024 维，仓库里用 last-token pooling）。模型文件不入库，需要自行准备并放到 `.env` 里 `MODEL_PATH` 指定的目录，二选一：

```powershell
# 方式一：模型放在别处，.env 里写绝对/相对路径
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir D:\projects\qwen3_model
# .env: MODEL_PATH=D:\projects\qwen3_model

# 方式二：模型放在项目目录下，MODEL_PATH 留空即可
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir .\models\Qwen3-Embedding-0.6B
```

目录里需要有 `config.json`、`tokenizer.json`、`model.safetensors`（约 1.2 GB）。CPU 上给 3903 个切块算向量约 3 小时，`FIN_EMB_SHARDS=4 python build_index.py` 可以按进程分片加速。

### 5. 增量重建与检索回归

在 `公司财报/` 里新增或替换 PDF 后，重新跑 `python build_index.py`（已解析过的文档有缓存）或 `python run.py --rebuild`。

如果只是改了 `app/bm25.py` 里的财经词表或同义词表，向量不用重算，但 BM25 索引要重拟合，新词才会进分词结果：

```powershell
python tools/rebuild_bm25.py
```

改完之后检查检索质量有没有退化：

```powershell
python tools/eval_retrieval.py
# 金标切块命中率、金标进入证据句比例、证据句实词覆盖率、逐题耗时
python tools/smoke_test.py
# 不打日志，直接在终端打印五道固定问题的证据句与来源
```

### 跨公司对比

界面顶部页签切到「跨公司对比」：选指标（营业收入 / 归母净利润 / 经营活动现金流净额 / 基本每股收益 / 资产负债率，或直接在指标框里输入财报表格行名，如「研发投入合计」「合同负债」「股东总数」）、选升序或降序、勾选左侧公司范围，点生成即可得到带数值条、披露口径和原文页码链接的排名表。接口形式：

```powershell
$body = @{ companies = @(); metric = "revenue"; sort = "desc" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/compare -Method Post -Body $body -ContentType "application/json"
```

`companies` 留空表示全部十家；指标名非法时返回 400 和中文错误信息。回归检查用 `python tools/eval_compare.py`，会复查五个指标的覆盖率、单位换算和排序。

自定义指标：`metric` 不限于五个预设，也可以直接写财报表格里的行名（最多 30 个字符），前端把下拉改成了输入框，预设只作为提示。规则没变：仍然只在各公司「主要会计数据 / 主要财务指标」章节的表格行名上匹配（自定义指标会额外放宽到行名中间包含匹配）；金额按表头单位折算成亿元，百分比和每股指标保持原单位，计数类随各公司原文单位（各家不一致时界面会提示）。取不到的公司进「未披露」，不凑数。

## 验收结论

十道验收题（含两道跨公司全景题）的召回块、证据句、标准答案和逐题判定见 [验收报告.md](验收报告.md)：正确 6 题（60%）、部分正确 3 题（30%）、错误 1 题（10%），平均响应 7.7 秒。报告里的原始召回数据可由 `python tools/acceptance.py` 重新生成（输出 `tools/acceptance_raw.json`）。

已知短板集中在「一个问句里塞多个指标或多个限定」的组合题，完整清单和改进建议见验收报告第四节。

## 说明

本地模型 Qwen3-Embedding-0.6B 是向量模型、不能生成文本，所以默认答案是抽取式的：页面展示财报里最相关的原文句子并给出处，每个数字都能通过 `#page=N` 链接回到源 PDF 核对。需要生成式回答时，在项目根目录 `.env` 里填 `LLM_API_KEY`（StepFun API Key）即可启用 AI 总结，不填或调用失败会自动回退到原文摘句。

安全约定：`.env`、`公司财报/*.pdf`、模型权重（`*.safetensors`、`*.bin`）、索引产物 `data/` 一律不入库，`.gitignore` 已逐条覆盖；克隆仓库后按上面第 3、4 步准备 PDF 与模型，再 `python build_index.py`。
