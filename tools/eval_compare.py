# 跨公司对比回归检查：定点抽取是否覆盖十家公司、单位归一化是否合理、排序是否单调。
#
# usage: python tools/eval_compare.py
import json
import sys
import time

sys.path.insert(0, ".")
from app.compare import METRICS, compare_companies, metric_from_question

AMOUNT_METRICS = ("revenue", "net_profit", "op_cashflow")
# 十家公司营业/利润/现金流的量级区间，超出即说明单位换算或口径取错
REVENUE_RANGE = (50.0, 5000.0)        # 亿元
CASHFLOW_RANGE = (-2000.0, 2000.0)    # 亿元
EPS_RANGE = (-3.0, 6.0)               # 元/股

QUESTIONS = [
    ("哪家公司的营业收入最高？", "revenue"),
    ("十家公司的净利润排名", "net_profit"),
    ("哪家企业经营活动现金流最强？", "op_cashflow"),
    ("各家资产负债率分别是多少", "debt_ratio"),
    ("每股收益谁最高", "eps"),
]


def load_chunks(path="data/chunks.jsonl"):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> int:
    t0 = time.time()
    chunks = load_chunks()
    failures = []

    for metric in AMOUNT_METRICS:
        out = compare_companies(chunks, None, metric)
        label = METRICS[metric].label
        for r in out["rows"]:
            if not r["page"] or not r["pdf_url"] or not r["raw_row"]:
                failures.append(f"{label}/{r['company_display']} 缺少出处信息")
        if metric == "revenue":
            bad = [r for r in out["rows"] if not (REVENUE_RANGE[0] <= r["value"] <= REVENUE_RANGE[1])]
            if bad:
                failures.append("营业收入量级异常：" + "、".join(
                    f"{r['company_display']}={r['value']:.1f}亿" for r in bad))
        if metric == "op_cashflow":
            bad = [r for r in out["rows"] if not (CASHFLOW_RANGE[0] <= r["value"] <= CASHFLOW_RANGE[1])]
            if bad:
                failures.append("经营现金流量量级异常：" + "、".join(
                    f"{r['company_display']}={r['value']:.1f}亿" for r in bad))
        print(f"[{label}] 取到 {len(out['rows'])}/{out['n_companies']} 家，"
              f"报告期 {len(out['periods'])} 种（{('、'.join(out['periods'])) or '-'}），"
              f"未取到 {len(out['missing'])} 家")
        for r in out["rows"]:
            print(f"   {r['company_display']:<8s} {r['value']:>12,.2f} 亿元  "
                  f"(原文 {r['value_raw']:,.2f} {r['raw_unit']}，第 {r['page']} 页)")
        if len(out["rows"]) < out["n_companies"]:
            print(f"   未取到：{('、'.join(m['company_display'] for m in out['missing'])) or '-'}")

    for metric in ("eps", "debt_ratio"):
        out = compare_companies(chunks, None, metric)
        label = METRICS[metric].label
        bad = [r for r in out["rows"]
               if metric == "eps" and not (EPS_RANGE[0] <= r["value"] <= EPS_RANGE[1])]
        if bad:
            failures.append("基本每股收益量级异常：" + "、".join(
                f"{r['company_display']}={r['value']}" for r in bad))
        print(f"[{label}] 取到 {len(out['rows'])}/{out['n_companies']} 家"
              f"（该指标并非所有公司都以表格披露，未取到属正常）")

    # 排序单调性：升序必须 <= 前一个，降序必须 >= 前一个
    for metric in METRICS:
        for sort in ("desc", "asc"):
            out = compare_companies(chunks, None, metric, sort)
            values = [r["value"] for r in out["rows"]]
            expect = sorted(values, reverse=(sort == "desc"))
            if values != expect:
                failures.append(f"{metric}/{sort} 排序不单调：{values[:5]}")

    # 公司筛选：只选两家时只应产出两家，且不应出现别家
    picked = compare_companies(chunks, ["上汽", "比亚迪"], "revenue")
    names = {r["company_display"] for r in picked["rows"]}
    if picked["n_companies"] != 2 or len(picked["rows"]) != 2 or names != {"上汽集团", "比亚迪"}:
        failures.append(f"公司筛选异常：scope={picked['n_companies']} rows={names}")

    for question, expected in QUESTIONS:
        got = metric_from_question(question)
        if got != expected:
            failures.append(f"指标识别错误：{question!r} -> {got!r}（应为 {expected!r}）")

    try:
        compare_companies(chunks, None, "not_a_metric")
        failures.append("未知指标没有报错")
    except ValueError:
        pass

    print()
    print(f"检查耗时 {time.time() - t0:.1f}s")
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for f in failures:
            print("  - " + f)
        return 1
    print("跨公司对比检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
