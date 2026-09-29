# 财报 PDF 下载脚本：从巨潮资讯网（cninfo）检索并下载上市公司定期报告。
#
# cninfo 同时收录沪市与深市公告，一份脚本覆盖全部十家公司。
#
# 安全约定：
#   1. 默认跳过 公司财报/ 下已存在的同名文件，绝不覆盖已有文件，
#      只有显式加 --overwrite 才会覆盖；
#   2. 保存文件名沿用仓库现有命名，app/ingest.py 靠文件名反推公司 key
#      （上汽 / 北汽蓝谷 / 宇通汽车 / 比亚迪 …），改名会让 key 与元信息对不上；
#   3. 只依赖标准库，不需要 requests。
#
# 用法：
#   python tools/fetch_reports.py --list                     # 只列出匹配到的公告，不下载
#   python tools/fetch_reports.py                            # 下载缺失的报告
#   python tools/fetch_reports.py --only 上汽,比亚迪 --overwrite
#   python tools/fetch_reports.py --year 2027 --overwrite     # 换一年重新抓
#   python tools/fetch_reports.py --url "https://static.cninfo.com.cn/xx.pdf" --name 新报告.pdf
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "公司财报"

# 巨潮的两个接口：公告检索 + 证券代码查 orgId
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
ORGID_URL = "https://www.cninfo.com.cn/new/information/topSearch/detailOfQuery"
PDF_BASE = "https://static.cninfo.com.cn/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
}

# cninfo 用 column 区分市场；逐个尝试，命中即停
COLUMN_CANDIDATES = ("sse", "szse", "bjse")
# 定期报告在 cninfo 的分类编码
CATEGORY = {
    "half": "category_bndbg_szsh",   # 半年度报告
    "annual": "category_ndbg_szsh",  # 年度报告
}
REPORT_WORDS = {"half": "半年度报告", "annual": "年度报告"}

# 标题里带这些字的不是完整报告，跳过
EXCLUDE_WORDS = ("摘要", "英文", "修订", "更正", "补充", "取消", "问询", "回复", "决议")
# 附件优先选带"全文"的
PREFER_WORDS = ("全文",)
MIN_PDF_BYTES = 50 * 1024


# ---------------------------------------------------------------- 公司清单
@dataclass(frozen=True)
class Company:
    code: str          # 证券代码，6 位
    key: str           # 公司简称（与仓库现有文件名对齐）
    template: str      # 保存文件名模板，可含 {year}
    report: str        # half=半年报，annual=年报
    year: int          # 报告年度
    display: str = ""  # 展示名，仅用于打印


# 命名与 公司财报/ 现有文件保持一致；两份原名里带年份，用 {year} 占位
COMPANIES = [
    Company("600104", "上汽", "上汽半年报.pdf", "half", 2026, "上汽集团"),
    Company("600733", "北汽蓝谷", "北汽蓝谷半年度报告.pdf", "half", 2026, "北汽蓝谷"),
    Company("600066", "宇通汽车", "宇通汽车半年报.pdf", "half", 2026, "宇通客车"),
    Company("601238", "广汽", "广汽半年报.pdf", "half", 2026, "广汽集团"),
    Company("002594", "比亚迪", "比亚迪：{year}年半年度报告.PDF", "half", 2026, "比亚迪"),
    Company("600418", "江淮汽车", "江淮汽车半年报.pdf", "half", 2026, "江淮汽车"),
    Company("601127", "赛力斯", "赛力斯半年报.pdf", "half", 2026, "赛力斯"),
    Company("600686", "金龙汽车", "金龙汽车半年报.pdf", "half", 2026, "金龙汽车"),
    Company("601633", "长城汽车", "长城汽车{year}年报.pdf", "annual", 2025, "长城汽车"),
    Company("000625", "长安汽车", "长安汽车：{year}年半年度报告.PDF", "half", 2026, "长安汽车"),
]


# ---------------------------------------------------------------- HTTP
def post_form(url, params, timeout, retries=3):
    last = None
    for attempt in range(retries):
        try:
            body = urllib.parse.urlencode(params).encode("utf-8")
            req = urllib.request.Request(
                url, data=body,
                headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"POST {url} 失败：{last}")


def download_pdf(url, dest, timeout, retries=3):
    # 先写 .part，校验通过再改名，避免半成品混进语料目录
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={**HEADERS, "Accept": "application/pdf, */*"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                blob = resp.read()
            if not blob.startswith(b"%PDF"):
                raise RuntimeError(f"返回内容不是 PDF（前 16 字节 {blob[:16]!r}）")
            if len(blob) < MIN_PDF_BYTES:
                raise RuntimeError(f"文件只有 {len(blob)} 字节，疑似错误页")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(blob)
            tmp.replace(dest)
            return len(blob)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"下载失败：{last}")


# ---------------------------------------------------------------- 检索
def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def lookup_org_id(code, timeout):
    # orgId 是 cninfo 内部的机构 id，检索公告必须带上；该接口字段名改过几次，
    # 这里全量扫描返回的 JSON，只要找到 code 与 orgId 同时出现的节点即可
    # 该接口只接受 POST：用 GET 带参数会被服务端判成系统异常，直接返回 HTTP 500
    payload = post_form(ORGID_URL, {"keyWord": code, "maxSecNum": 10, "maxListNum": 10}, timeout)
    for node in _walk(payload):
        if str(node.get("code", "")).strip() == code and node.get("orgId"):
            return str(node["orgId"]).strip()
    raise RuntimeError(f"cninfo 未查到证券代码 {code} 的 orgId")


def fmt_date(ms):
    if not ms:
        return "-"
    return time.strftime("%Y-%m-%d", time.localtime(ms / 1000))


def pick_announcements(items, company):
    word = REPORT_WORDS[company.report]
    year = str(company.year)
    hits = []
    for item in items:
        title = re.sub(r"\s+", "", (item.get("announcementTitle") or "")).strip()
        if not title or word not in title or year not in title:
            continue
        if any(bad in title for bad in EXCLUDE_WORDS):
            continue
        adjunct = (item.get("adjunctUrl") or "").strip()
        if not adjunct:
            continue
        hits.append({
            "title": title,
            "url": PDF_BASE + adjunct.lstrip("/"),
            "size": int(item.get("adjunctSize") or 0),
            "date": fmt_date(int(item.get("announcementTime") or 0)),
        })
    # 同一份报告常有多个附件：优先"全文"，其次取体积最大的
    hits.sort(key=lambda h: (any(w in h["title"] for w in PREFER_WORDS), h["size"]), reverse=True)
    return hits


def date_windows(company):
    """公告检索的时间窗。年报按报告年度在次年 1-4 月披露，只搜当年会一条都搜不到。"""
    year = company.year
    if company.report == "annual":
        return [f"{year + 1}-01-01~{year + 1}-06-30", f"{year}-01-01~{year}-12-31"]
    return [f"{year}-01-01~{year}-12-31"]


def find_report(company, org_id, timeout):
    year = company.year
    last_error = ""
    for column in COLUMN_CANDIDATES:
        for window in date_windows(company):
            params = {
                "pageNum": 1,
                "pageSize": 30,
                "column": column,
                "tabName": "fulltext",
                "plate": "",
                "stock": f"{company.code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": CATEGORY[company.report],
                "trade": "",
                "seDate": window,
                "sortName": "",
                "sortType": "",
                "isHLtitle": "false",
            }
            try:
                payload = post_form(QUERY_URL, params, timeout)
            except RuntimeError as exc:
                last_error = f"{column} 市场查询失败：{exc}"
                continue
            hits = pick_announcements(payload.get("announcements") or [], company)
            if hits:
                best = dict(hits[0])
                best["others"] = hits[1:4]
                return best
            last_error = f"{column} 市场 {window} 没有 {year} 年{REPORT_WORDS[company.report]}"
    raise RuntimeError(last_error or "未检索到公告")


# ---------------------------------------------------------------- 主流程
def resolve_companies(only):
    if not only:
        return list(COMPANIES)
    # 先精确匹配简称/代码/展示名，也支持"宇通"这类不完整输入；
    # 但"汽车"这种会命中多家，必须让用户选，不能静默多抓
    wanted = [t.strip() for t in re.split(r"[,，]", only) if t.strip()]
    picked, unknown = [], []
    for token in wanted:
        exact = [c for c in COMPANIES if token in (c.key, c.code, c.display)]
        if len(exact) == 1:
            hit = exact[0]
        else:
            partial = [c for c in COMPANIES if token in c.key or token in c.display]
            if len(partial) == 1:
                hit = partial[0]
            elif len(partial) > 1:
                raise SystemExit(f"“{token}”匹配到多家：" +
                                 ", ".join(f"{c.key}({c.code})" for c in partial) + "，请写全称")
            else:
                unknown.append(token)
                continue
        if hit not in picked:
            picked.append(hit)
    if unknown:
        known = ", ".join(f"{c.key}({c.code})" for c in COMPANIES)
        raise SystemExit(f"未知公司：{', '.join(unknown)}；可选：{known}")
    return picked


def main():
    parser = argparse.ArgumentParser(
        description="从巨潮资讯网下载上市公司定期报告 PDF",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录，默认 公司财报/")
    parser.add_argument("--only", default="", help="只处理指定公司，逗号分隔，可用简称或代码")
    parser.add_argument("--year", type=int, default=0, help="覆盖报告年度，默认用脚本内置值")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名文件")
    parser.add_argument("--list", action="store_true", help="只列出能匹配到的公告，不下载")
    parser.add_argument("--sleep", type=float, default=1.5, help="每次请求间隔秒数")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次请求超时秒数")
    parser.add_argument("--url", default="", help="直接下载指定 PDF URL，跳过检索")
    parser.add_argument("--name", default="", help="配合 --url 指定保存文件名")
    args = parser.parse_args()

    out_dir = Path(args.out)

    # 直接下载模式：已知链接时最省事，也是检索接口变动时的兜底
    if args.url:
        if not args.name:
            raise SystemExit("--url 需要同时提供 --name（保存文件名）")
        dest = out_dir / args.name
        if dest.exists() and not args.overwrite:
            print(f"[跳过] {dest.name} 已存在（加 --overwrite 才会覆盖）")
            return 0
        size = download_pdf(args.url, dest, args.timeout)
        print(f"[完成] {dest.name}  {size / 1024 / 1024:.1f} MB -> {dest}")
        return 0

    companies = resolve_companies(args.only)
    print(f"共 {len(companies)} 家公司，输出目录 {out_dir}")

    ok, skipped, failed = [], [], []
    for index, company in enumerate(companies):
        if args.year:
            company = Company(company.code, company.key, company.template,
                              company.report, args.year, company.display)
        filename = company.template.format(year=company.year)
        dest = out_dir / filename
        label = f"{company.display or company.key}({company.code}) {company.year}年{REPORT_WORDS[company.report]}"
        if index:
            time.sleep(args.sleep)
        try:
            org_id = lookup_org_id(company.code, args.timeout)
            best = find_report(company, org_id, args.timeout)
        except Exception as exc:
            failed.append((label, str(exc)))
            print(f"[失败] {label}：{exc}")
            continue

        note = "" if not best["others"] else f"（另有 {len(best['others'])} 个附件未用）"
        print(f"[匹配] {label} -> {best['title']}  {best['date']}  {best['size'] / 1024 / 1024:.1f} MB {note}")
        if best["others"]:
            for other in best["others"]:
                print(f"       备选：{other['title']}  {other['date']}")
        if args.list:
            skipped.append((label, "仅列出"))
            continue
        if dest.exists() and not args.overwrite:
            skipped.append((label, dest.name))
            print(f"[跳过] {dest.name} 已存在（加 --overwrite 才会覆盖）")
            continue
        try:
            size = download_pdf(best["url"], dest, args.timeout)
        except Exception as exc:
            failed.append((label, str(exc)))
            print(f"[失败] {label}：下载 {exc}")
            continue
        ok.append((label, dest.name))
        print(f"[完成] {dest.name}  {size / 1024 / 1024:.1f} MB -> {dest}")

    print()
    print(f"完成 {len(ok)}，跳过 {len(skipped)}，失败 {len(failed)}")
    if ok:
        print("已写入：" + ", ".join(name for _, name in ok))
    if failed:
        print("\n失败明细：")
        for label, reason in failed:
            print(f"  - {label}：{reason}")
        print("\n提示：若检索接口有变动，可用 --url 直接下载附件；"
              "也可以先用浏览器打开 www.cninfo.com 按公司名 + 报告类型手工确认。")
        return 1
    if ok and not skipped:
        print("\n下一步：重新构建索引（PDF 解析有缓存，只处理新文件）：python build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
