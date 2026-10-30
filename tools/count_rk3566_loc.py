#!/usr/bin/env python3
"""ADAS Edge code line counter; ARM implementation separated from tests, RTL and archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tokenize
from collections import defaultdict
from pathlib import Path

EXT = {
    ".py": "Python", ".c": "C", ".h": "C/C++", ".cpp": "C++",
    ".cc": "C++", ".hpp": "C/C++", ".qml": "QML",
    ".js": "JavaScript", ".mjs": "JavaScript", ".html": "HTML",
    ".css": "CSS", ".sh": "Shell", ".cmd": "Batch",
    ".v": "Verilog", ".sv": "SystemVerilog", ".m": "MATLAB",
}
SKIP_DIR = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", "build", "dist", "generated", "vendor",
    "third_party", "third-party", "external", "results",
    "assets", "artifacts", "checkpoints", ".loc_report",
}
SKIP_FILE = {"ui_mainwindow.py", "resources_rc.py"}


def category(path: str) -> str:
    if path.startswith("archive/"):
        return "历史归档"
    if path.startswith(("rtl/", "fpga/", "hardware/fpga/")):
        return "FPGA RTL"
    if path.startswith(("competition_ai/", "competition_io/",
                        "src/adas_edge/", "deployment/hud/")):
        return "ARM核心"
    if path.startswith("deployment/lubancat/"):
        return "研发支撑" if "/tools/" in path else "ARM核心"
    if path.startswith(("deployment/rknn/", "tools/", "tests/",
                        "multistream_framework/", "matlab/")):
        return "研发支撑"
    return "其他代码"


def python_counts(data: bytes, lines: list[str]) -> tuple[int, int]:
    code, comments = set(), set()
    try:
        for tok in tokenize.tokenize(io.BytesIO(data).readline):
            if tok.type == tokenize.COMMENT:
                comments.add(tok.start[0])
            elif tok.type in (tokenize.ENCODING, tokenize.ENDMARKER,
                              tokenize.NL, tokenize.NEWLINE,
                              tokenize.INDENT, tokenize.DEDENT):
                continue
            elif tok.string.strip():
                end = tok.end[0] if tok.type == tokenize.STRING else tok.start[0]
                for row in range(tok.start[0], end + 1):
                    if 0 < row <= len(lines) and lines[row - 1].strip():
                        code.add(row)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return generic_counts(lines, ".py")
    return len(code), len(comments - code)


def generic_counts(lines: list[str], suffix: str) -> tuple[int, int]:
    code = comment = 0
    in_block = in_html = False
    shell = suffix in (".sh", ".cmd")
    matlab = suffix == ".m"
    for line in lines:
        s = line.strip()
        if not s:
            continue
        has_code = has_comment = False
        i = 0
        quote = None
        while i < len(line):
            pair = line[i:i + 2]
            if in_html:
                has_comment = True
                if line.startswith("-->", i):
                    in_html = False
                    i += 3
                else:
                    i += 1
                continue
            if in_block:
                has_comment = True
                if pair == "*/":
                    in_block = False
                    i += 2
                else:
                    i += 1
                continue
            ch = line[i]
            if quote:
                has_code = True
                if ch == chr(92):
                    i += 2
                elif ch == quote:
                    quote = None
                    i += 1
                else:
                    i += 1
                continue
            if ch.isspace():
                i += 1
                continue
            if suffix == ".html" and line.startswith("<!--", i):
                in_html = has_comment = True
                i += 4
                continue
            if pair == "/*" and not shell and not matlab:
                in_block = has_comment = True
                i += 2
                continue
            if pair == "//" and not shell and not matlab and suffix not in (".html", ".css"):
                has_comment = True
                break
            if (shell or suffix == ".py") and ch == "#":
                if i == 0 and line.startswith("#!"):
                    has_code = True
                else:
                    has_comment = True
                break
            if matlab and ch == "%":
                has_comment = True
                break
            if ch in ('"', "'", "`"):
                quote = ch
            has_code = True
            i += 1
        if has_code:
            code += 1
        elif has_comment:
            comment += 1
    return code, comment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--csv", default="rk3566_loc.csv")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIR)
        for name in sorted(names):
            file = Path(base, name)
            suffix = file.suffix.lower()
            if suffix not in EXT or name in SKIP_FILE or name.endswith((".min.js", "_pb2.py")) or file.is_symlink():
                continue
            relative = file.relative_to(root).as_posix()
            data = file.read_bytes()
            if b"\x00" in data[:8192]:
                continue
            lines = data.decode("utf-8-sig", "replace").splitlines()
            nonblank = sum(bool(line.strip()) for line in lines)
            blank = len(lines) - nonblank
            c, comments = (python_counts(data, lines) if suffix == ".py"
                           else generic_counts(lines, suffix))
            rows.append({
                "范围": category(relative), "语言": EXT[suffix], "文件": relative,
                "总行": len(lines), "代码行": c,
                "注释行": comments, "空行": blank,
                "SHA256": hashlib.sha256(data).hexdigest(),
            })
    order = ["ARM核心", "研发支撑", "FPGA RTL", "历史归档", "其他代码"]
    totals = {}
    for group in order:
        subset = [r for r in rows if r["范围"] == group]
        totals[group] = {
            "files": len(subset), "total": sum(r["总行"] for r in subset),
            "code": sum(r["代码行"] for r in subset),
            "comments": sum(r["注释行"] for r in subset),
            "blank": sum(r["空行"] for r in subset),
        }
    core = [r for r in rows if r["范围"] == "ARM核心"]
    seen, duplicates = set(), []
    for r in core:
        if r["SHA256"] in seen and r["代码行"]:
            duplicates.append(r)
        else:
            seen.add(r["SHA256"])
    unique_code = totals["ARM核心"]["code"] - sum(r["代码行"] for r in duplicates)
    lines_md = [
        "# ADAS Edge / RK3566 代码量报告", "",
        "统计口径：仅源文件；不含 vendor、编译产物、图片、日志和测试结果。",
        "",
        "| 分类 | 文件数 | 总行数 | 代码行 | 注释行 | 空行 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    print("分类                    文件数      总行数      代码行      注释行      空行")
    for group in order:
        t = totals[group]
        print(f"{group:<20} {t['files']:>7,} {t['total']:>11,} {t['code']:>11,} {t['comments']:>11,} {t['blank']:>9,}")
        lines_md.append(f"| {group} | {t['files']:,} | {t['total']:,} | {t['code']:,} | {t['comments']:,} | {t['blank']:,} |")
    print(f"\nARM 核心去除完全相同文件后的代码行：{unique_code:,}；重复文件数：{len(duplicates)}")
    lines_md += [
        "", f"**ARM 核心代码行：{totals['ARM核心']['code']:,}**",
        f"（去除 SHA256 完全相同的文件后：{unique_code:,} 行）",
        "", "## ARM 核心最大的 20 个源文件", "",
        "| 文件 | 代码行 | 总行 |", "|---|---:|---:|",
    ]
    for r in sorted(core, key=lambda x: x["代码行"], reverse=True)[:20]:
        print(f"{r['代码行']:>7,}  {r['文件']}")
        lines_md.append(f"| `{r['文件']}` | {r['代码行']:,} | {r['总行']:,} |")
    out = Path(args.csv)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=["范围", "语言", "文件", "总行", "代码行", "注释行", "空行", "SHA256"])
        writer.writeheader()
        writer.writerows(rows)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("\n".join(lines_md) + "\n")
    print(f"\n明细 CSV：{out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
