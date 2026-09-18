#!/usr/bin/env python3
"""将课程中文原文同步到官网；--check 只检查，不写入文件。"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs/tutorial.md"
DEFAULT_SITE_ROOT = Path("/Users/jackyansongli/jackyansongli.github.io")
TARGET = Path("docs/content/docs/zh/subagent-tutorial.md")
REPOSITORY_URL = "https://github.com/JackYansongLi/learn-harness"
DESCRIPTION = (
    "以 AlexNet 论文为例，通过调用现成 Subagent 和实现复现难度 Subagent 两道练习，"
    "构建论文复现调查 Agent。"
)


def render_page(source: str) -> str:
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            if not title:
                raise ValueError("教程的第一个 H1 标题不能为空")
            body = "".join(lines[:index] + lines[index + 1 :])
            break
    else:
        raise ValueError("教程中找不到 H1 标题")

    body = body.replace(
        "../examples/alexnet/paper.pdf",
        f"{REPOSITORY_URL}/blob/main/examples/alexnet/paper.pdf",
    )
    frontmatter = (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"description: {json.dumps(DESCRIPTION, ensure_ascii=False)}\n"
        "prev: false\n"
        "next: false\n"
        "head:\n"
        "  - tag: script\n"
        "    attrs:\n"
        "      type: module\n"
        "      src: /scripts/subagent-diagrams.js\n"
        "  - tag: style\n"
        "    content: '.sl-markdown-content blockquote :is(th, td) { min-width: 7rem; } "
        ".sl-markdown-content > table :is(th, td) { min-width: 9rem; }'\n"
        "---\n\n"
    )
    links = (
        f"课程仓库：[learn-harness]({REPOSITORY_URL})；"
        f"源码教程：[docs/tutorial.md]({REPOSITORY_URL}/blob/main/docs/tutorial.md)。\n"
    )
    return frontmatter + links + body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root", type=Path, default=DEFAULT_SITE_ROOT, help="官网仓库根目录"
    )
    parser.add_argument("--check", action="store_true", help="内容相同返回 0，不同返回 1；不写入")
    args = parser.parse_args(argv)
    target = args.site_root.expanduser().resolve() / TARGET
    try:
        expected = render_page(SOURCE.read_bytes().decode("utf-8")).encode("utf-8")
        current = target.read_bytes() if target.exists() else None
        if current == expected:
            print(f"已同步：{target}")
            return 0
        if args.check:
            print(f"待同步：{target}")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(expected)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"同步失败：{exc}\n")
    print(f"已更新：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
