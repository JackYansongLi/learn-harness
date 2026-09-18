# 协作约定

- `docs/tutorial.md` 是教程的权威源。官网保留中文原文，不翻译，不修改或生成英文页面。
- 每次修改教程后，在本仓库运行 `uv run python scripts/sync_tutorial.py`，再运行 `uv run python scripts/sync_tutorial.py --check`。脚本默认同步到 `/Users/jackyansongli/jackyansongli.github.io/docs/content/docs/zh/subagent-tutorial.md`；官网仓库换位置时用 `--site-root` 指定根目录。
- `docs/diagrams/` 中的教程配图由同一脚本复制并核对；修改流程图源码时，也要更新对应图片，不要单独修改官网副本。
- 同步后，按官网仓库的现有项目命令构建中文官网并检查页面，再同步提交、推送课程和官网两个仓库。网站根目录的 `AGENTS.md` 也应记录这项同步约定。
- `solution/` 仅保留在本地并继续忽略，不追踪、不提交教师答案。
