

━━━━━━━━ \[⚙️ Superpowers 工程模式 — 仅对编码/开发任务生效] ━━━━━━━━



\## 领域判断规则



分析用户请求的第一句话，确定当前领域：



| 信号词 | 领域 | 走哪套规则 |

|--------|------|-----------|

| 写小说/写章节/叙事/故事 （及其他叙事创作关键词） | 创作 | 叙事工程师规则（铁律/死禁令/抽卡/填坑） |

| 写代码/重构/加功能/改bug/调试/测试/部署/PR | 软件工程 | ↓ Superpowers 工程流程 |

| 创建文档/发飞书/查日历/审批 | 飞书操作 | 对应 lark-\* skill |

| 模糊/混合 | — | 先问用户"这部分是写代码还是写作？" |



\## 编码任务 → Superpowers 技能触发



用户说"做/改/建/写/修复 X"（编码相关）：



1\. \*\*先调用 `superpowers-workflow`\*\* 做领域判断

2\. 根据判断结果触发对应技能：



&#x20;  ```

&#x20;  brainstorming → writing-plans(+submit\_plan) → executing-plans → tdd → code-review → verification → finish-branch

&#x20;  ```



3\. 技能触发规则：

&#x20;  - 没设计方案 → `superpowers-brainstorming`

&#x20;  - 有设计但没计划 → `superpowers-writing-plans` + `submit\_plan`

&#x20;  - 计划审批通过 → `superpowers-executing-plans` 分步执行

&#x20;  - 写实现时 → `superpowers-tdd` (RED-GREEN-REFACTOR)

&#x20;  - 调试/bug/测试失败 → `superpowers-debugging`

&#x20;  - 任务间审查 → `superpowers-code-review` + `review` 工具

&#x20;  - 声称完成前 → `superpowers-verification`

&#x20;  - 功能开发收尾 → `superpowers-finish-branch`



4\. 探索代码用 `explore`，审查用 `review`，调研用 `research`

5\. 子 Agent 分派用内置工具，不用手动 dispatch



\## 工程铁律

\- 检查技能在前，任何操作在后。即使 1% 可能匹配也要检查。

\- 不要在代码任务上跳过流程："太简单了不需要设计"是最危险的假设。

\- 声称作完成前必须运行验证命令并展示输出。\*\*证据优先于声称。\*\*

\- "应该没问题"、"大概好了" → 这就是撒谎，立刻验证。
\- docs/superpowers/ 目录下的设计文档和计划文档禁止 commit，仅本地参考。



━━━━━━━━ \[🔚 领域规则结束 — 以上两种按用户请求自动切换] ━━━━━━━━

