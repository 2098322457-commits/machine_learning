# 闽江流域南平次日日均流量预报

最终模型在 **2021 年验证集 RMSE 为 88.69 m³/s**，在 **2022–2023 年历史测试集 RMSE 为 91.45 m³/s**。组员初版在相同日期的 RMSE 为 118.94 m³/s，降幅 **23.11%**。[最终版说明](minjiang-v6/README.md) · [历次优化与对照表现](minjiang-v6/优化记录.md)

## 数据与任务

小组任务使用既有的八张气象、水文和站点 CSV，预测南平断面下一天的 GloFAS 模拟日均流量。2025 DCIC 原题考核逐小时入库流量，两者使用不同的标签与评估口径。本仓库的实验分数对应小组定义的日尺度任务。[原始表与时间边界](minjiang-v3/数据质量.md)

官方初赛按未来第 9–24 小时逐时入库流量的 RMSE 排名，得分为 `1/(1+RMSE)`，预测准确度占 100%。决赛另看模型创新性、工程实用性、技术报告和答辩，五项权重依次为准确度 50%、创新性 20%、实用性 15%、报告 10%、答辩 5%。[比赛任务](https://www.digitalchina.gov.cn/2025/cxds/mtbd/202503/t20250303_4983504.htm) · [官方评分规则](https://www.datafountain.cn/competitions/1102/datasets)

## 在新克隆的仓库复现

使用 Python 3.12。在仓库根目录运行以下 PowerShell 命令。Linux 和 macOS 可以把 Python 路径写成 `./.venv/bin/python`。

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r .\minjiang-v3\requirements.txt
& $py .\minjiang-v6\reproduce.py
```

第一次运行会校验固定提交中的数据并生成模型、逐日预测、评估和区间。若用浅克隆，请先运行 `git fetch --unshallow`，让脚本可以读取数据来源提交。汇总成绩随代码保存，逐日输出在本地生成。
