# NetOps Fault Lab

用路径探测定位可疑链路，并决定下一条该测哪里。

[![CI](https://github.com/chrischen-coder/netops-fault-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/chrischen-coder/netops-fault-lab/actions/workflows/ci.yml)
[Python 3.10+](pyproject.toml) · [MIT 许可证](LICENSE)

[English](README.md) · [算法](docs/method.zh-CN.md) · [实验](docs/experiments.md) · [输入格式](docs/input.md)

两条到数据库的探测都失败，到缓存的探测正常。故障可能在核心链路，也可能在数据库接入链路。
只看失败次数，两条链路无法区分。继续收集同样路径上的告警，也未必能定位问题。

NetOps Fault Lab 根据已知路径和探测结果给出候选排名，指出当前无法区分的链路，再推荐最值得补测的路径。
输入为结构化 JSON；输出为 CLI、JSON 或离线 HTML 报告。

![实际运行报告：两个候选无法区分，建议补测 r1 到 r2](docs/assets/report.png)

上图来自仓库的合成样例。补入一次 `r1 → r2` 失败结果后，核心链路的模型概率从 49.8% 变为 96.7%。
这次结果由样例预设，用于演示推断过程，没有向真实网络发送探测。[补测后的报告截图](docs/assets/after-probe.png)。

## 快速开始

Python 3.10+：

```bash
git clone https://github.com/chrischen-coder/netops-fault-lab.git
cd netops-fault-lab
python -m venv .venv
```

macOS/Linux：`source .venv/bin/activate`。Windows PowerShell：`.venv\Scripts\Activate.ps1`。

```bash
python -m pip install .
netops-fault-lab diagnose examples/ambiguous.json
netops-fault-lab diagnose examples/after-probe.json
netops-fault-lab diagnose examples/ambiguous.json --format html --output report.html
```

第一条命令的关键输出：

```text
Status: ambiguous
core                 0.498
database             0.498
Indistinguishable: core, database
Next probe: probe-core (r1 -> r2); expected gain 0.665 bits
```

打开 `report.html` 查看路径、排名和证据。文件必须尚不存在。核心推断只依赖 Python 标准库。
未传入 `--model` 时使用演示参数，不能把输出概率直接当成真实网络的正确率。

## 解决哪几个问题

| 问题 | 当前提供的能力 |
| --- | --- |
| 一条链路故障引发多条路径告警，单看失败次数无法定位 | 同时利用失败和成功探测，按贝叶斯后验排序 |
| 两个故障位置会产生相同观测，算法仍给出唯一答案 | 检查路径签名，把无法区分的候选一起列出 |
| 探测有成本，不知道下一步该查哪里 | 按期望信息增益／成本给候选探测排序 |
| 论文或方案只报告平均准确率，看不出何时失效 | 分别测试噪声变化、55% 漏采和重复告警，保存逐例结果 |

## 算法与训练

假设一个时间窗口内至多有一条故障链路。对每个候选计算：

```text
后验概率 ∝ 先验概率 × 每条探测结果在该故障下出现的概率
下一条探测 = 期望减少的不确定性 / 探测成本 最大的候选
```

命中故障路径的失败率、背景失败率和健康先验可以从带标签的数据估计，采用 Beta(1,1) 平滑。
基线包含失败路径投票、训练过的逻辑回归和贝叶斯推断。它们使用相同训练／测试划分。

采用的是已有的统计方法。项目的工作重点是可复现的定位与补测实验、无法区分的情况和失败分析。
[公式、假设和相关研究](docs/method.zh-CN.md)。

## 已测结果

3 个种子；每个种子使用 24 个训练拓扑、24 个不同的测试拓扑，每个拓扑 12 个事件。
每个条件共有 864 个测试事件，全部由生成器构造。

| 条件 | 失败路径投票 Top-1 | 逻辑回归 Top-1 | 贝叶斯 Top-1 |
| --- | ---: | ---: | ---: |
| 训练与测试噪声相同 | 56.6% | 72.5% | 72.3% |
| 背景失败增加、故障信号减弱 | 24.4% | 37.7% | 37.2% |
| 55% 探测漏采 | 40.6% | 49.3% | 49.7% |
| 同一批告警复制 4 份 | 56.6% | 72.6% | 72.1% |

贝叶斯没有明显胜过逻辑回归。另一个实验从 3 条初始探测记录出发，允许最多 4 次补测：
**信息增益选择达到 72.5%，随机选择为 57.2%**。两者共享候选池和预先采样的探测结果。
观察到的差值为 15.3 个百分点，按拓扑配对 bootstrap 的 95% 区间为 11.8–18.5 个百分点。

![原始实验结果生成的定位与主动探测对照图](docs/assets/benchmark.png)

也要看置信度：重复告警后，被模型接受的结果中，错误比例从 2.6% 升至 12.6%；
噪声变化时达到 38.2%。`0.9` 阈值不能保证 90% 正确率。

这些实验用于检查方法行为，不能推导生产网准确率或故障恢复时间。
[数据、逐例预测、学习参数与测量口径](docs/experiments.md)。

## 复现实验

```bash
python -m pip install '.[bench]'
netops-fault-lab benchmark --output runs/local
netops-fault-lab fit benchmarks/results/v0.1/train-7.jsonl.gz --output fitted-model.json
netops-fault-lab diagnose examples/ambiguous.json --model fitted-model.json
python -m unittest discover -s tests -v
```

输出目录需尚不存在。实验会保存数据、拓扑划分、学习参数、逐例预测和主动探测记录。

## 适用范围

当前适用于已知静态路径上的单链路故障或健康状态，假设探测结果在给定故障下相互独立。
路由变化、多处同时故障、相关告警、应用层根因和参数漂移，需要另外建模。
同一路径上的健康与故障样本也可能偶然产生相同结果，后验排名不是故障证明。

首版完成的是离线算法与实验工具。后续优先接入可控网络实验和真实探测，再验证相关噪声、多故障模型；
[分阶段计划](docs/roadmap.zh-CN.md)。

[GPU 环境与训练自检](experiments/gpu/README.zh-CN.md)已在 8 张 RTX 5090 上验证：
DDP 梯度同步、Qwen3-VL 图文推理和单卡 LoRA 更新均已跑通。这是环境验证，尚未进行大模型故障定位评测。

[贡献指南](CONTRIBUTING.md) · [版本记录](CHANGELOG.md) · [软件引用](CITATION.cff)

MIT 许可证。示例和实验数据均自行生成。
