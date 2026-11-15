# W4 最小独立复现包

复制本目录即可使用，不读取原项目、ledger或旧实验。Python依赖见requirements.txt；默认解释器为任务指定路径，可用W4_PYTHON指定已安装依赖的Python。Linux、systemd用户会话可用、约2GiB可用内存。运行：

```bash
bash run.sh
```

输出到本目录新建的output/，拒绝覆盖；再次重放需给未存在的新目录：`bash run.sh /absolute/new/output`。资源限制2GiB、无swap、最多两线程，外层540秒硬限。requirements是本次实测版本；虚拟环境安装命令为 `python -m pip install -r requirements.txt`。BLAS和CPU差异允许浮点末位差异，数学证书使用标准库有理数/384-bit有向舍入，不依赖BLAS。

固定seed及域在seeds.json。重放d32、seed0，从Gaussian键值重新生成真实训练轨迹，F1–F6各一个代表点：F1 mean rank，F2 Adam16代理误差，F3 γ4 Sign错误训练候选方差，F4 e=.1效率和γ16 Adam/Muon比值及标量根，F5 γ16 τ=0/2独立参数噪声MSR，F6 independent GD4的固定/同数据损失匹配旧任务MSR。生成raw.json及六面板representatives.png/pdf，另检查小矩阵稠密排名与正确分数、GD谱闭式，并运行W4_certificate.py重证 `0.050156130219 < e_* < 0.050156130220`。证书原实现来自项目p04_checks/check_l1_l3_l4.py，随包完整携带，未改写原件。moments.json携带原冻结Adam积分，重放不拟合效率。

小规模点用于验证生成与测量流程，不要求低维随机数恰好等于渐近曲线，也不复现所有seed置信区间、P1NUM密网格bracket或全部图中每一条曲线。F4实测比值不验证F.2；F6 MSR不推出rank分布，FT rank预测仍为conjecture。正式W4的原始分数/排名和所有seed统计在上级目录W4_raw.jsonl；S1全分数在W4_full_*.npz/W4_legacy_*.npz。正式注册与报告在项目根W4_PREREG.md、W4_RESULTS.md；重放无需这些文件。

SHA256SUMS验证包源码和配置内容；output/不列入源码清单。预注册sha256可检测修改，不能证明第三方认证的冻结时间。所有输出保留查询索引与原始排名，不把查询当独立数据seed。旧实验的四项失败、EXP2未分辨、FT 316项retrieval等价失败均不因本包改变。
