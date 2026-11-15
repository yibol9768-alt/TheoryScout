审查 21 独立核验；不导入、执行或复制 `aa3_checks/` 或以往审查的实现，不调用 LLM API，不运行预注册有限维实验。

所有 Python 命令使用相同前缀：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -u review21_checks/records_numeric.py
```

将最后的程序依次换成 `decimal_enclosures.py`、`algebra_and_tables.py`、`design_reference.py`、`finish_audit.py`。输出采用 exclusive-create；重放应为输出选择本目录内的新名字，不覆盖已保存结果。`finish_audit.py` 应在其他计算和报告完成后运行。

- `records_numeric.py`：独立 Fraction 扫描二维切片及一维覆盖，检查整数端点、差值、根前单调性记录、来源散列；独立从未校正 m/v 原始两步 Adam 递推求解匹配根，以复步法传播导数，复算功效及其敏感性。
- `decimal_enclosures.py`：新写的 Decimal 48 位有向区间及通用多项式算术；独立算 Machin π、半正态密度和带余项积分，重新认证四个完整历史矩形、右端存在性覆盖和六条近阈值首次根。只从证书读取参数盒，不读取被审积分包围。生成器使用的闭式量由原始残差和偏差修正的矩递推独立推导；没有复用被审区间算术。
- `algebra_and_tables.py`：六项符号恒等式、19 项显示十进制外向放宽核验、独立算术原语对精确分数的检查。
- `design_reference.py`：在原始端点实现上对独立 H、G 做确定性联合求积，复算独立坐标影响函数方差参照；不是高维训练数据仿真。
- `finish_audit.py`：结束输入完整性、独立细分的精确覆盖、产物散列与资源汇总。

`decimal_enclosures.log`、`decimal_enclosures_run2.log` 是保守区间不能直接认证旧单元/右端时的停止记录，不是发现反向符号。`decimal_enclosures_run3.log` 是过宽的原始残差商表达式的主动终止记录。最终实现先做精确代数消去相同 β 项，再用全新后端认证，仍可按需细分；有效完整结果以最终 `decimal_enclosures.json` 和 `decimal_enclosures_run4.log` 为准。以上日志保留，不覆盖、不冒充完成证书。

`input_hashes.json` 在本次独立数值计算前生成；`input_integrity.json` 在结束时检查。正式审查结论见上级目录 `PROOF_REVIEW21.md`。
