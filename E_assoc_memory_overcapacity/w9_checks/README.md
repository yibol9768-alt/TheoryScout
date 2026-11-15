# W9 复现说明

理论见根目录 W9_EXTENSION.md。本目录只用 NumPy/SciPy，不调用 LLM API。

- run.py：三维度、五种数据情形、五种实际算法；formal_rows.jsonl 为原始 600 个算法端点，formal_metadata.json 记录资源。
- report.py：按数据集种子汇总、配对差、9 项闭式常数核验；输出 RESULTS.md、summary.csv、paired_contrasts.json、closed_form_checks.json。
- focus.py：独立种子定向核查 Sign 首步的同边际效率/同谱不同基反例；focus_rows.jsonl、focus_summary.json、FOCUS.md。其追加设计不是预注册。
- audit.py：核对输入文档散列、代码语法、原始端点完整性、数学环境配对，以及 (W9.26) 在对称和非对称二点分布上的精确有理数恒等式；输出 audit.json。
- smoke_*：正式计算前的单数据集程序核查，不用于统计结论。

运行命令（从项目根目录）：

    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
      env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
      /home/liuyibo/venvs/theory/bin/python w9_checks/run.py

随后用同一资源前缀运行 report.py 和 focus.py。为防止覆盖原始结果，原始 JSONL/汇总 CSV 使用排他创建模式。复跑请使用下述新输出目录机制，不删除或覆盖交付数据：

    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
      env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
      W9_OUTPUT_DIR=w9_checks/repro_01 \
      /home/liuyibo/venvs/theory/bin/python w9_checks/run.py

对 report.py、focus.py 使用同一个 W9_OUTPUT_DIR。每次复跑另选新子目录。

理论归一化：用预测 Q 归一化全部候选分数；排序不受这个正比例影响。错误候选排除正确对角；高斯数据中的 ties 概率为零，广泛检查的非高斯探索端点若有 ties 则用并列中点作数值诊断。

区间以独立数据集种子为单位，不能把查询数乘种子数当独立样本量。首轮 CDF 指标是固定五阈值、汇总查询诊断；定向核查另在每集固定的 32 个查询上计算完整错误候选的逐查询 sup-CDF 差。均不构成渐近定理的有限维概率保证。

程序只固定有限步、缩放的学习率/Adam stabilizer；没有测试固定物理 epsilon、归一化值或 top-1 容量。值对角各向异性的 §5 推论未额外数值核查。
