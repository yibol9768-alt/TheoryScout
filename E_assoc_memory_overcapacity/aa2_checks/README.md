# AA2 数值核验

理论主文件：[`../AA2_EXT_RETRIEVAL.md`](../AA2_EXT_RETRIEVAL.md)。全部数值为诊断，不是证明或严格区间证书。

- `cue_check.py`：主目标“弱线索查询、全部训练候选、固定比例名单”的模拟；计算 value leverage、分数/CDF 距离和召回，以及三个理论交点。
- `cue_results.json`：d=64、128、256；每维六个数据集、2048 次查询；五个优化器方向。
- `cue_large_results.json`：同配置，d=512、1024；为了检查近交点的有限维偏差而追加。
- `retrieval_check.py`：确定性积分/交点核验及主文件第 6 节的强参数噪声检索模拟。
- `retrieval_results.json`：辅助目标的全部原始记录、汇总和输入前后哈希。
- `audit_artifacts.py`：只读检查记录数、交点代数关系、Python 语法、所有已记录输入的哈希；产生新的审计清单。
- `audit_results.json`：最终输入/产出 SHA-256 清单和审计结果；不把数值诊断认证为数学证明。

所有计算均使用：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python aa2_checks/audit_artifacts.py \
  --output aa2_checks/audit_NEW.json
```

其他复跑命令见主文件。输出必须使用不存在的新路径，程序拒绝覆盖。
标准误以独立数据集为单位。近交点的有限维排序尚无误差保证，d=1024 的 e=.1 样本也不能可靠验证其很小的极限优势。
