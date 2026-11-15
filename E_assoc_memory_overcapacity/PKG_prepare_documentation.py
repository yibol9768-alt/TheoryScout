from pathlib import Path
p=Path('supplement/README.md');s=p.read_text()
s=s.replace('|完整实验重跑设计与规模检查|', '|从查询级 NPZ 独立重建测量值并重走表格管线（小规模不作确认）|`python reproduce.py query-tables --output recomputed/query_tables`|约 2 秒|独立读取 30 个查询归档、生成 300 个测量值；统计管线通过，确认端点保持不合格|\n|完整实验重跑设计与规模检查|')
s=s.replace('表格命令生成 MR 主表', '`measurements_from_queries.py` 可从重放或完整重跑的 NPZ 独立提取测量值；`query-tables` 演示从随包代表性查询归档开始的完整管线。表格命令生成 MR 主表')
p.write_text(s)
