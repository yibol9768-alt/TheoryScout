"""Repair inline math delimiters in the W8 draft only."""
import re
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'W8_FINITE_D.md'
text=p.read_text()
text=text.replace('未从项目外读取源码','没有导入其他项目源码')
text=text.replace('第二个核查脚本的时间、内存另见 '+chr(96)+'W8_verification.json'+chr(96)+'。所有计算阶段总计远低于 20 分钟。',
    '第二个核查脚本用时 8.13 秒，cgroup memory.peak=83894272 字节，ru_maxrss=105.86 MiB。两个数值阶段共 56.87 秒。')
old='具体结果见 '+chr(96)+'W8_verification.json'+chr(96)+'；集中结论依靠 §2–3 的证明，不能从有限数据集 SD 的几档变化推断速率。'
new='极限 recall 为 .314311。\n\n| d | 数据集平均 recall | 数据集 SD |\n|---:|---:|---:|\n| 32 | .305969 | .009026 |\n| 64 | .308578 | .005366 |\n| 96 | .311517 | .003140 |\n\n这几档均值仍有负偏。集中结论依靠 §2–3 的证明，不能从有限数据集 SD 的变化推断速率；§4 的方差修正也没有自动给出该 recall 偏差的系数。\n\n五个 Sign 精确矩核查的最大绝对标准化误差为 0.781 SE，GD→Sign 条件均值核查为 −0.699 SE，全部通过上述诊断。'
text=text.replace(old,new)
lines=text.splitlines(keepends=True)
out=[]
display=False
code=False
for line in lines:
    st=line.strip()
    if st.startswith(chr(96)*3):
        code=not code
        out.append(line)
        continue
    if st==r'\[':
        display=True
        out.append(line)
        continue
    if st==r'\]':
        display=False
        out.append(line)
        continue
    if code or display:
        out.append(line)
        continue
    result=''
    i=0
    while i<len(line):
        if line[i]!='(':
            result+=line[i]
            i+=1
            continue
        j=i+1
        depth=1
        while j<len(line) and depth:
            if line[j]=='(': depth+=1
            if line[j]==')': depth-=1
            j+=1
        if depth:
            result+=line[i:]
            break
        inside=line[i+1:j-1]
        mathlike=bool(re.search(r'[\\_^=<>/]',inside) or re.fullmatch(r'[A-Za-z](?:,[A-Za-z])*',inside))
        if mathlike and not re.search(r'[\u4e00-\u9fff]',inside):
            result+='$'+inside+'$'
        else:
            result+=line[i:j]
        i=j
    out.append(result)
p.write_text(''.join(out))
