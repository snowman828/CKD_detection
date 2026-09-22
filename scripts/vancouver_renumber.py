"""Vancouver 引用重编号（2026-08-28）
规则：正文角标按首次出现顺序编号 1,2,3...；参考文献列表同步重排。
修复：GBD/Kovesdy 补入引言首段但未重排导致的乱序（2,1,15,3,16,...）
"""
import re, sys

SRC = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/论文/MACKI_Dry_Manuscript_FINAL_投稿完整版.md"

SUP = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}
REV = {v:k for k,v in SUP.items()}

def sup2num(s): return int(''.join(SUP[c] for c in s))
def num2sup(n):
    return ''.join(REV[c] for c in str(n))

md = open(SRC, encoding='utf-8').read()
head, refs_section = md.split('## References')
refs_part = '## References' + refs_section

# 0) 保护单位符号（m² 平方米）——防止被误判为引用角标（2026-08-28 教训：m²→m¹ 误伤）
head = head.replace('m²', '@@M2@@').replace('m³', '@@M3@@')

# 1) 扫描正文角标首次出现顺序
seen = []
for m in re.finditer(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', head):
    n = sup2num(m.group(0))
    if n not in seen:
        seen.append(n)
print('旧编号首次出现顺序:', seen)

# 2) 构建映射 old -> new
mapping = {old: i+1 for i, old in enumerate(seen)}
print('映射:', mapping)

# 3) 替换正文角标
def repl(m):
    old = sup2num(m.group(0))
    return num2sup(mapping[old])
new_head = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', repl, head)
# 恢复单位符号
new_head = new_head.replace('@@M2@@', 'm²').replace('@@M3@@', 'm³')

# 4) 重排参考文献列表（解析 "N. text" 行）
ref_lines = re.findall(r'^(\d+)\.\s+(.*)$', refs_part, re.M)
ref_texts = {int(n): t for n, t in ref_lines}
print('参考文献条目数:', len(ref_texts))
ordered = []
for new_n in range(1, len(seen)+1):
    old_n = [k for k, v in mapping.items() if v == new_n][0]
    ordered.append(f"{new_n}. {ref_texts[old_n]}")
new_refs = '## References\n\n' + '\n'.join(ordered) + '\n'

# 5) 输出
out = new_head + new_refs
open(SRC, 'w', encoding='utf-8').write(out)
print('已写入:', SRC)

# 6) 验证：重扫新正文
new_seen = []
for m in re.finditer(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', new_head):
    n = sup2num(m.group(0))
    if n not in new_seen:
        new_seen.append(n)
print('新编号首次出现顺序:', new_seen)
assert new_seen == list(range(1, len(seen)+1)), "重编号失败：仍乱序"
print('✅ Vancouver 顺序校验通过：', len(seen), '条引用按首次出现顺序 1→', len(seen))
