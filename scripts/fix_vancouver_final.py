"""Vancouver 重编号最终版（v3，2026-08-28）—— 从干净源 EN v2.0 重建 FINAL
目标顺序（正文首次出现）：1=Jager 2=GBD 3=WHA 4=ZhangL 5=Kovesdy 6=Inker 7=KDIGO
  8=Shi 9=Chu 10=Sabanayagam 11=ZhangK 12=KIDS 13=DeepDKD 14=Holmstrom 15=TRIPOD 16=CDC
当前（EN v2.0）：1=Jager 2=WHA 3=ZhangL 4=Inker 5=KDIGO 6=Shi 7=Chu 8=Sabanayagam
  9=ZhangK 10=KIDS 11=DeepDKD 12=Holmstrom 13=Collins 14=CDC 15=GBD 16=Kovesdy
"""
import re

EN = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/论文/MACKI_Dry_Manuscript_EN_v2.0.md"
FIN = r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行/论文/MACKI_Dry_Manuscript_FINAL_投稿完整版.md"

DIG = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}
def sup(n): return ''.join(DIG[c] for c in str(n))

md = open(EN, encoding='utf-8').read()
head, refs_part = md.split('## References')

# 0) 2.1 患病率引用改 CDC（当前误引 ¹³=TRIPOD；'national estimates¹³' 唯一匹配 2.1）
head = head.replace('national estimates¹³', 'national estimates@@CDC@@')
assert '@@CDC@@' in head

# 1) 保护单位符号
head = head.replace('m²', '@@M2@@').replace('m³', '@@M3@@')
# 2) 保护两位上标 ¹⁰-¹⁶
for n in range(10, 17):
    head = head.replace(sup(n), f'@@T{n}@@')
# 3) 单字符上标占位（防链式）
head = head.replace('¹', '@@A@@')  # Jager
head = head.replace('²', '@@B@@')  # WHA
head = head.replace('³', '@@C@@')  # ZhangL
head = head.replace('⁴', '@@D@@')  # Inker
head = head.replace('⁵', '@@E@@')  # KDIGO
head = head.replace('⁶', '@@F@@')  # Shi
head = head.replace('⁷', '@@G@@')  # Chu
head = head.replace('⁸', '@@H@@')  # Sabanayagam
head = head.replace('⁹', '@@I@@')  # ZhangK
# 4) 还原为目标编号
head = head.replace('@@A@@', sup(1))   # Jager → 1
head = head.replace('@@T15@@', sup(2)) # GBD → 2
head = head.replace('@@B@@', sup(3))   # WHA → 3
head = head.replace('@@C@@', sup(4))   # ZhangL → 4
head = head.replace('@@T16@@', sup(5)) # Kovesdy → 5
head = head.replace('@@D@@', sup(6))   # Inker → 6
head = head.replace('@@E@@', sup(7))   # KDIGO → 7
head = head.replace('@@F@@', sup(8))   # Shi → 8
head = head.replace('@@G@@', sup(9))   # Chu → 9
head = head.replace('@@H@@', sup(10))  # Sabanayagam → 10
head = head.replace('@@I@@', sup(11))  # ZhangK → 11
head = head.replace('@@T10@@', sup(12)) # KIDS → 12
head = head.replace('@@T11@@', sup(13)) # DeepDKD → 13
head = head.replace('@@T12@@', sup(14)) # Holmstrom → 14
head = head.replace('@@CDC@@', sup(15)) # CDC → 15（2.1 节，首次出现早于 TRIPOD）
head = head.replace('@@T13@@', sup(16)) # TRIPOD → 16（4.5 节）
head = head.replace('@@M2@@', 'm²').replace('@@M3@@', 'm³')

# 5) References 重排（目标顺序，文本取自 EN v2.0）；保留 References 之后的尾部（Figures/Supplementary）
ref_lines = re.findall(r'^(\d+)\.\s+(.*)$', refs_part, re.M)
ref_text = {int(n): t for n, t in ref_lines}
ORDER = [1, 15, 2, 3, 16, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13]  # 旧编号 → 目标 1..16（13=TRIPOD→16, 14=CDC→15）
assert len(ref_text) == 16, f"EN v2.0 References 应为 16 条，实际 {len(ref_text)}"
new_refs = "## References\n\n" + "\n".join(f"{i+1}. {ref_text[old]}" for i, old in enumerate(ORDER)) + "\n\n"
# 尾部（References 之后：Figures and tables / Supplementary）—— 2026-08-28 修复：先前版本丢失了图片节
tail = ""
if '## Figures' in refs_part:
    tail = refs_part.split('## Figures', 1)[1]
    tail = '## Figures' + tail

open(FIN, 'w', encoding='utf-8').write(head + new_refs + tail)
print("已写入 FINAL:", FIN)

# 6) 验证（先保护单位符号，避免摘要 m² 被误认为角标）
head_v = head.replace('m²', '@@M2@@').replace('m³', '@@M3@@')
def sup2num(s):
    RM = {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}
    return int(''.join(RM[c] for c in s))
seen = []
for m in re.finditer(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', head_v):
    n = sup2num(m.group(0))
    if n not in seen: seen.append(n)
print('正文首次出现顺序:', seen)
assert seen == list(range(1, 17)), "编号仍不正确！"
assert not re.search(r'(?<![A-Za-z])m¹', head), "单位符号异常"
assert 'm²' in head, "单位符号缺失"
assert 'Sabanayagam¹⁰' in head, "Sabanayagam 应为 10"
assert 'Global Burden of Disease analyses²' in head, "GBD 应为 2"
assert 'priority³' in head, "WHA 应为 3"
assert 'United States⁵' in head, "Kovesdy 应为 5"
assert 'estimates¹⁵' in head, "CDC 应为 15"
assert 'TRIPOD+AI¹⁶' in head, "TRIPOD 应为 16"
# References 校验
fin = open(FIN, encoding='utf-8').read()
r2 = re.findall(r'^(\d+)\.\s+(.*)$', fin.split('## References')[1], re.M)
nums = [int(n) for n, _ in r2]
assert nums == list(range(1, 17)), "References 编号不连续"
print("✅ 全部通过：16 条引用 Vancouver 顺序 1→16，正文/References/单位符号/关键位置全部正确")
