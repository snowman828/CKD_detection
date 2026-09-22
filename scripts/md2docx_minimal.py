# -*- coding: utf-8 -*-
"""md2docx_minimal.py —— 纯标准库 md→docx 转换器（零依赖）
支持：标题(#/##/###) / 段落(加粗**、斜体*) / 表格(|) / 图片(![](path) 或 Figures 段路径行) / 列表(-)
输出：Word 2007+ 兼容 .docx（zip+OOXML）
"""
import re, zipfile, os, shutil, html
from xml.sax.saxutils import escape

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' \
       'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ' \
       'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" ' \
       'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" ' \
       'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'

def para_xml(text, style=None, bold=False, italic=False, align=None):
    pPr = ""
    if style:
        pPr += f'<w:pStyle w:val="{style}"/>'
    if align:
        pPr += f'<w:jc w:val="{align}"/>'
    rPr = ""
    if bold: rPr += "<w:b/>"
    if italic: rPr += "<w:i/>"
    run = f'<w:r><w:rPr>{rPr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    return f'<w:p><w:pPr>{pPr}</w:pPr>{run}</w:p>'

def rich_para_xml(text, style=None):
    """支持 **bold** 和 *italic* 的行内标记"""
    parts = []
    tokens = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for tk in tokens:
        if not tk: continue
        if tk.startswith("**") and tk.endswith("**"):
            parts.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape(tk[2:-2])}</w:t></w:r>')
        elif tk.startswith("*") and tk.endswith("*") and len(tk) > 2:
            parts.append(f'<w:r><w:rPr><w:i/></w:rPr><w:t xml:space="preserve">{escape(tk[1:-1])}</w:t></w:r>')
        else:
            parts.append(f'<w:r><w:t xml:space="preserve">{escape(tk)}</w:t></w:r>')
    pPr = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f'<w:p><w:pPr>{pPr}</w:pPr>{"".join(parts)}</w:p>'

def table_xml(rows):
    """rows: list[list[str]]"""
    border = ('<w:tblBorders>' + "".join(
        f'<w:{k} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        for k in ["top","left","bottom","right","insideH","insideV"]) + '</w:tblBorders>')
    body = ""
    for ri, row in enumerate(rows):
        cells = ""
        for cell in row:
            shd = '<w:shd w:val="clear" w:fill="F2F2F2"/>' if ri == 0 else ""
            cells += (f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/>{shd}</w:tcPr>'
                      f'<w:p><w:r><w:t xml:space="preserve">{escape(cell)}</w:t></w:r></w:p></w:tc>')
        body += f'<w:tr>{cells}</w:tr>'
    return (f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{border}</w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="4000"/><w:gridCol w:w="4000"/><w:gridCol w:w="4000"/>'
            f'<w:gridCol w:w="4000"/><w:gridCol w:w="4000"/></w:tblGrid>{body}</w:tbl>')

def image_xml(rId, width_px, height_px, max_w_emu=6.4e6):
    """嵌入图片（EMU 尺寸，限制最大宽度）"""
    w_emu = int(min(width_px * 9525, max_w_emu))
    h_emu = int(height_px * 9525 * (w_emu / (width_px * 9525)))
    return f'''<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{w_emu}" cy="{h_emu}"/><wp:docPr id="1" name="fig"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="1" name="fig"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rId}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{w_emu}" cy="{h_emu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''

CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Default Extension="jpeg" ContentType="image/jpeg"/>
<Default Extension="jpg" ContentType="image/jpeg"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''

RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

def doc_rels(n):
    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>']
    for i in range(n):
        rels.append(f'<Relationship Id="rIdImg{i+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{i+1}.png"/>')
    rels.append('</Relationships>')
    return "\n".join(rels)

STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="22"/></w:rPr>
<w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
<w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
<w:rPr><w:b/><w:sz w:val="24"/></w:rPr><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>
<w:rPr><w:b/><w:i/><w:sz w:val="22"/></w:rPr><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/>
<w:rPr><w:i/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>'''

def md_to_docx(md_path, out_path, figure_map=None):
    """figure_map: {标题关键词: (png路径,)} —— 匹配 Figure legends 行时插图"""
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()
    body = []
    media = []          # [(rId, path, w, h)]
    rId_counter = [2]
    i = 0
    pending_caption = None

    def add_figure(path):
        from PIL import Image
        im = Image.open(path)
        w, h = im.size
        rId = f"rIdImg{rId_counter[0]}"
        rId_counter[0] += 1
        media.append((rId, path))
        body.append(image_xml(rId, w, h))
        return f"rIdImg{rId_counter[0]-1}"

    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()
        # 图片行：- **Figure N** ... `path.png` 或 ![](path)
        fig_match = re.match(r"^- \*\*(Figure \d+|Graphical Abstract)\*\*(.*?)\*`([^`]+\.(?:png|jpg|jpeg))`\*", stripped)
        m_img = re.match(r"!\[(.*?)\]\(([^)]+)\)", stripped)
        if fig_match:
            title, desc, path = fig_match.group(1), fig_match.group(2), fig_match.group(3)
            full = os.path.join(os.path.dirname(md_path), path) if not os.path.isabs(path) else path
            if not os.path.exists(full):
                # 尝试相对 09 目录 / 项目根
                for base in [r"G:/项目文件/CKD多模态AI早诊_本项目专属归档/09_无湿实验执行",
                             r"G:/项目文件/CKD多模态AI早诊_本项目专属归档"]:
                    alt = os.path.join(base, path)
                    if os.path.exists(alt):
                        full = alt; break
            if os.path.exists(full):
                from PIL import Image
                im = Image.open(full); w, h = im.size
                rId = f"rIdImg{rId_counter[0]}"; rId_counter[0] += 1
                media.append((rId, full))
                body.append(image_xml(rId, w, h))
                body.append(rich_para_xml(f"{title}{desc}", style="Caption"))
            i += 1; continue
        if m_img:
            path = m_img.group(2)
            full = os.path.join(os.path.dirname(md_path), path) if not os.path.isabs(path) else path
            if os.path.exists(full):
                from PIL import Image
                im = Image.open(full); w, h = im.size
                rId = f"rIdImg{rId_counter[0]}"; rId_counter[0] += 1
                media.append((rId, full))
                body.append(image_xml(rId, w, h))
            i += 1; continue
        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i+1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = [header]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body.append(table_xml(rows))
            continue
        # 标题
        hm = re.match(r"^(#{1,3})\s+(.*)", stripped)
        if hm:
            lvl, txt = len(hm.group(1)), hm.group(2).strip()
            body.append(rich_para_xml(txt, style=f"Heading{lvl}"))
            i += 1; continue
        # 列表
        if stripped.startswith("- ") or stripped.startswith("* "):
            body.append(rich_para_xml("• " + stripped[2:]))
            i += 1; continue
        # 引用块/分隔线/空行
        if stripped.startswith(">"):
            body.append(rich_para_xml(stripped.lstrip("> "), style="Caption"))
            i += 1; continue
        if stripped in ("---", "***") or not stripped:
            i += 1; continue
        # 普通段落
        body.append(rich_para_xml(stripped))
        i += 1

    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:document {W_NS}><w:body>{"".join(body)}'
                f'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
                f'</w:body></w:document>')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/_rels/document.xml.rels", doc_rels(len(media)))
        for idx, (rId, path) in enumerate(media):
            z.write(path, f"word/media/image{idx+1}.png")
    return len(media)

if __name__ == "__main__":
    import sys
    md = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else md.replace(".md", ".docx")
    n = md_to_docx(md, out)
    print(f"OK: {out} | 嵌入图片 {n} 张 | {os.path.getsize(out)/1024:.0f} KB")
