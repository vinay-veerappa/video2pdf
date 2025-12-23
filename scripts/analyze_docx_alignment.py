
import zipfile
import re
import os
import xml.etree.ElementTree as ET

def analyze_docx(docx_path):
    """
    Extracts the sequence of Images and their following Text from a DOCX file.
    """
    with zipfile.ZipFile(docx_path) as z:
        # 1. Read Relationships (to map rId to filenames)
        rels_xml = z.read("word/_rels/document.xml.rels")
        rels_root = ET.fromstring(rels_xml)
        ns_rels = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
        
        # Map: rId -> target (media/image1.png)
        rid_map = {}
        for rel in rels_root.findall('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
            rid = rel.get('Id')
            target = rel.get('Target')
            rid_map[rid] = target

        # 2. Read Document Content
        doc_xml = z.read("word/document.xml")
        doc_root = ET.fromstring(doc_xml)
        
        # Namespaces
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
              'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
              'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
              'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
              
        # Iterate paragraphs
        content_sequence = []
        last_image = None
        
        for p in doc_root.findall('.//w:p', ns):
            # Check for Images
            # Images are usually in w:r/w:drawing/wp:inline/a:graphic/a:graphicData/pic:pic/pic:blipFill/a:blip/@r:embed
            # Simplification: Find any blip and get its r:embed
            # Note: namespace wildcard might be needed or deep traversal
            
            # Simple string check for now to find rId reference? 
            # Better to strictly parse xml, but deep structure is complex.
            # Let's try finding all 'blip' elements in this paragraph
            
            p_xml = ET.tostring(p, encoding='unicode')
            
            # Find r:embed="..."
            # This is a bit hacky but works for standard parsing if xml parsing is tedious
            img_ids = re.findall(r'embed="([a-zA-Z0-9]+)"', p_xml)
            
            for rid in img_ids:
                if rid in rid_map and "media" in rid_map[rid]:
                    img_name = os.path.basename(rid_map[rid])
                    last_image = img_name
                    content_sequence.append({'type': 'image', 'value': img_name})
            
            # Extract Text
            texts = p.findall('.//w:t', ns)
            text_content = "".join([t.text for t in texts if t.text])
            
            if text_content.strip():
                content_sequence.append({'type': 'text', 'value': text_content.strip()[:50] + "...", 'associated_image': last_image})

    return content_sequence

if __name__ == "__main__":
    path = r"C:\Users\vinay\video2pdf\output\Bootcamp Classroom - Week 7 Day 1 - 9_30 trade\Bootcamp Classroom - Week 7 Day 1 - 9_30 trade_final.docx"
    print(f"Analyzing {os.path.basename(path)}...")
    seq = analyze_docx(path)
    
    print("\n--- Document Structure (Full Dump) ---")
    current_img = "None"
    for item in seq:
        if item['type'] == 'image':
             current_img = item['value']
             print(f"\n[IMAGE CHANGE] >>> {current_img}")
        else:
             print(f"  [TXT] ({item['value'][:30]}...) --> Linked to: {current_img}")

