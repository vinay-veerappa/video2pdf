import json, os, sys
d = json.load(open(r'C:\ICT_Videos\Testing\_triage_summary.json'))
total = {"text_dominant":0,"image_dominant":0,"co_dependent":0,"near_empty":0}
total_pages = 0
print(f"{'PDF':45s} {'pages':>5s}  {'text':>5s} {'image':>5s} {'co-dep':>6s} {'empty':>5s}")
print("-"*80)
for p in d['pdfs']:
    c = p['dominance_counts']
    total_pages += p['pages']
    for k in total: total[k] += c[k]
    name = os.path.basename(p['pdf'])
    print(f"{name:45s} {p['pages']:5d}  {c['text_dominant']:5d} {c['image_dominant']:5d} {c['co_dependent']:6d} {c['near_empty']:5d}")
print("-"*80)
print(f"{'TOTAL':45s} {total_pages:5d}  {total['text_dominant']:5d} {total['image_dominant']:5d} {total['co_dependent']:6d} {total['near_empty']:5d}")
pct = {k: (total[k]*100//total_pages if total_pages else 0) for k in total}
print(f"{'PCT':45s}        {pct['text_dominant']:4d}%  {pct['image_dominant']:4d}%   {pct['co_dependent']:4d}%   {pct['near_empty']:3d}%")
print()
print(f"rendered pages (image+co-dependent): {len(d['rendered_pages'])}")