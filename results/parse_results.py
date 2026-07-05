import os
import re

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
base_dir = os.path.dirname(os.path.abspath(__file__))  # the results/ directory

def read_file(filepath):
    # Try different encodings due to PowerShell Tee-Object default to UTF-16LE
    for enc in ['utf-16', 'utf-16-le', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            continue
    return None

results = {
    'baseline': {},
    'dann': {},
    'coral': {},
    'dfa': {}
}

# Official numbers from the IOHunter paper (Table 3, Only PreTrain F1-Macro)
paper_baseline_f1 = {
    'china': '0.5814+-0.0589',
    'iran': '0.7278+-0.0143',
    'UAE': '0.8393+-0.0593',
    'cuba': '0.8991+-0.0535',
    'russia': '0.7977+-0.0193',
    'venezuela': '0.9099+-0.0107'
}

for country in countries:
    # Baseline
    b_file = os.path.join(base_dir, f"zero-shot_baseline_{country}.txt")
    if os.path.exists(b_file):
        content = read_file(b_file)
        if content:
            results['baseline'][country] = content

    # DANN
    d_file = os.path.join(base_dir, f"zero-shot_DANN_{country}.txt")
    if os.path.exists(d_file):
        content = read_file(d_file)
        if content:
            results['dann'][country] = content

    # CORAL
    c_file = os.path.join(base_dir, f"zero-shot_CORAL_{country}.txt")
    if os.path.exists(c_file):
        content = read_file(c_file)
        if content:
            results['coral'][country] = content

    # DFA
    dfa_file = os.path.join(base_dir, f"zero-shot_DFA_{country}.txt")
    if os.path.exists(dfa_file):
        content = read_file(dfa_file)
        if content:
            results['dfa'][country] = content

# Parser metrics
def parse_metrics(content):
    metrics = {}
    patterns = {
        'acc': r'\[TEST\] accuracy:\s*([0-9\.\+-]+)',
        'prec': r'\[TEST\] precision:\s*([0-9\.\+-]+)',
        'f1_macro': r'\[TEST\] f1_macro:\s*([0-9\.\+-]+)',
        'f1_micro': r'\[TEST\] f1_micro:\s*([0-9\.\+-]+)',
    }
    
    for name, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            metrics[name] = m.group(1)
        else:
            metrics[name] = "N/A"
    return metrics

print("\n" + "="*115)
print(f"{'Country':<12} | {'Paper Baseline F1':<18} | {'DANN F1':<18} | {'CORAL F1':<18} | {'DFA-GFM F1 (Ours)':<18} | {'DFA vs Base':<12}")
print("="*115)

for country in countries:
    b_f1 = paper_baseline_f1[country]
    d_f1 = "N/A"
    c_f1 = "N/A"
    dfa_f1 = "N/A"
    
    if country in results['baseline']:
        bm = parse_metrics(results['baseline'][country])
        if bm.get('f1_macro') != "N/A":
            b_f1 = bm.get('f1_macro')
            
    if country in results['dann']:
        dm = parse_metrics(results['dann'][country])
        d_f1 = dm.get('f1_macro')
        
    if country in results['coral']:
        cm = parse_metrics(results['coral'][country])
        c_f1 = cm.get('f1_macro')
        
    if country in results['dfa']:
        dfam = parse_metrics(results['dfa'][country])
        dfa_f1 = dfam.get('f1_macro')
        
    imp_str = "N/A"
    try:
        if b_f1 != "N/A" and dfa_f1 != "N/A":
            b_val = float(b_f1.split('+-')[0])
            dfa_val = float(dfa_f1.split('+-')[0])
            diff = dfa_val - b_val
            imp_str = f"{diff*100:+.2f}%"
    except Exception:
        pass
        
    print(f"{country:<12} | {b_f1:<18} | {d_f1:<18} | {c_f1:<18} | {dfa_f1:<18} | {imp_str:<12}")
print("="*115)

print("\n=== Detailed DFA-GFM Metrics ===")
for country in countries:
    if country in results['dfa']:
        dm = parse_metrics(results['dfa'][country])
        print(f"{country:<12} -> Acc: {dm.get('acc')}, Prec: {dm.get('prec')}, F1-Macro: {dm.get('f1_macro')}, F1-Micro: {dm.get('f1_micro')}")
