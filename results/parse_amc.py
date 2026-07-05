import os
import re

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
base_dir = "c:/Users/minelab/Desktop/projects/ssm"

def read_file(filepath):
    # Try different encodings due to PowerShell redirection UTF-16LE or UTF-8
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
    'dfa': {},
    'amc': {}
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

    # DFA
    dfa_file = os.path.join(base_dir, f"zero-shot_DFA_{country}.txt")
    if os.path.exists(dfa_file):
        content = read_file(dfa_file)
        if content:
            results['dfa'][country] = content

    # AMC
    amc_file = os.path.join(base_dir, f"zero-shot_AMC_{country}.txt")
    if os.path.exists(amc_file):
        content = read_file(amc_file)
        if content:
            results['amc'][country] = content

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

print("\n" + "="*125)
print(f"{'Country':<12} | {'Paper Baseline F1':<18} | {'DFA-GFM F1':<18} | {'AMC-GFM F1 (Ours)':<18} | {'AMC vs Base':<12} | {'AMC vs DFA':<12}")
print("="*125)

for country in countries:
    b_f1 = paper_baseline_f1[country]
    dfa_f1 = "N/A"
    amc_f1 = "N/A"
    
    if country in results['baseline']:
        bm = parse_metrics(results['baseline'][country])
        if bm.get('f1_macro') != "N/A":
            b_f1 = bm.get('f1_macro')
            
    if country in results['dfa']:
        dfam = parse_metrics(results['dfa'][country])
        dfa_f1 = dfam.get('f1_macro')
        
    if country in results['amc']:
        amcm = parse_metrics(results['amc'][country])
        amc_f1 = amcm.get('f1_macro')
        
    imp_base_str = "N/A"
    imp_dfa_str = "N/A"
    try:
        if b_f1 != "N/A" and amc_f1 != "N/A":
            b_val = float(b_f1.split('+-')[0])
            amc_val = float(amc_f1.split('+-')[0])
            diff = amc_val - b_val
            imp_base_str = f"{diff*100:+.2f}%"
    except Exception:
        pass

    try:
        if dfa_f1 != "N/A" and amc_f1 != "N/A":
            dfa_val = float(dfa_f1.split('+-')[0])
            amc_val = float(amc_f1.split('+-')[0])
            diff = amc_val - dfa_val
            imp_dfa_str = f"{diff*100:+.2f}%"
    except Exception:
        pass
        
    print(f"{country:<12} | {b_f1:<18} | {dfa_f1:<18} | {amc_f1:<18} | {imp_base_str:<12} | {imp_dfa_str:<12}")
print("="*125)

print("\n=== Detailed AMC-GFM Metrics ===")
for country in countries:
    if country in results['amc']:
        amcm = parse_metrics(results['amc'][country])
        print(f"{country:<12} -> Acc: {amcm.get('acc')}, Prec: {amcm.get('prec')}, F1-Macro: {amcm.get('f1_macro')}, F1-Micro: {amcm.get('f1_micro')}")
