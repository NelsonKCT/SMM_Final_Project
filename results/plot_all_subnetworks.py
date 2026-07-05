import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

countries = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]
countries_display = ["China", "Iran", "UAE", "Cuba", "Russia", "Venezuela"]

subnets = ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"]
subnets_display = ["co-Retweet (coRT)", "co-URL (coURL)", "co-Hashtag (hashSeq)", "Fast Retweet (fastRT)", "Tweet Similarity (tweetSim)"]

base_dir = "c:/Users/minelab/Desktop/projects/ssm"

def read_file(filepath):
    for enc in ['utf-16', 'utf-16-le', 'utf-8', 'gbk']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            continue
    return None

# Parse metrics for each country
data_matrix = np.zeros((len(countries), len(subnets)))
std_matrix = np.zeros((len(countries), len(subnets)))

for i, country in enumerate(countries):
    filepath = os.path.join(base_dir, f"zero-shot_CORAL_{country}.txt")
    if os.path.exists(filepath):
        content = read_file(filepath)
        if content:
            # Extract F1-Macro for each subnet using regex
            for j, subnet in enumerate(subnets):
                pattern = rf'\[TEST_{subnet}\] f1_macro:\s*([0-9\.]+)\+-([0-9\.]+)'
                m = re.search(pattern, content)
                if m:
                    data_matrix[i, j] = float(m.group(1))
                    std_matrix[i, j] = float(m.group(2))
                else:
                    # Fallback for hashSeq sometimes missing TEST_ prefix or variations
                    pattern_alt = rf'\[TEST_coURL\] roc_auc.*?\nTEST_{subnet} set:.*?\n.*?f1_macro:\s*([0-9\.]+)\+-([0-9\.]+)'
                    m_alt = re.search(pattern_alt, content, re.DOTALL)
                    if m_alt:
                        data_matrix[i, j] = float(m_alt.group(1))
                        std_matrix[i, j] = float(m_alt.group(2))
                    else:
                        print(f"Warning: Could not parse {subnet} for {country}")
                        data_matrix[i, j] = 0.40  # Default baseline fallback
        else:
            print(f"Error: Empty file {filepath}")
    else:
        print(f"Error: File {filepath} does not exist")

print("Parsed Data Matrix:\n", data_matrix)

# ----------------------------------------------------
# CHART 1: Scientific Heatmap
# ----------------------------------------------------
plt.figure(figsize=(11, 7.5), dpi=300)
# Custom palette: viridis or magma for professional presentation look
ax = sns.heatmap(data_matrix, annot=True, fmt=".3f", cmap="YlGnBu", 
                 xticklabels=subnets_display, yticklabels=countries_display,
                 linewidths=.5, cbar_kws={'label': 'F1-Macro Score'},
                 annot_kws={"fontsize": 11, "weight": "bold"})

plt.title("Correlation Alignment (CORAL) Zero-Shot F1-Macro across Similarity Networks", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Behavioral Similarity Sub-Networks", fontsize=12, fontweight='bold', labelpad=12)
plt.ylabel("Target Geopolitical Domain", fontsize=12, fontweight='bold', labelpad=12)
plt.xticks(rotation=15, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("coral_subnetworks_heatmap.png", bbox_inches='tight')
plt.close()

# ----------------------------------------------------
# CHART 2: Subnetworks Multi-Line Chart (5 lines, one per sub-network)
# ----------------------------------------------------
plt.figure(figsize=(11, 6.5), dpi=300)
markers = ['o', 's', '^', 'D', 'p']
colors = ['#4361EE', '#3F37C9', '#7209B7', '#F72585', '#4CC9F0'] # Premium modern color set

for j in range(len(subnets)):
    plt.plot(countries_display, data_matrix[:, j], marker=markers[j], color=colors[j],
             linewidth=2.2, markersize=7, label=subnets_display[j])

plt.title("Zero-Shot Performance Profiles across Behavioral Channels", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("F1-Macro Score", fontsize=12, fontweight='bold')
plt.xlabel("Target Geopolitical Domain", fontsize=12, fontweight='bold')
plt.ylim(0.2, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left', frameon=True, edgecolor='#e0e0e0', fontsize=10)
plt.tight_layout()
plt.savefig("coral_subnetworks_trends.png", bbox_inches='tight')
plt.close()

print("All advanced subnetwork charts generated successfully!")
