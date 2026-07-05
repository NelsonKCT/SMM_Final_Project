# run_experiments_dfa.ps1
# Runs the 6 DFA-GFM (decoupled feature alignment) experiments (one per target
# country) and saves the output of each run to its corresponding .txt log file.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== Starting 6 DFA-GFM (decoupled alignment) experiments ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Running DFA-GFM - target country: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Run DFA, echoing output to the screen while also writing the log file
    python run_MultiModalGNN_CrossAttention_CrossCountry_DFA.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_DFA_$country.txt"
}

Write-Host "====== All 6 DFA-GFM experiments finished! ======" -ForegroundColor Yellow
