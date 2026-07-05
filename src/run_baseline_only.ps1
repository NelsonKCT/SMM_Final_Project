# run_baseline_only.ps1
# Re-runs only the 6 baseline experiments (one per target country) and saves
# the output of each run to its corresponding .txt log file.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== Starting 6 baseline experiments ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Running baseline - target country: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Run the baseline, echoing output to the screen while also writing the log file
    python run_MultiModalGNN_CrossAttention_CrossCountry.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_baseline_$country.txt"
}

Write-Host "====== All 6 baseline experiments finished! ======" -ForegroundColor Yellow
