# run_experiments_coral.ps1
# Runs the 6 Deep CORAL experiments (one per target country) and saves the
# output of each run to its corresponding .txt log file.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

Write-Host "====== Starting 6 Deep CORAL experiments ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Running Deep CORAL - target country: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Run CORAL, echoing output to the screen while also writing the log file
    python run_MultiModalGNN_CrossAttention_CrossCountry_CORAL.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_CORAL_$country.txt"
}

Write-Host "====== All 6 Deep CORAL experiments finished! ======" -ForegroundColor Yellow
