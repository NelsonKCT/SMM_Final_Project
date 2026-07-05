# run_experiments.ps1
# Runs the 6 baseline and 6 DANN experiments (one per target country) and
# saves the output of each run to its corresponding .txt log file.

$countries = @("china", "iran", "UAE", "cuba", "russia", "venezuela")

# 1. Run the 6 baseline experiments
Write-Host "====== Starting 6 baseline experiments ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Running baseline - target country: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Run the baseline, echoing output to the screen while also writing the log file
    python run_MultiModalGNN_CrossAttention_CrossCountry.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_baseline_$country.txt"
}

# 2. Run the 6 DANN experiments
Write-Host "====== Starting 6 DANN experiments ======" -ForegroundColor Green
foreach ($country in $countries) {
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Running DANN - target country: $country ..." -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Run DANN, echoing output to the screen while also writing the log file
    python run_MultiModalGNN_CrossAttention_CrossCountry_DANN.py --dataset $country --device 0 --epochs 1000 --splits 5 | Tee-Object -FilePath "../../zero-shot_DANN_$country.txt"
}

Write-Host "====== All 12 experiments finished! ======" -ForegroundColor Yellow
