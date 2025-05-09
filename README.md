# Email A/B Testing Project

Welcome to the **Email A/B Testing Project**! This repository contains a comprehensive analysis of an email marketing campaign for a SaaS company’s freemium productivity tool. The goal is to optimize email performance by identifying the best email variant, send times, and user segments to maximize user engagement (open rates, click-through rates, and conversions to paid plans). The project includes data cleaning, exploratory data analysis (EDA), and A/B testing, implemented in Jupyter Notebooks, with scripts and a pipeline in development.

## Project Overview
A SaaS company wants to convert free users to paid subscribers through targeted email campaigns. We designed and analyzed an A/B test with three email variants (A, B, C) using a dataset of ~500,000 email interactions. The analysis uncovers which variant drives the highest engagement, optimal send times, and key user segments, providing actionable insights for the marketing team.

### Key Findings
- **Variant A Outperforms**: Achieved a 29.57% open rate and 1.53% conversion rate, significantly better than Variant B (21.26%, 0.67%) and C (26.43%, 0.21%).
- **Morning Sends Win**: Emails sent in the morning had a 27.71% open rate, compared to 27.31% (afternoon) and 23.01% (evening).
- **Active Users Engage Most**: Active users showed a 27.17% open rate, compared to inactive (18.48%) and new (25.78%) users.
- **Visual-Heavy Layouts Perform Slightly Better**: 25.90% open rate vs. 25.72% (balanced) and 25.59% (text-heavy).

## Repository Structure
```
email_ab_testing_project/
├── data/
│   ├── raw/                    # Raw dataset (not included in repo)
│   └── processed/              # Cleaned dataset (cleaned_email_data.csv)
├── notebooks/
│   ├── data_cleaning.ipynb     # Data cleaning and preprocessing
│   ├── eda.ipynb               # Exploratory data analysis
│   └── ab_testing.ipynb        # A/B testing and statistical analysis
├── scripts/                    # (In progress) Python scripts for automation
│   ├── cleaning.py
│   ├── eda.py
│   └── ab_testing.py
├── pipeline/                   # (In progress) Pipeline orchestration
│   └── main.py
├── outputs/
│   ├── eda/                    # EDA metrics and plots
│   └── ab_testing/             # A/B testing results
├── README.md                   # This file
└── requirements.txt            # Project dependencies
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/adnanrahmanpoor/email_ab_testing_project.git
   cd email_ab_testing_project
    ```    
2. Install Dependencies:
    Ensure Python 3.8+ is installed. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
    Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scipy`, `statsmodels`, `scikit-learn`, `lifelines`.

3. Prepare Data:
    Place the raw dataset (`email_campaign_data.csv`) in `data/raw/`. The dataset should include columns: `user_id`, `variant`, `send_time`, `content_layout`, `device`, `user_segment`, `account_age`, `feature_usage`, `open`, `click`, `convert`, `timestamp`.
4. Run Notebooks:
    Open Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    Run `data_cleaning.ipynb`, `eda.ipynb`, and `ab_testing.ipynb` in order.
## Usage
* **Data Cleaning**: Run `data_cleaning.ipynb` to preprocess the dataset, handling missing values, removing ~15,000 duplicates, and validating data types.
* **EDA**: Run `eda.ipynb` to explore patterns (e.g., variant performance, send time trends) with 20+ visualizations and metrics tables.
* **A/B Testing**: Run `ab_testing.ipynb` to compare variants using Z-tests, post-hoc testing, propensity score matching, and survival analysis.
* **Scripts and Pipeline**: (In progress) Once completed, `scripts/` will contain modular code, and `pipeline/main.py` will orchestrate the workflow.

## Future Work
* Convert notebooks to Python scripts (`cleaning.py`, `eda.py`, `ab_testing.py`) for automation.
* Implement a pipeline (`main.py`) to streamline cleaning, EDA, and A/B testing.
* Add unit tests and CI/CD for robustness.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements. Contact adnanrahmanpoor for collaboration.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
Built as a portfolio project to demonstrate data science skills in marketing analytics. Inspired by real-world SaaS marketing challenges.

---
