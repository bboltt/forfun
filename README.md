Commercial Deposit Features
deposit_comm_coeff_var_90d, deposit_comm_kurtosis_90d, deposit_comm_skewness_90d, deposit_comm_max_min_ratio_90d (Missing Rate: ~91%)

Justification: These higher-order statistical features (Coefficient of Variation, Kurtosis, Skewness, Max/Min Ratio) are missing for the 91% of customers who do not hold a commercial deposit account. For the business-holding segment, we adopted a uniform 90-day window strategy. As noted in the correlation analysis, while consumer accounts supported multiple time windows (7d, 30d, 180d), the commercial data is sparser. We therefore retained the 90-day versions of these distribution metrics as the single, robust standard for capturing the shape and volatility of business cash flows, avoiding the noise of shorter windows while preserving the signal for irregular or "spiky" transaction behavior often associated with commercial fraud.

deposit_comm_bal_change_rate_norm_abs_30d & deposit_comm_bal_change_rate_norm_abs_90d (Missing Rate: ~91%)

Justification: These features measure the magnitude of balance changes (velocity) over monthly and quarterly horizons. They are retained to capture significant capital movements in business accounts. Unlike the statistical distribution features where a single window sufficed, keeping both the 30-day and 90-day change rates allows the model to distinguish between a recent, sudden ramp-up in activity (30d) versus a sustained trend of balance shifting (90d), which helps differentiate between legitimate business growth and potential bust-out accumulation.

account_days_deposit_comm_negative_7d & account_days_deposit_comm_negative_90d (Missing Rate: ~91%)

Justification: These are critical financial distress indicators. A missing value is a neutral "safe" signal (no commercial account), while a populated value indicates the number of days a business was in overdraft. We retained both the 7-day and 90-day versions because they signal different risk profiles: the 7-day feature detects immediate, acute liquidity crises (often a precursor to immediate default), while the 90-day feature identifies chronic insolvency or long-term reliance on overdrafts. Both signals are highly predictive of credit failure and fraud intent.
Credit Card Usage & History Features
cc_days_since_last_pd & cc_bal_current_max (Missing Rate: ~90%)

Justification: These features are structurally missing for customers without a consumer credit card. However, for the ~10% of customers who do hold a credit card, these metrics provide some of the strongest signals for financial distress and bust-out fraud. cc_days_since_last_pd (Days Since Past Due) acts as a critical delinquency indicator, while cc_bal_current_max captures utilization spikes. Retaining them allows the model to score "Credit Active" customers with high precision, while treating the missing population as a neutral baseline.

cons_cc_max_open_age (Missing Rate: ~82%)

Justification: Similar to the checking and savings age features, this metric isolates the credit card holder population. It was retained to measure the "depth" of the credit relationship. A missing value indicates no credit card relationship, whereas a low value for an existing cardholder indicates a "fresh" account, which carries a different risk profile compared to a long-standing line of credit.

Application PII & Velocity Features
apps_num_addrs, apps_num_emails, apps_num_dobs (Missing Rate: ~46%)

Justification: These features are missing for customers who have not submitted an application within the 3-year lookback window. This missingness effectively segments the population into "Passive Customers" (no recent apps) vs. "Active Applicants." For the active group, these features are critical stability indicators—a high count of unique addresses or emails associated with a single SSN is a classic synthetic identity or account takeover signal. We retain them to ensure the model can detect PII instability among active applicants.

cc_apps_in_last_180_days & rdo_apps_in_last_180_days (Missing Rate: ~80%)

Justification: These are pure velocity features. The high missing rate simply reflects that 80% of customers have not applied for a Credit Card (CC) or Retail Deposit (RDO) online in the last 6 months. This is a valid "safe" signal. The presence of a value (even a low one) indicates recent demand for credit or new accounts. These are retained as standard velocity checks to detect rapid application bursts.

Consumer Lending Features
cons_lending_avg_open_age_vs_all (Missing Rate: ~89%)

Justification: This feature captures the relative maturity of a customer's personal loans compared to their other relationships. The ~89% missing rate identifies non-borrowers. It is retained because, for borrowers, a discrepancy between loan age and general relationship age (e.g., a brand new loan for a long-time customer vs. a brand new loan for a brand new customer) offers a nuanced view of "new account" risk that aggregate age features miss.
