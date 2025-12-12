During the feature selection process, an analysis was conducted to evaluate features with elevated missing rates. In this dataset, missing values are not indicative of poor data quality; rather, they are structural and informational. A missing value represents a specific customer state—such as "Never had this product" or "Never closed an account"—which provides a distinct and valuable signal for modeling fraud risk.

We have elected to retain the following features with missing rates above 40% because they capture unique behavioral signals distinct from the general population and represent specific customer segments that broader features obscure.

Account Age & History Features
cons_savings_max_open_age (Missing Rate: ~57%)

Justification: This feature is retained to explicitly capture the savings-specific customer segment. Its correlation with the checking account equivalent (cons_checking_max_open_age) is moderate, and the significant difference in missing rates (57% vs 12%) confirms that these features track different customer profiles. The high missing rate accurately flags customers who do not own a savings product, a distinction that would be lost if merged into a generic "max age" feature.

max_closed_age (Missing Rate: 55.3%)

Justification: Retained as the representative feature for the "Closed Account" cluster. While correlated with avg_closed_age, this feature provides a distinct signal regarding the maximum longevity of a customer's relationship prior to churn. The 55% missing rate correctly identifies the "Loyal" or "New" customer base who have never closed an account. This binary distinction (Has Closed History vs. No Closed History) is a critical risk factor that justified retaining the feature despite the high missingness.

cons_checking_avg_closed_age (Missing Rate: 74.8%)

Justification: Retained to capture churn behavior specific to checking accounts. The correlation with the savings equivalent is below the exclusion threshold, meaning checking account churners behave differently than savings account churners. The high missing rate reflects the large portion of the customer base that has either never had a checking account or never closed one.

cons_savings_avg_closed_age (Missing Rate: 82.7%)

Justification: Retained to capture churn behavior specific to savings accounts. Similar to the checking feature, the extreme missingness is a valid signal representing the absence of savings account churn history. Keeping this separate allows the model to detect risk patterns unique to savings products without conflating them with checking activity.

cons_investment_avg_closed_age (Missing Rate: ~95%)

Justification: This feature is retained because its high missing rate provides a clear structural signal: it identifies customers who either do not possess a consumer investment product or have never closed one. Despite the high missingness, the feature demonstrates a moderate Information Value relative to the mean, indicating that among the small population of customers who do have a history of closing investment accounts, the "age at closure" is a predictive risk factor worth preserving.

Loan & Mortgage Performance Features
il_max_days_delinquent & il_tot_late_fees_per_pmt (Missing Rate: ~95%)

Justification: These Installment Loan (IL) features are retained despite ~95% missingness because the missing values structurally identify customers who do not hold an installment loan product. For the ~5% of the population that does hold such loans, these metrics demonstrated moderate Information Value. Retaining them allows the model to leverage specific repayment performance signals (delinquency and late fees) for borrowers, which are strong indicators of financial stress or fraud intent, without diluting the signal for non-borrowers.

ml_num_delinquent_pmt & ml_num_foreclosure_stop (Missing Rate: ~94%)

Justification: Similar to the installment loan features, these Mortgage Loan (ML) features are missing for the vast majority of customers simply because they do not have a mortgage with the bank. They were retained due to moderate Information Value, ensuring the model can detect severe credit deterioration events (such as foreclosure stops) within the mortgage-holding sub-population, a high-value signal that cannot be captured by generic account features.
