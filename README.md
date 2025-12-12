During the feature selection process, an analysis was conducted to evaluate features with elevated missing rates. In this dataset, missing values are not indicative of poor data quality; rather, they are structural and informational. A missing value represents a specific customer state—such as "Never had this product" or "Never closed an account"—which provides a distinct and valuable signal for modeling fraud risk.

We have elected to retain the following features with missing rates above 40% because they capture unique behavioral signals distinct from the general population and represent specific customer segments (e.g., savings-only customers or churned customers) that broader features obscure.

Account Age & History Features
cons_savings_max_open_age (Missing Rate: ~57%)

Justification: This feature is retained to explicitly capture the savings-specific customer segment. Its correlation with the checking account equivalent (cons_checking_max_open_age) is moderate, and the significant difference in missing rates (57% vs 12%) confirms that these features track different customer profiles. The high missing rate accurately flags customers who do not own a savings product, a distinction that would be lost if merged into a generic "max age" feature.

max_closed_age (Missing Rate: 55.3%)

Justification: Retained as the representative feature for the "Closed Account" cluster. While correlated with avg_closed_age, this feature provides a distinct signal regarding the maximum longevity of a customer's relationship prior to churn. The 55% missing rate correctly identifies the "Loyal" or "New" customer base who have never closed an account. This binary distinction (Has Closed History vs. No Closed History) is a critical risk factor that justified retaining the feature despite the high missingness.

cons_checking_avg_closed_age (Missing Rate: 74.8%)

Justification: Retained to capture churn behavior specific to checking accounts. The correlation with the savings equivalent is below the exclusion threshold, meaning checking account churners behave differently than savings account churners. The high missing rate reflects the large portion of the customer base that has either never had a checking account or never closed one.

cons_savings_avg_closed_age (Missing Rate: 82.7%)

Justification: Retained to capture churn behavior specific to savings accounts. Similar to the checking feature, the extreme missingness is a valid signal representing the absence of savings account churn history. Keeping this separate allows the model to detect risk patterns unique to savings products without conflating them with checking activity.
