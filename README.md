During the feature selection process, an analysis was conducted to evaluate features with elevated missing rates. In this dataset, missing values are not indicative of poor data quality; rather, they are structural and informational. A missing value represents a specific customer state—such as "Never had this product" or "Never closed an account"—which provides a distinct and valuable signal for modeling fraud risk.

We have elected to retain the following features despite their high missing rates because they demonstrate strong Information Value (IV), capture unique behavioral signals distinct from the general population, or serve as the strongest representative of a correlated cluster.

1. Account Age & History Features
cons_checking_max_open_age (Missing Rate: 12.4%): Retained as the primary feature in the "Account Age" cluster. Although correlated with avg_acct_age, this feature possesses the highest Information Value (0.43) in the group. Its missing rate differs significantly from avg_acct_age (0%), indicating that it effectively isolates the checking-account sub-population, whereas the average age metric conflates all product types.

cons_savings_max_open_age (Missing Rate: ~56%): Retained to capture the savings-specific customer segment. Its correlation with the checking equivalent is moderate, and the widely divergent missing rates confirm that these features track different customer behaviors and product ownership profiles.

max_closed_age (Missing Rate: 55.3%): Retained as the representative feature for the "Closed Account" cluster. While correlated with avg_closed_age, max_closed_age provides a distinct signal regarding the maximum longevity of a customer's relationship before churn. The missingness here correctly identifies "Loyal" or "New" customers who have never closed an account, a critical distinction for the model.

cons_checking_avg_closed_age (Missing Rate: 74.8%) & cons_savings_avg_closed_age (Missing Rate: 82.7%): Both features are retained because the correlation between them is below the exclusion threshold. They capture product-specific churn behavior that cannot be represented by a single aggregate feature; the high missing rates accurately reflect the subset of customers who have historically closed these specific product types.

rcif_open_age_spread: Retained after the removal of max_open_age and avg_open_age from its cluster. It serves as the surviving metric to capture the diversity in account maturity (the gap between oldest and newest accounts).

rcif_prod_herfindahl: Retained as a measure of product diversity. It was only correlated with the MIN(accts_open_30d_ago, total_open_accts) feature, which was dropped due to low IV (<0.1), leaving this Herfindahl index as the best metric for relationship complexity.

2. Balance Volatility Features
deposit_cons_max_min_ratio_180d (Missing Rate: 5.2%): Retained to capture long-term balance volatility. This feature showed the highest IV (0.47) in its cluster, outperforming the 90-day version.

deposit_cons_max_min_ratio_30d & deposit_cons_max_min_ratio_7d: Retained to capture medium and short-term volatility signals. The 7-day feature (IV 0.35) provides a distinct "immediate risk" signal that is separate from the long-term trends captured by the 180-day feature.

deposit_cons_coeff_var_180d & deposit_cons_coeff_var_30d: Retained to represent stability over time. The 90-day version was dropped to break the correlation chain, allowing these two distinct windows to represent long-term vs. medium-term behavioral stability without redundancy.

deposit_cons_coeff_var_7d: Retained despite an 0.82 correlation with the 30-day version. Given the high IV of these volatility features and the scarcity of fraud labels in the dataset, we opted to keep this feature to maximize our ability to detect rapid, short-term changes in account behavior.
