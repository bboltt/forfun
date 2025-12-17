1. Credit Card History & Usage
cc_days_since_last_pd (90%): Missing for customers who have never had a past-due event or hold no credit card. With an IV of 0.055, this feature is retained because, for the card-holding population, recent delinquency is a direct measure of financial distress and a top-tier fraud precursor.

cc_days_since_last_payment (83%): Missing for customers with no payment history found. Retained with an IV of 0.051 because an active card with no recent payments is a strong indicator of "run-up" behavior or potential bust-out activity.

cc_days_since_last_nsf (82%): Missing for customers who have never triggered an NSF event. Despite the high missing rate, the IV of 0.053 supports retaining this as a high-precision "Risk Flag," where the presence of a value indicates reckless account usage.

cc_overlimit_flag (82%): Missing for customers who have never exceeded their credit limit. Retained with an IV of 0.058 as a binary trigger for financial distress; customers hitting limits often exhibit behaviors highly correlated with default.

2. Cross-Product Latency & Spread Features
rcif_days_dep_to_cred (82%) & rcif_days_dep_to_loan (88%): Missing for customers who do not hold both deposit and lending products. Retained with IVs of ~0.10 because they measure the "Speed of Cross-Sell." A gap of ~0 days (concurrent opening) is a classic bust-out signal, while a long gap indicates organic growth.

cons_lending_avg_open_age_vs_all (89%): Missing for non-borrowers. With an IV of 0.109, this is retained to isolate personal loan borrowers and score the specific risk of "New Borrower" behavior relative to their total relationship tenure.

cons_cc_avg_open_age_vs_all (82%): Missing for non-cardholders. Retained with a strong IV of 0.127 to effectively score "New Credit on Old Profile" scenarios (potential Account Takeover) versus standard new customers.

cons_savings_avg_open_age_vs_all (57%): Missing for customers without savings accounts. With an IV of 0.102, this is retained to measure cross-sell maturity, distinguishing between long-term customers adding savings (low risk) and new entities (higher risk).

3. Application Velocity & PII
cc_apps_in_last_180_days & _90_days (80%): Missing when no credit card applications are found in the window. Retained with an IV of 0.106 to capture "Application Storms"—bursts of applications that are characteristic of synthetic identity attacks.

rdo_apps_in_last_180_days & _90_days (80%): Missing when no retail deposit applications are found. Retained (IV ~0.106) as a standard velocity counter to detect sudden spikes in account opening attempts.

apps_num_addrs, emails, dobs, ids, phones (46%): Missing for passive customers with no applications in the last 3 years. These are critical for active applicants, with strong IVs ranging from 0.20 to 0.28, proving that PII instability (e.g., one SSN using multiple names/addresses) is a major fraud predictor.

days_since_last_application (46%): Missing for passive customers. Retained with an IV of 0.160 to measure recency of demand and distinguish active credit seekers from the passive customer base.

4. Account Churn & Retention
cons_savings_avg_closed_age (83%): Missing for customers who have never closed a savings account. Retained because the very high IV of 0.350 confirms that for the churned sub-population, the age at closure is a massive predictor of risk.

cons_checking_avg_closed_age (75%): Missing for customers who have never closed a checking account. Retained with an IV of 0.260 to capture checking-specific attrition, which carries a distinct risk signal from savings churn.

avg_closed_age (55%) & max_closed_age (55%): Missing for "Loyal" customers who have never closed an account. Retained with IVs of 0.260 and 0.200 respectively, as they identify customers with deep historical ties who have recently severed relationships.

rcif_days_since_any_close (55%): Missing for customers with no closed accounts. Retained (IV 0.090) to track the recency of churn, providing a dynamic behavioral signal that static "Age" features miss.

5. Savings Account Profile
cons_savings_max_open_age (57%): Missing for customers with no open savings account. Retained with an IV of 0.141 as it captures the depth of the savings relationship for that specific segment.

cons_savings_min_acct_age (45%): Missing for customers who never had a savings account. Retained because its extremely high IV of 0.333 confirms it is critical for scrutinizing the "freshness" of the newest savings account, a common vector for new account fraud.
