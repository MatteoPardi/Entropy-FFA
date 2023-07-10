Let's suppose we want to compare the results:
- FFA20: E(TS) = n +- dn = 90.2 +- 0.3
- FFA+Ent20: E(TS) = y +- dy = 98.3 +- 0.5

where mean and std are over 5 trials. (nobs = 5).

In the paper's tables, before the peer review, the boolean criterion used to mark in bold a value was
- n_is_bold = (n >= y)
- y_is_bold = (y >= n)

A peer reviewer suggested to perform statistical test to check if the difference among means are singnificant.
Thus, we decided to change the criterion, and use the following one, based on a Welch's t-test, with significance level alpha=0.05.

- pvalue = scipy.stats.ttest_ind_from_stats(n, dn, nobs, y, dy, nobs, equal_var=False, alternative='two-sided')
- n_is_bold = (n >= y) or (pvalue > alpha and y < 100)
- y_is_bold = (y >= n) or (pvalue > alpha and n < 100)

Explanation:
1. the winner (the one with bigger E(TS) mean) is always bold
2. the loser is not bold if the winner is 100.0+-0 and the loser is x+-z with x < 100
3. if the situation is not 2.:
    - if pvalue > alpha, the loser is bold 
    - if pvalue < alpha, the loser is not bold 