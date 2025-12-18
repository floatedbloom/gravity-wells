''' 
Conduct ANOVA data analysis on subreddit data from gravity well simulation.
'''
import argparse # for parsing commandline arguments
import sys      # for sys.argv to check number of arguments
import pathlib  # for manipulating datfile path info
import pandas as pd     # for DataFrames
import matplotlib.pyplot as plt     # for plotting
import seaborn as sns   # for easy plotting
import scipy.stats as stats     # for anova function f_oneway
# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols
# use tukeyhsd to distinguish which subreddits are not matching
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# QQ-plot
import statsmodels.api as sm

if __name__ == "__main__":
    if not len(sys.argv) > 1:
        sys.exit("Must call script from commandline with data filename")
    my_parser = argparse.ArgumentParser(
        description=__doc__, # Use auto python doc description
        formatter_class=argparse.RawDescriptionHelpFormatter)                     
    my_parser.add_argument('datfile',
        help="specify data file containing subreddit TSM data",
        type=str)
    my_parser.add_argument('-g', '--genGraph',dest='genGraph',action='store_true')
    my_parser.set_defaults(genGraph=False)
    my_args = my_parser.parse_args()

    df = pd.read_csv(my_args.datfile)
# format df to match layout required by ANOVA analysis
    x_label = 'subreddit'
    y_label = 'AvgTSM'
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], 
            value_vars=df.columns.to_list(),var_name=x_label,
            value_name=y_label)

# if requested, generate a boxplot to see the data distribution by treatments. 
## Using boxplot, we can easily detect differences between different treatments
    if my_args.genGraph:
        ax = sns.boxplot(x=x_label, y=y_label, data=df_melt, 
                width=0.5,color='#99c2a2')
        ax = sns.stripplot(x=x_label, y=y_label, data=df_melt, color='#7d0013')
        plt.xticks(rotation=60)     # rotate x-axis labels
        plt.subplots_adjust(bottom=0.2) # ensure plt labels are not cut off
        plt.show()

# stats f_oneway functions takes the groups as input and returns ANOVA F 
## and p value
    codestr = f'fvalue, pvalue = stats.f_oneway(df["'
    codestr += '"], df["'.join(col for col in df.columns)
    codestr += '"])'
    print(f'\nExecuting the statement: \n\t{codestr}')
    exec(codestr)
    print(f'\tfvalue = {fvalue}, pvalue = {pvalue}')

# Confirm with Ordinary Least Squares (OLS) model
    print('\nconfirming ANOVA result with OLS model')
    model = ols(f'{y_label} ~ C({x_label})', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
# output (ANOVA F and p value)
    print(anova_table)

    threshold = 0.05
    pval = anova_table['PR(>F)'][0]
    if pval<threshold and pvalue<threshold:
        print('\nConduct tukeyhsd test to determine which subreddits disagree.\n')
# Both ANOVA techniques agree that p_val<<0.05, so we have sufficient evidence 
## to say that the mean values across each subreddit are not equal.
## Thus, we can proceed to perform Tukey’s Test to determine exactly which 
## subreddit means are different.
        tk = pairwise_tukeyhsd(endog=df_melt[y_label], groups=df_melt[x_label], 
                alpha=threshold)
        tkdf = pd.DataFrame(tk.summary())
        tkdf = tkdf.rename(columns=tkdf.iloc[0]).drop(tkdf.index[0])
        # save Tukey results to csv file
        tkfp = pathlib.Path(my_args.datfile)
        tkfp = tkfp.with_name(tkfp.stem+'_tuk').with_suffix('.csv')
        tkdf.to_csv(tkfp,index=False)
        print(tk)
    else:
        print('\nAt least one result indicates mean values are equal:')
        print(f'\tpvalue from scipy.stats.f_oneway = {pvalue}')
        print(f'\tpvalue from OLS model = {pval}')
# test residuals of ANOVA for valid assumptions of normality
# use Shapiro-Wilk test for normality
    print('\nRunShapiro-Wilk test for normality')
    sw_stat, sw_pval = stats.shapiro(model.resid)
    if sw_pval < 0.5:
        print(f'\tShapiro pval = {sw_pval} indicates residuals are NOT normal')
    else:
        print(f'\tShapiro pval = {sw_pval} indicates residuals ARE normal')
    if my_args.genGraph: # if graph generation specified
# now, generate QQ-plot to check for linearity and histogram for 
## checking normality
        print('\nCheck validity of ANOVA normality assumptions with QQ-plot and ' +
                'histogram of residuals')
        figure, (ax1,ax2) = plt.subplots(1, 2)
# QQ-plot
        #plt.subplot(1, 2, 1)  # row 1, column 1, count 1
        sm.qqplot(model.resid, line='s', ax=ax1)
        ax1.set_title('QQ-plot of ANOVA residuals')
        ax1.set_xlabel("Theoretical Quantiles")
        ax1.set_ylabel("Standardized Residuals")
# histogram
        #plt.subplot(1, 2, 2)  # row 1, column 2, count 1
        ax2.hist(model.resid, bins='auto', histtype='bar', ec='k')
        ax2.set_title('histogram plot of ANOVA residuals')
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel('Frequency')
        
        plt.suptitle('Check ANOVA assumptions')
        plt.tight_layout()
        plt.show()
# use Bartlett’s test to check the assumption of homogeneity of variances
    print("\nrun Bartlett's test for homogeneity")
    codestr = f'bstat, bp = stats.bartlett(df["'
    codestr += '"], df["'.join(col for col in df.columns)
    codestr += '"])'
    print(f'\nExecuting the statement: \n\t{codestr}')
    exec(codestr)
    print(f'\nBartlett stat = {bstat}, Bartlett p val = {bp}')
    if bp < 0.5:
        print(f'\tBartlett pval = {bp} indicates variances are NOT homegenous')
    else:
        print(f'\tBartlett pval = {bp} indicates variances ARE homegenous')
# use Levene’s test to check the assumption of homogeneity of variances
## in case the data is not truly normal
    print("\nrun Levene's test for homogeneity")
    codestr = f'lstat, lp = stats.levene(df["'
    codestr += '"], df["'.join(col for col in df.columns)
    codestr += '"])'
    print(f'\nExecuting the statement: \n\t{codestr}')
    exec(codestr)
    print(f'\nLevene stat = {bstat}, Levene p val = {lp}')
    if bp < 0.5:
        print(f'\tLevene pval = {lp} indicates variances are NOT homegenous')
    else:
        print(f'\tLevene pval = {lp} indicates variances ARE homegenous')
    print('\nRegardless of the outcome, it has been noted (Weerahandi, S. ' +
            '(1995). ANOVA under Unequal Error Variances. Biometrics, 51(2), ' +
            '589–599. https://doi.org/10.2307/2532947) that ANOVA is relatively ' +
            'robust to heterogenous variances, particularly for data with equal ' +
            'sample sizes)')
