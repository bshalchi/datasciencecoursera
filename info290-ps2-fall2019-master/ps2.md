```python name="setup" hide_output=true
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 

# import seaborn
```

# 0. Write functions 

```python
def estimate_ate(): 
    '''
    return values 
    '''
```

# 1. What happens when pilgrims attend the Hajj pilgrimage to Mecca? 

On the one hand, participating in a common task with a diverse group of pilgrims might lead to increased mutual regard through processes identified in *Contact Theories*. On the other hand, media narritives have raised the spectre that this might be accompanied by "antipathy toward non-Muslims". [Clingingsmith, Khwaja and Kremer (2009)](https://dash.harvard.edu/handle/1/3659699) investigates the question. 

Using the data here, test the sharp null hypothesis that winning the visa lottery for the pilgrimage to Mecca had no effect on the views of Pakistani Muslims toward people from other countries. Assume that the Pakistani authorities assigned visas using complete random assignment. Use, as your primary outcome the `views` variable, and as your treatment feature `success`. If you're ambitious, write your fucntion generally so that you can also evaluate feeligns toward specific nationalities.

```python
d = pd.read_csv('./data/clingingsmith_2009.csv')
```

a. Using either `pandas`, group the data by `success` and report whether views toward others are generally more positive among lottery winners or lottery non-winners. 

```python
ate = estimate_ate()
```

b. But is this a meaningful difference, or could it just be randomization noise? Conduct 10,000 simulated random assignments under the sharp null hypothesis to find out. (Don't just copy the code from the async, think about how to write this yourself.) 

```python

```

c. How many of the simulated random assignments generate an estimated ATE that is at least as large as the actual estimate of the ATE? 

```python
num_larger = 'replace text' 
```

Please, make sure that you include your result into the printed space using the **`r num_larger`** inline code idiom for this, and all other answers. 

d. What is the implied *one-tailed* p-value? 

```python
p_value_one_tailed = 'replace text' 
```

e. How many of the simulated random assignments generate an estimated ATE that is at least as large *in absolute value* as the actual estimate of the ATE? 

```python
number_more_extreme = 'replace text'
```

f. What is the implied two-tailed p-value? 

```python
p_value_two_tailed = 'replace text' 
```

# 2. Randomization Inference Practice
McElhoe and Conner (1986) suggest using a *new* instrument called a "Visiplume" measure pollution. The EPA has a standard method for measuring pollution. Because they're good scientists, McElhoe and Conner want to validate that their instrument is measuring the same levels of pollution as the EPA instrument. 

To do so, they take six readings -- one with each instrument -- at a single site. The recorded response is the ratio of the Visiplume reading to the EPA standard reading, and the values that are recorded are: 0.950, 0.978, 0.762, 0.733, 0.823, and 1.011.

Suppose that we want to test the question, "Do the Visiplume readings and the EPA standard readings produce similar enough estimates?"

> (The point of this question is to demonstrate that randomization inference works as a general inferrential paradigm, without *necessairily* being tied to an experiment.)

1. How would you structure the sharp-null hypothesis -- that Visiplume and the EPA reaings are the same -- in this case? 

2. Suppose that our summary of the data is the sum of the ratios. That is, in the test that we conducted, we obsered $0.95 + ... + 1.011 = 5.257$. Using randomization inference, test the sharp-null hypothesis that you formed in the first part of the question. Produce a histogram of the test statistic under the sharp null that compares against the 5.257 value from the test, and also produce a two-sided p-value. 

```python
p_value = 'replace text' 
```

# 3. Term Limits Aren't Good. 

Naturally occurring experiments sometimes involve what is, in effect, block random assignment. For example, [Rocio Titiunik](https://sites.google.com/a/umich.edu/titiunik/publications) , in [this paper](http://www-personal.umich.edu/~titiunik/papers/Titiunik2016-PSRM.pdf) studies the effect of lotteries that determine whether state senators in TX and AR serve two-year or four-year terms in the aftermath of decennial redistricting. These lotteries are conducted within each state, and so there are effectively two distinct experiments on the effects of term length.

The "thoery" in the news (such as it is), is that legislators who serve 4 year terms have more time to slack off and not produce legislation. If this were true, then it would stand to reason that making terms shorter would increase legislative production. 

One way to measure legislative production is to count the number of bills (legislative proposals) that each senator introduces during a legislative session. The table below lists the number of bills introduced by senators in both states during 2003. 

```python
d = pd.read_stata('./data/titiunik_2010.dta')
d.head()
```

a. Using either `pandas`, group the data by state and report the mean number of bills introduced in each state. Does Texas or Arkansas seem to be more productive? Then, group by two- or four-year terms (ignoring states). Do two- or four-year terms seem to be more productive? **Which of these effects is causal, and which is not?** Finally, group by state and term-length. How, if at all, does this change what you learn? 

```python

```

b. For each state, estimate the standard error of the estimated ATE. 

```python
se_ate = 'replace text'
```

c. Use equation (3.10) to estimate the overall ATE for both states combined. 

```python
overall_ate = 'replace text'
```

d. Explain why, in this study, simply pooling the data for the two states and comparing the average number of bills introduced by two-year senators to the average number of bills introduced by four-year senators leads to biased estimate of the overall ATE. 

```python

```

e. Insert the estimated standard errors into equation (3.12) to estimate the stand error for the overall ATE. 

```python
se_overall_ate = 'replace text'
```

f. Use randomization inference to test the sharp null hypothesis that the treatment effect is zero for senators in both states. Here we mean: estimate the *overall ate* (which is, the weighted average of the block ate) as the internal part of your RI loop. 

```python
p_value = 'replace text'
```

g. **IN Addition:** Plot histograms for both the treatment and control groups in each state (for 4 histograms in total).

# 3. Cluster Randomization
Use the data in the table below to explore the consequences of cluster random assignment. (Assume that there are three clusters in treatment and four clusters in control.) Note that there is no randomization inference that is necessary to complete this problem because we have observed the *impossible* **science table**.  


```python
d = pd.read_csv('./data/clustering_data.csv')
```

a. Suppose the clusters are formed by grouping observations {1,2}, {3,4}, {5,6}, ... , {13,14}. Use equation (3.22) to calculate the standard error. Note that, because we have the full schedule of potential outcomes -- the science table -- it is possible to estimate $cov(\bar{Y}_{j}(0), \bar{Y}_{j}(1))$. If we did not posess this information, then we would need to work with equation 3.23. 

```python
def clustered_se(fill_out_args): 
    '''
    do stuff
    '''
    
```

b. Suppose that clusters are instead formed by grouping observations {1,14}, {2,13}, {3,12}, ... , {7,8}. Use equation (3.22) to calculate the standard error assuming half of the clusters are randomly assigned to treatment. 

```python

```

c. Why do the two methods of forming clusters lead to different standard errors? What are the implications for the design of cluster randomized experiments? 

```python

```

# 4. Sell Phones? 

Suppose that you are working for a company that sells online display advertisements. (The generation's smartest minds, lost to chasing those clicks...) On client, a consumer electronics company is considering using your ad network to run a large campaign. In order to evaluate its effectiveness, they want to run a smaller experiment to estimate the causal impact of the ads on sales of one of their smartphones. 

**The facts** 

- The experiment campaign will run for one week within a randomly samples sub-population of 800,000 users
- The cost per *impression* -- someone seeing the ad -- is $0.20. 
- The client tells you that they make a profit of \$100 every time someone purchases one of their smarphones (e.g. the device costs \$400 to manufacture, and are sold for \$500.)
- When they are **not** running the advertising campaign, the historic rate of purchasing has been that 0.004 of the population (0.4%) makes a purchase of this smartphone. 
- Assume that everyone who is assigned to the treatment group actually sees the ad. 
- Suppose there are no long-run effects and all the effects are measured within that week.


a. How large does the treatment effect need to be in order for the campaign to have positive value for the company? 

```python

```

b. Suppose the measured effect were to be 0.3 percentage points. If users are split 50:50 between the treatment group (exposed to iPhone ads) and control group (exposed to unrelated advertising or nothing; something you can assume has no effect), what will be the confidence interval of your estimate on whether people purchase the phone?

```python

```

  + **Hint:** The standard error for a two-sample proportion test is $\sqrt{p(1-p)*(\frac{1}{n_{1}}+\frac{1}{n_{2}})}$ where $p=\frac{x_{1}+x_{2}}{n_{1}+n_{2}}$, where $x$ and $n$ refer to the number of “successes” (here, purchases) over the number of “trials” (here, site visits). The length of each tail of a 95% confidence interval is calculated by multiplying the standard error by 1.96.
  
c. Based on this confidence interval, if the effect were 0.3 percentage points, would you recommend running the production campaign among the whole population? Why or why not?

d. Your boss at the newspaper, worried about potential loss of revenue, says he is not willing to hold back a control group any larger than 1% of users. What would be the width of the confidence interval for this experiment if only 1% of users were placed in the control group and 99% were placed in the treatment group?

```python

```

# 5. Sports Cards
Here you will find a set of data from an auction experiment by John List and David Lucking-Reiley ([2000](https://drive.google.com/file/d/0BxwM1dZBYvxBNThsWmFsY1AyNEE/view?usp=sharing)).  

```python
d = pd.read_csv('./data/list_data_2019.csv')
d.head()
```

In this experiment, the experimenters invited consumers at a sports card trading show to bid against one other bidder for a pair trading cards.  We abstract from the multi-unit-auction details here, and simply state that the treatment auction format was theoretically predicted to produce lower bids than the control auction format.  We provide you a relevant subset of data from the experiment.

In this question, we are asking you to produce p-values and confidence intervals in three different ways: (1) Using a `t.test`, using a regression, and using randomization inference. 

a. Using a `t.test`, compute a 95% confidence interval for the difference between the treatment mean and the control mean.

```python
t_test_result = 'replace text'
```
You should be able to look into `str(t_test_result)` to find the pieces that you want to pull to include in your written results.

b. In plain language, what does this confidence interval mean? (Put your answer in bold!) 

c. Regression on a binary treatment variable turns out to give one the same answer as the standard analytic formula you just used.  Demonstrate this by regressing the bid on a binary variable equal to 0 for the control auction and 1 for the treatment auction.

```python
# mod = smf.ols()
# mod_fit = mod.fit()
# mod_fit.summary()
```

d. Calculate the 95% confidence interval you get from the regression. There is a built in, 

```python

```

e. On to p-values. What p-value does the regression report? Note: please use two-tailed tests for the entire problem. (Should be able to pull this from the summary. And, you should try to do so with a call that _name_ calls for the parameter you're interested in, in this case, `uniform_price_auction`.) 

```python

```

f. Now compute the same p-value using randomization inference.

```python

```

g. Pull the same p-value from the `t.test`. 

```python

```

h. Compare the two p-values in parts (e) and (f). Are they much different? Why or why not? How might your answer to this question change if the sample size were different?

```python

```
