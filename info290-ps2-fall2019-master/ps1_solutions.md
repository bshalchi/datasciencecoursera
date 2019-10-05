---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 1. Potential Outcomes Notation 
Explain the following notation: 

- $Y_{i}(1)$


The potential outcome to treatment for unit $i$. 


- $E[Y_{i}(1)|d_{i}=0]$


The expectation of the potential outcomes to treatment, of those people who we assign to control. *A counterfacutal outcome.*


- The difference, if any, between $E[Y_{i}(1)]$ and the notation $E[Y_{i}(1)|d_{i}=1]$. 


The first is the expected value of *all* individuals potential outcomes to treatment, without respect to if they are going to be assigned to treatment, or to control. The second is the expected value of the potential outcomes to treatment, given that we've assigned the units $i$ to be in treatment. 

**IF THE ASSIGNMENT MECHANISM IS RANDOM** then these two population estimates are the same in expectation. If it isn't random, then there's no guarantee that these two values are the same. 


- The difference, if any, between $E[Y_{i}(1)|d_{i}=1]$ and the notation $E[Y_{i}(1)|D_{i}=1]$. Use exercise 2.7 from FE to give a concrete example of the difference.


# 2. Potential Outcomes Practice 


Use the values in the following table to illustrate that $E[Y_{i}(1)] - E[Y_{i}(0)] = E[Y_{i}(1) - Y_{i}(0)]$.

```python
import pandas as pd
import numpy as np
```

```python
d = pd.DataFrame({
    'y0':  [5,3,10,5,10], 
    'y1':  [6, 8, 12, 5, 8], 
    'tau': [1,5,2,0,-2]}
)
```

```python
# first statement 
np.isclose(a = d['y1'].mean() - d['y0'].mean(), b = d['tau'].mean())
```

# 3. Conditional Expectations 
Consider the following table. 

```python
d_full = pd.DataFrame({
    'row_label' : [1, 2, 3, 4, 5, 6, 7, 'average'], 
    'y0' : [10, 15, 20, 20, 10, 15, 15, 15], 
    'y1' : [15, 15, 30, 15, 20, 15, 30, 20], 
    'tau': [5, 0, 10, -5, 10, 0, 15, 5]
})
d = d_full.iloc[:7,] # what the except index? check for the future!  
```

```python
d
```

Use the values in the table to calculate, for each cell, 

- The number of observations in the cell; and, 
- The percentage of the data that number of responses constitues. 

That is, for each cell do the following: 

- Fill in the number of observations in each of the nine cells; 
- Indicate the percentage of all subjects that fall into each of the nine cells. 
- At the bottom of the table, indicate the proportion of subjects falling into each category of $Y_{i}(1)$. 
- At the right of the table, indicate the proportion of subjects falling into each category of $Y_{i}(0)$. 
- Use the table to calculate the conditional expectation that $E[Y_{i}(0)|Y_{i}(1) > 15]$. 
- Use the table to calculate the conditional expectation that $E[Y_{i}(1)|Y_{i}(0) > 15]$. 

```python
d.loc[d['y1'] > 15, 'y0'].mean()
```

```python
d.loc[d['y0'] > 15, 'y1'].mean()
```

# 4. More Practice with Potential Outcomes 

Suppose we are interested in the hypothesis that children playing outside leads them to have better eyesight. Consider the following population of ten representative children whose visual acuity we can measure. (Visual acuity is the decimal version of the fraction given as output in standard eye exams. Someone with 20/20 vision has acuity 1.0, while someone with 20/40 vision has acuity 0.5. Numbers greater than 1.0 are possible for people with better than “normal” visual acuity.)

```python
d = pd.DataFrame({
    'child':  np.arange(1,11),
    'y0'   : [1.1, 0.1, 0.5, 0.9, 1.6, 2.0, 1.2, 0.7, 1.0, 1.1], 
    'y1'   : [1.1, 0.6, 0.5, 0.9, 0.7, 2.0, 1.2, 0.7, 1.0, 1.1]
})
d
```

In the table, state $Y_{i}(1)$ means “playing outside an average of at least 10 hours per week from age 3 to age 6,” and state $Y_{i}(0)$ means “playing outside an average of less than 10 hours per week from age 3 to age 6.”   $Y_{i}$ represents visual acuity measured at age 6.


Complete the following: 
    
- Compute the individual treatment effect for each of the ten children.  Note that this is only possible because we are working with hypothetical potential outcomes; we could never have this much information with real-world data. (We encourage the use of computing tools on all problems, but please describe your work so that we can determine whether you are using the correct values.)

```python
d['tau'] = d['y1'] - d['y0']
d
```

- In a single paragraph, tell a story that could explain this distribution of treatment effects.


This vector of treatment effects could arise because, for _most_ people, there is no effect of being inside or outside when they play. However, some individuals may have a genetic disposition -- known as *mothra* -- that leads them to look directly into the sun whenever they are outside.

Sadly, this person, individual #4, if they are let outside to play, will look directly into the sun, and as a result will have a lower visual acuity if they are allowed to play outside. 




- For this population, what is the true average treatment effect (ATE) of playing outside.

```python
d['tau'].mean()
```

- Suppose we are able to do an experiment in which we can control the amount of time that these children play 
outside for three years.  We happen to randomly assign the odd-numbered children to treatment and the even-numbered children to control.  What is the estimate of the ATE you would reach under this assignment? (Again, please describe your work.)

```python
np.arange(1,10,2)
```

```python
odd_idx = np.arange(1,10,2)
even_idx = [2, 4, 6 ,8, 10] # a couple of options
odd_treats = d.loc[d['child'].isin(odd_idx), 'y1'].mean()
even_controls = d.loc[d['child'].isin(even_idx), 'y0'].mean()
```

```python
odd_treats - even_controls
```

- How different is the estimate from the truth?  Intuitively, why is there a difference?


Maybe its close? Its only 0.02 away from the truth. But, maybe that's far? Because its 50% larger than the truth. Yargh, like a pirate! We need a notion of uncertainty! 


- We just considered one way (odd-even) an experiment might split the children. How many different ways (every possible way) are there to split the children into a treatment versus a control group (assuming at least one person is always in the treatment group and at least one person is always in the control group)?


If we have only two people that we were assigning, what would the entire treatment vector space look like? 

One looks like `[1, 0]`. 
Another looks like `[0, 1]`. 

What about if there are three people who we can assign? 

`[1, 1, 0]`
`[1, 0, 1]` 
`[0, 1, 1]` 
`[0, 0, 1]`, 
`[0, 1, 0]`, 
`[1, 0, 0]`. 

When we put all of these options together, we've got: 

```python
2**10 - 2
```

We can define this more clearly by defining out a choose operator: 

```python
def choose_operator(N, c): 
    result = np.math.factorial(N) / (np.math.factorial(N - c) * np.math.factorial(c))
    return(result)
```

```python
np.sum([choose_operator(10, i) for i in range(1,10)])
```

- Suppose that we decide it is too hard to control the behavior of the children, so we do an observational study instead.  Children 1-5 choose to play an average of more than 10 hours per week from age 3 to age 6, while Children 6-10 play less than 10 hours per week.  Compute the difference in means from the resulting observational data.

```python
d.loc[d['child'] < 6, 'y1'].mean() - d.loc[d['child'] >= 6, 'y0'].mean()
```

- Compare your answer in (h) to the true ATE.  Intuitively, what causes the difference?





# 5. Randomization and Experiments 

Suppose that a researcher wants to investigate whether after-school math programs improve grades. The researcher randomly samples a group of students from an elementary school and then compare the grades between the group of students who are enrolled in an after-school math program to those who do not attend any such program. Is this an experiment or an observational study? Why? 

> This is quite obviously an observational study. The students whose parents **select** to have their kids attend the after school programming are (or at least might be) different than the parents who **select** to have their kids *not* attend the after school programming. In fact, in Berkeley the reason that kids do not attend after school math programs is because instead, their parents have hired a PhD tutor to work with them on discrete math over the summer. And, as a result, the after school math program would, "Be a waste of time for my child, Copernicus." 


# 6. Lotteries

A researcher wants to know how winning large sums of money in a national lottery affect people's views about the estate tax. The research interviews a random sample of adults and compares the attitudes of those who report winning more than $10,000 in the lottery to those who claim to have won little or nothing. The researcher reasons that the lottery choose winners at random, and therefore the amount that people report having won is random. 


- Critically evaluate this assumption. 
- Suppose the researcher were to restrict the sample to people who had played the lottery at least once during the past year. Is it safe to assume that the potential outcomes of those who report winning more than \$10,000 are identical, in expectation, to those who report winning little or nothing? 

**Clarifications**

1. Please think of the outcome variable as an individual’s answer to the survey question “Are you in favor of raising the estate tax rate in the United States?”
2. The hint about potential outcomes could be rewritten as follows: Do you think those who won the lottery would have had the same views about the estate tax if they had actually not won it as those who actually did not win it? (That is, is $E[Y_{i}0|D=1] = E[Y_{i}0|D=0]$, comparing what would have happened to the actual winners, the $|D=1$ part, if they had not won, the $Y_{i}(0)$ part, and what actually happened to those who did not win, the $Y_{i}(0)|D=0$ part.) In general, it is just another way of asking, "are those who win the lottery and those who have not won the lottery comparable?"
3. Assume lottery winnings are always observed accurately and there are no concerns about under- or over-reporting.

> One immediate problem is that while the lottery choose winners at random from those who choose to play the lottery -- there is still selection into the types of people who play the lottery. In fact, there is a [strong argument](https://taxfoundation.org/are-lottery-taxes-regressive-and-what-does-regressive-mean-anyway/) to be made that Lottery systems serve as a retrogressive tax against lower-income individuals. So, those who choose to play might be lower SES, and therefore have different views about the estate tax than those who choose to abstain. 



# 7. Inmates and Reading
A researcher studying 1,000 prison inmates noticed that prisoners who spend at least 3 hours per day reading are less likely to have violent encounters with prison staff. The researcher recommends that all prisoners be required to spend at least three hours reading each day. Let $d_{i}$ be 0 when prisoners read less than three hours each day and 1 when they read more than three hours each day. Let $Y_{i}(0)$ be each prisoner's PO of violent encounters with prison staff when reading less than three hours per day, and let $Y_{i}(1)$ be their PO of violent encounters when reading more than three hours per day. 

In this study, nature has assigned a particular realization of $d_{i}$ to each subject. When assessing this study, why might one be hesitant to assume that ${E[Y_{i}(0)|D_{i}=0] = E[Y_{i}(0)|D_{i}=1]}$ and $E{[Y_{i}(1)|D_{i}=0] = E[Y_{i}(1)|D_{i}=1]}$? In your answer, give some intuitive explanation in English for what the mathematical expressions mean.



I'm worried that the types of people who select into reading or not, might be fundimentally different from one another. For example, the thugs over at Stanford who have chosen to be business thugs might must have been more likley to get into fights **and** not read. Whereas, the folks who choose to work at Berkeley are both more likely to read, *and* less likely to get into a fight. 


So, if I "make" people read, then I'm not actually changing their *type*, instead, I'm just changing whether they read or not. But, the pattern that the researcher saw in the first 1,000 person sample *only* had type based differences, not intervention based differences. 

```python

```
