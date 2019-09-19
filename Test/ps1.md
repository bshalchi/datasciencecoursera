# 1. Potential Outcomes Notation 
Explain the following notation: 


- $Y_{i}(1)$ 



- $E[Y_{i}(1)|d_{i}=0]$ 



- The difference, if any, between $E[Y_{i}(1)]$ and the notation $E[Y_{i}(1)|d_{i}=1]$. 




- The difference, if any, between $E[Y_{i}(1)|d_{i}=1]$ and the notation $E[Y_{i}(1)|D_{i}=1]$. Use exercise 2.7 from FE to give a concrete example of the difference.



### Brandon's Response to #1:
- The potential outcome to treatment for the ith subject.
- The expectation of the potential outcome to treatment for the ith subject when in the control group.
- The expectation of the potential outcome to treatment for the ith subject explains the first notation. The expectation of the potential outcome to treatment for the ith subject when in the treatment group explains the second notation.
- The first notation denotes the potential outcome to treatment for the ith subject when exposed to the treatment. The second notation denotes the potential outcome to treatment for the ith subject "who would be treated under some hypothetical allocation of treatments" (Gerber & Green, 2012, p. 24). For example, the possibility of a village's potential outcome to treatment being assigned to treatment across all other possibilites of being assigned to treatment. 




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
d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y0</th>
      <th>y1</th>
      <th>tau</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>8</td>
      <td>-2</td>
    </tr>
  </tbody>
</table>
</div>



### Brandon's Response to # 2

The expectation of the Average Treatment Effect (ate) is equal to the difference between the expectation of the potential outcome to treatment for the ith subject and the expectation of the potential outcome to control for the ith subject. For example:


```python
subject0control = d["y0"][0]
print("The potential outcome to control for subject 0 is: " + str(subject0control))

subject0treatment = d["y1"][0]
print("The potential outcome to treatment for subject 0 is: " + str(subject0treatment))

teOfSubject0 = subject0treatment - subject0control
print("The Treatment Effect of subject 0 is: " + str(teOfSubject0))

EofY0 = d["y0"].mean()
print("The average of the potential outcome to control when in the control group for all five subjects is: " + str(EofY0))

EofY1 = d["y1"].mean()
print("The average of the potential outcome to control when in the treatment group for all five subjects is: " + str(EofY1))

# The values of the vector tau is filled with each subjects Treatment Effect, which is: 
ate = d["y1"] - d["y0"]
print(ate)
print("So the Average Treatment Effect of y1 - y0 (or the vector tau) is: " + str(ate.mean()))


```

    The potential outcome to control for subject 0 is: 5
    The potential outcome to treatment for subject 0 is: 6
    The Treatment Effect of subject 0 is: 1
    The average of the potential outcome to control when in the control group for all five subjects is: 6.6
    The average of the potential outcome to control when in the treatment group for all five subjects is: 7.8
    0    1
    1    5
    2    2
    3    0
    4   -2
    dtype: int64
    So the Average Treatment Effect of y1 - y0 (or the vector tau) is: 1.2


# 3. Conditional Expectations 
Consider the following table. 


```python
d = pd.DataFrame({
    'row_label' : [1, 2, 3, 4, 5, 6, 7, 'average'], 
    'y0' : [10, 15, 20, 20, 10, 15, 15, 15], 
    'y1' : [15, 15, 30, 15, 20, 15, 30, 20], 
    'tau': [5, 0, 10, -5, 10, 0, 15, 5]
})
d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_label</th>
      <th>y0</th>
      <th>y1</th>
      <th>tau</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>20</td>
      <td>30</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20</td>
      <td>15</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>10</td>
      <td>20</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>15</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>15</td>
      <td>30</td>
      <td>15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>average</td>
      <td>15</td>
      <td>20</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



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

| $Y_{i}(0)$ \ $Y_{i}(1)$ | 15 | 20 | 30 | Marginal Distribution |
|-------------------------|----|----|----|-----------------------|
| 10                      | ** | ** | ** | **                    |
| 15                      | ** | ** | ** | **                    |
| 20                      | ** | ** | ** | **                    |
| Marginal Distribution   | ** | ** | ** | 1.0                   |

### Brandon's Response to #3:

| $Y_{i}(0)$ \ $Y_{i}(1)$ | 15  | 20  | 30 | Marginal Distribution |
|-------------------------|---- |---- |----|-----------------------|
| 10                      |  1  |  1  |  0 | 2/7                    |
| 15                      |  2  |  0  |  1 | 3/7                    |
| 20                      |  1  |  0  |  1 | 2/7                    |
| Marginal Distribution   | 4/7 | 1/7 | 2/7| 1.0                   |





```python
df = pd.DataFrame({
    "y0": [10, 15, 20],
    "15": [1, 2, 1],
    "20": [1, 0 ,0],
    "30": [0, 1, 1],
    "Marginal Distribution": [0, 0, 0]
})


# % of all subjects in each cell

dfPercent = pd.DataFrame({
    "y0": [10, 15, 20],
    "15": [1/7, 2/7, 1/7],
    "20": [1/7, 0 ,0],
    "30": [0, 1/7, 1/7],
})


def fun(dataframe):
    """Takes df and sums vector values for specified df."""
    answer = 0
    float(answer)
    for i in dataframe:
        answer += i
    return answer

dfPercent["Marginal Distribution of Y1"] = [fun(dfPercent["15"]), fun(dfPercent["20"]), fun(dfPercent["30"])]
dfPercent["Marginal Distribution of Y1"]
print(dfPercent)
    
print(sum(dfPercent["Marginal Distribution of Y1"]))
    
# 
```

       y0        15        20        30  Marginal Distribution of Y1
    0  10  0.142857  0.142857  0.000000                     0.571429
    1  15  0.285714  0.000000  0.142857                     0.142857
    2  20  0.142857  0.000000  0.142857                     0.285714
    0.9999999999999999


- The conditional expectation of $E[Y_{i}(0)|Y_{i}(1) > 15]$ 
- The conditional expectation of $E[Y_{i}(1)|Y_{i}(0) > 15]$


```python
print("The conditional expectation of E[Y_{i}(0)|Y_{i}(1) > 15] is " + str(d[d["y1"] > 15].y0.mean()))
print("The conditional expectation of E[Y_{i}(1)|Y_{i}(0) > 15] is " + str(d[d["y0"] > 15].y1.mean()))
```

    The conditional expectation of E[Y_{i}(0)|Y_{i}(1) > 15] is 15.0
    The conditional expectation of E[Y_{i}(1)|Y_{i}(0) > 15] is 22.5


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>child</th>
      <th>y0</th>
      <th>y1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.1</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.1</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1.6</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1.2</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.7</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1.1</td>
      <td>1.1</td>
    </tr>
  </tbody>
</table>
</div>



In the table, state $Y_{i}(1)$ means “playing outside an average of at least 10 hours per week from age 3 to age 6,” and state $Y_{i}(0)$ means “playing outside an average of less than 10 hours per week from age 3 to age 6.”   $Y_{i}$ represents visual acuity measured at age 6.

Complete the following: 
    
1. Compute the individual treatment effect for each of the ten children.  Note that this is only possible because we are working with hypothetical potential outcomes; we could never have this much information with real-world data. (We encourage the use of computing tools on all problems, but please describe your work so that we can determine whether you are using the correct values.)
2. In a single paragraph, tell a story that could explain this distribution of treatment effects.
- What might cause some children to have different treatment effects than others?
- For this population, what is the true average treatment effect (ATE) of playing outside.
- Suppose we are able to do an experiment in which we can control the amount of time that these children play outside for three years.  We happen to randomly assign the odd-numbered children to treatment and the even-numbered children to control.  What is the estimate of the ATE you would reach under this assignment? (Again, please describe your work.)
- How different is the estimate from the truth?  Intuitively, why is there a difference?
- We just considered one way (odd-even) an experiment might split the children. How many different ways (every possible way) are there to split the children into a treatment versus a control group (assuming at least one person is always in the treatment group and at least one person is always in the control group)?
- Suppose that we decide it is too hard to control the behavior of the children, so we do an observational study instead.  Children 1-5 choose to play an average of more than 10 hours per week from age 3 to age 6, while Children 6-10 play less than 10 hours per week.  Compute the difference in means from the resulting observational data.
- Compare your answer in (h) to the true ATE.  Intuitively, what causes the difference?

### Brandon's Response to #4:

#### #1


```python
# Individual treatment effect for each of the ten children & explanation

d["tau"] = d["y1"] - d["y0"]
print(d["tau"])

print("The mean of the vector tau (the ATE) is: " + str(d["tau"].mean()))
```

    0    0.0
    1    0.5
    2    0.0
    3    0.0
    4   -0.9
    5    0.0
    6    0.0
    7    0.0
    8    0.0
    9    0.0
    Name: tau, dtype: float64
    The mean of the vector tau (the ATE) is: -0.040000000000000015


The vector tau has values of the treatment effect for each of the ten children. That is, we subtract each childrens' potential outcome to control from their potential outcome to treatment to get their individual treatment effect.

#### #2 & #3

#### Single paragraph, tell a story that could explain this distribution of treatment effects; what might cause some children to have different treatment effects than others?

The distribution of treatment effects seen in tau could be explained mostly by chance (there is no sharp null, but close to it). For instance, some children could have been very tired before taking their second eye assessment (or even before their first eye assessment), making slight mistakes in their evaluation. In other words, some children may have made small mistakes in their eye assessment, so the noise (those that are not a Sharp Null) could be from random errors. Likewise, there could be errors in the instrumentation when measuring their visual acuity.

#### #4


```python
print("The true average treatment effect (ATE) of playing outside is: " + str(d["tau"].mean()))
```

    The true average treatment effect (ATE) of playing outside is: -0.040000000000000015


#### #5


```python
# odd to treatment & even to control 
odd = d[d["child"] % 2 == 1]
even = d[d["child"] % 2 == 0]

ate = odd["y1"].mean() - even["y0"].mean() 
print("The Average Treatment Effect under this new assignment is approximately: "+ str(ate))


```

    The Average Treatment Effect under this new assignment is approximately: -0.060000000000000164



```python
# odd to treatment & even to control 
oddDf = d.loc[[0, 2, 4, 6, 8], ["y1"]]
evenDf = d.loc[[1, 3, 5, 7, 9], ["y0"]]
    
treatmentEffect = list(np.array(oddDf["y1"]) - np.array(evenDf["y0"]))

def average(x):
    return sum(x)/len(x)

print("The Average Treatment Effect under this new assignment is approximately: "+ str(average(treatmentEffect)))
```

    The Average Treatment Effect under this new assignment is approximately: -0.06000000000000003


#### #6

The Average Treatment Effect (ATE) under the new assignment is slightly different than the true ATE (specifically a little smaller). This could be from random outliers in the data, simply random noise. There could be some other factor not considered which can cause this little difference, such as instrumentation error when measuring, etc.

#### #7


```python
answer = 2**10-2
print("There are " + str(answer) + " ways to split the children into a treatment versus a control group (assuming at least one person is always in the treatment group and at least one person is always in the control group)")
```

    There are 1022 ways to split the children into a treatment versus a control group (assuming at least one person is always in the treatment group and at least one person is always in the control group)


#### #8

I'm not too sure if y1 should be considered as it is the introduction of treatment; is it asking us to assume the treatment column (y1) does not exist since it is an observational study? I'll assume this isn't the case for the sake of this problem set:


```python
moreThanTenH = d[d["child"] <= 5]
lessThanTenH = d[d["child"] >= 6]
# print(moreThanTenH)
# print(lessThanTenH)

differenceInMeans = moreThanTenH["y1"].mean() - lessThanTenH["y0"].mean()
print("The difference in means from the resulting data is approximately: " + str(differenceInMeans))

```

    The difference in means from the resulting data is approximately: -0.43999999999999995


#### #9

It seems the average treatment effect (ATE) of #8 is different from the true ATE because the anomolies (child 2 & 5) are in one group while there are no anomilies in the second group. Perhaps it also has to do with the way in which people are grouped together--there is no random assignment to a treatment group and control group.

# 5. Randomization and Experiments 

Suppose that a reasearcher wants to investigate whether after-school math programs improve grades. The researcher randomly samples a group of students from an elementary school and then compare the grades between the group of students who are enrolled in an after-school math program to those who do not attend any such program. Is this an experiment or an observational study? Why? 

### Brandon's Respone to #5:

This is an observational study as there is no intervention done by the researcher, but only random sampling. The researcher randomly samples a group from an elementary school and compares those students who are in the after school program to those who are not--the researcher does not intervene to create variation. The researcher also assumes homogeneity (identical groups) between those who enroll in after-school care and those who do not. Since the people in those two groups were not randomly assigned to one of the two groups, there are many ways those groups may differ (e.g., socioeconomic status, parental involvement, etc.). To make such a causal claim, an experiment is needed for further investigation.

An experiment not only requires an intervention by the researcher to create some type of variation, but also requires the systematic tracking of that variation. Also, an experiment requires randomly assigning children to a treatment group (after-school math program) and a control group (no math after-school program) to move toward homogeneity between the groups (the noise--heterogeneity--would be ideally spread across equally for both the treatment and control group if participants are randomly assigned to the treatment and control group).

On the whole, the study outlined in the prompt is merely an observational study as there is no intervention from the researcher to create variation--the researcher is merely observing variables that covary together. Additionally, there is no guarantee of homogeneity between those that go to after-school math programs and those that don't since children were not randomly assigned to one the two groups. 

# 6. Lotteries

A researcher wants to know how winning large sums of money in a national lottery affect people's views about the estate tax. The research interviews a random sample of adults and compares the attitudes of those who report winning more than $10,000 in the lottery to those who claim to have won little or nothing. The researcher reasons that the lottery choose winners at random, and therefore the amount that people report having won is random. 

- Critically evaluate this assumption. 
- Suppose the researcher were to restrict the sample to people who had played the lottery at least once during the past year. Is it safe to assume that the potential outcomes of those who report winning more than $10,000 are identical, in expectation, to those who report winning little or nothing? 

**Clarifications**

1. Please think of the outcome variable as an individual’s answer to the survey question “Are you in favor of raising the estate tax rate in the United States?”
2. The hint about potential outcomes could be rewritten as follows: Do you think those who won the lottery would have had the same views about the estate tax if they had actually not won it as those who actually did not win it? (That is, is $E[Y_{i}0|D=1] = E[Y_{i}0|D=0]$, comparing what would have happened to the actual winners, the $|D=1$ part, if they had not won, the $Y_{i}(0)$ part, and what actually happened to those who did not win, the $Y_{i}(0)|D=0$ part.) In general, it is just another way of asking, "are those who win the lottery and those who have not won the lottery comparable?"
3. Assume lottery winnings are always observed accurately and there are no concerns about under- or over-reporting.


### Brandon's Response to #6

The researchers are assuming all the people who play the lottery have an equal chance of winning, when, in fact, that's not the case. For example, while there are people who buy one lottery ticket, there are many others who buy multiple lottery tickets. Some people may buy a ticket once a year, while others buy a ticket once a day. Even those who choose to buy a lottery ticket in the first place can be seen as different from someone who doesn't, which could have an effect on their views about the estate tax. 

# 7. Inmates and Reading
A researcher studying 1,000 prison inmates noticed that prisoners who spend at least 3 hours per day reading are less likely to have violent encounters with prison staff. The researcher recommends that all prisoners be required to spend at least three hours reading each day. Let $d_{i}$ be 0 when prisoners read less than three hours each day and 1 when they read more than three hours each day. Let $Y_{i}(0)$ be each prisoner's PO of violent encounters with prison staff when reading less than three hours per day, and let $Y_{i}(1)$ be their PO of violent encounters when reading more than three hours per day. 

In this study, nature has assigned a particular realization of $d_{i}$ to each subject. When assessing this study, why might one be hesitant to assume that ${E[Y_{i}(0)|D_{i}=0] = E[Y_{i}(0)|D_{i}=1]}$ and $E{[Y_{i}(1)|D_{i}=0] = E[Y_{i}(1)|D_{i}=1]}$? In your answer, give some intuitive explanation in English for what the mathematical expressions mean.


### Brandon's Response to #7

Expecting the potential outcome of violent encounters with prison staff to go down when someone who currently reads less than three hours begins to read more than three hours is ill-founded since this conclusion rests upon the assumption of the two groups being homogeneous. The two groups could differ significantly, which would lead them to read more or less. For instance, there could be a different causal factor which explains why those inmates who read more than three hours have lower violent encouters. Perhaps inmates who score higher on the personality traits of Conscientiousness and Openness to Experience tend to have greater self control (small number of violent encoutners with staff) and curiousity of intellectual growth (reads more than three hours a day). Put differently, the two groups could differ on personality traits which could lead the potential outcome of having less violent encounters, etc.

In other words, we cannot assume the two groups are identical as there is unaccounted heterogeneity between the two groups.
