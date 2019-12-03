---
title: "Logistic Regression Model"
subtitle: From distributions to linear models - Part 2
author: "Olivia Lin"
date: 2 December 2018
layout: default
tags: modelling
---
# Logistic Regression Model
## From distributions to linear models - Part 2

<div class="block">
	<center>
		<img src="{{ site.baseurl }}/img/tutheadermixed.png" alt="Img">
	</center>
</div>



## Tutorial Aims

#### <a href="#one"> 1. Cleaning and preparing data</a>

#### <a href="#two"> 2. Splitting data into training and test sets</a>

#### <a href="#three"> 3. Handling class imbalance</a>

#### <a href="#four"> 4. Building a logistic regression model</a>


This tutorial is the continuation of the tutorial on linear modelling, you can check out <a href = "https://ourcodingclub.github.io/2017/02/28/modelling.html#distributions" target="_blank">"From distributions to linear models"</a> here. if you are still wondering what types of data distribution there are and how a general linear model is built!

In the first part of the tutorial, we've learned how to create a general linear model with a Gaussian data distribution, that is when the data are normally distributed and homoscedastic. We've also slightly looked at how to run a model with a Poisson or binomial distribution. However, that is not enough to give us a more accurate results. Here, we'll learn about 1) cleaning the dataset ready for modelling, 2) handling class imbalance, 3) building the logistic regression model and at last 4) predicting the accuracy of the test data.


## What is a logistic regression model and why?

In linear regression, the response variable (y) can only be continuous. However, in some cases the response variable is categorial or discrete e.g. presence vs absence, true vs false, alive vs death. Therefore, we can't always use linear regression to model our output when the response variable is a binary categorial variable. This is where a logistic regression model comes into play. It gives us a mathematical equation to determine the probability of an event taken place.

<center><img src="{{ site.baseurl }}/image/image1.png" alt="Img" style="width: 800px;"/></center>
<center>Source: <a href="https://medium.com/@maithilijoshi6/a-comparison-between-linear-and-logistic-regression-8aea40867e2d" target="blank">Medium</a></center>



The use of statistic models to predict the likely occurrence of an event is gaining importance in __wildlife management__ and __conservation planning__. Logistic regression in particular often helps ecologists to model the presence and absence of a species in the survey sites in respond to a set of environmental variables. It is also often used to predict a species' response to environmental perturbations.

Here, we'll examine how the different environmental variables affect the presence of frogs and predict the probability of frogs presence. The frogs' presence here is a binomial variable because the response is either "present" or "absent".

You can get all of the resources for this tutorial from <a href="https://github.com/oliviapyui/logitrepo" target="blank">this GitHub repository</a>. Click `clone and download` and download the repository as a zip file, then unzip it.

<a name="one"></a>

## 1. Cleaning and preparing the dataset

Open a new R script in RStudio and add in some details (e.g. title of the script, date, your name, your contact details etc.). You might also want to set the working directory of this script by using `setwd()`. We're going to use the packages `dplyr` and `caret` for handling class imbalance later. If you're not familiar with `dplyr`, you can check out the tutorial <a href = "https://ourcodingclub.github.io/2017/01/06/data-manip-intro.html" target="_blank">"Intro to Data Manipulation"</a> . It is a very powerful package with various packages in built for tidying data.

<a id="Acode01" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code01" markdown="1">

```r
# Please install these packages first if you don't already have them
# install.packages("DAAG")
# install.packages("dplyr")
# install.packages("caret")
# install.packages("ggplot2")
# install.packages("InformationValue")

# Load libraries ----

library(DAAG)  # for the "frogs" dataset
library(dplyr)  # for data manipulation
library(caret)  # for data splitting
library(ggplot2)  # for plotting graph
library(InformationValue)  # for finding optimal prob cutoff
```

The dataset we're using is `frogs` which is already in the `DAAG` package. We can just directly call for `frogs` in our script after loading the `DAAG` library.

Let's use `glimpse()` from `dplyr` to explore the dataset and see if there are any structural errors (e.g. if there are any misspelled names in the column). Although it does a similar thing as `str()` does, `glimpse()` makes viewing every column possible and more observations can be shown.

<a id="Acode02" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code02" markdown="1">

```r
# Explore the data ----

glimpse(frogs)
```

Data cleaning is an essential step in machine learning (i.e. the use of statistical models and algorithms) which identifies and fixes incorrect, inaccurate, irrelevant and incomplete data. It's good to keep in mind that "_better data beats fancier algorithms_". Missing values can mess up the model, so let's use the base R function `sapply()` to show us the count of NA values in each column.

`sapply()` passes an argument (i.e. `sum(is.na(x))`) to all elements of the input and returns the output as a vector or matrix. `lapply()` also does a similar job by returning the output as a list instead. One way to easily remember the difference between these two functions is that `lappy` starts with an 'l', as it suggests 'list'.

<a id="Acode03" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code03" markdown="1">

```r
# 1. Prepare and clean the data ----

# Check if there is any missing value in each column
sapply(frogs, function(x)sum(is.na(x)))  # Summing the count of NA values in each column

lapply(frogs, function(x)sum(is.na(x)))
# Here you can explore lappy(), which does the same thing as sapply
# but returns output as a list

# No missing values are found, so we can proceed to the next step
```

There are many ways we can use to remove the missing values. Alternatively, we can use `complete.cases[frogs, ]` to create a copy of the dataset without the missing values or `na.omit(frogs)` to simply ignore all the NA values. However, we cannot explore how many missing values there are in the dataset using these two functions. While removing NA values also means removing information, these two functions are not very desirable when working on dataset with a small amount of observations. Getting an idea of how many missing values there are in each column is a good practice. It helps us to decide if we want to simply remove them or impute values (e.g. 0, mode or mean) to them based on our understanding of the data, experimental design and algorithm.

A matrix is returned which tells us that there are no missing values in each of the columns. We can now proceed to the next step.

<div class="bs-callout-grey" markdown="1">

#### Creating section titles in an R script

Adding 4 dashes `----` at the end of a comment will make it as a section title. We can use it to help us outline the script and navigate to the section of the script we want.

</div>

Onto our next step of data cleaning, we're going to discard the irrelevant data and change the data types for the model. "northing" and "easting" are to be discarded since they are reference points on a map. We'll be using `select()` from the package `dplyr` to select the columns we want.

The response (dependent) variable and explanatory (independent) variables are numeric. Variables with internal ordering can be size, age, year etc.. Since there are internal orderings of the explanatory variables, we don't have to convert them to factor variables. However, the response variable in logistic regression must be categorical, we'll have to change it to a factor with 2 levels.

<a id="Acode04" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code04" markdown="1">

```r
# Select only the columns we need
frogs <- select(frogs, -c(northing, easting))  # select all except northing and easting

# Change frogs' presence to factor variables
frogs$pres.abs <- as.factor(frogs$pres.abs)

# pres.abs is now factor variable presented as either 1 or 0
```

<a name="two"></a>

## 2. Splitting data into training and test sets

In using machine learning algorithm, we'll usually split the data into a training set and a test set. The training set will be used to fit the model while the test set is to be used for testing the model, i.e. evaluating the accuracy of the model.

To split the data, we'll use `createDataPartition()` from the package `caret`. `set.seed()` is a pseudorandom number generator. Setting the seed can give us reproducible results so you'll create the same training set and same statistical results as I do in this tutorial! In other words, if you change the number, you'll get a different results. You can try replacing "100" at the end of the tutorial to see the difference!

<a id="Acode05" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code05" markdown="1">

```r
# 2. Data splitting ----

# Prepare Training and test data

# Set the seed to give us reproducible results
set.seed(100)  # 100 is the starting point of the generation of a sequence of random numbers

# Choose the rows for training data
train_data_index <-
  createDataPartition(frogs$pres.abs,
                      p = 0.7,  # 70% of the rows go to training
                      list = F)  # return the results as a matrix but not a list

# Partitioning the data into two sets
train_data <- frogs[train_data_index, ]  # the chosen 70% of the rows as training data
test_data <- frogs[-train_data_index, ]  # choose the remaining rows as test data
```

You can see that there are 150 observations for `train_data` in the "Environment" panel on the right hand side, which is about 70% of the 212 observations in `frogs`.

<a name="three"></a>

## 3. Handling class imbalance

The class balance or called class bias, is the uneven proportion of "0" and "1" in the response variable. This affects the statistical results and prediction capabilities of a logistic regression model. There are still debates on whether the data should be balanced in the ecological context as sampling ratio reflecting the true representation of the real context is often hard to achieve within the practical sampling framework. Nonetheless, a balanced class is still commonly preferred by ecologists in fitting a logistic regression model. Also, machine learning algorithms work best when the classes are in proportionate ratio.

The use of `createDataPartition` will preserve the proportion of the classes in the original `frogs` data set. Let's check the proportion of "0"s and "1"s in the train data.

<a id="Acode06" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code06" markdown="1">

```r
# 3. Class imbalance ----

# Check the proportion of "0"s and "1"s in the training data
table(train_data$pres.abs)

# Find the proportion of "0"s and "1"s
table(frogs$pres.abs)

# the absence and presence response are split into approximately 1.7:1 ratio
```

We can see that there are 94 "0"s and 56 "1"s which are in an approximately 1.7:1 ratio. This is called class imbalance. So how can we handle the imbalance?

<div class="bs-callout-blue" markdown="1">

#### Dealing with imbalanced class

There are mainly two ways to handle class imbalance: upsampling and downsampling.

1. __Upsampling__
As the name suggests, it is the process of randomly sampling the minority class repeatedly until its size is the same as the majority class i.e. __oversampling the minority class__.

2. __Downsampling__
Downsampling is the opposite of upsampling. It is the process of randomly sampling fewer observations in the majority class so it reaches the same size as the minority class i.e. __undersampling the majority class__.

__So which one should we use?__
The decision to do upsampling or downsampling or even other methods depends on the sampling method in the experimental design and our understanding of the real context.

</div>

Here, __upsampling__ is preferred as it implies no loss of information and we don't have a large dataset. We'll use `upSample()` from the package `caret`. If you want to perform downsampling, you can simply replace `upSample()` with `downSample()`.

<a id="Acode07" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code07" markdown="1">

```r
# Upsampling ----

set.seed(100)  # Set the seed at 100 again

# Select the predictor variables columns
x_col <- select(train_data, -c(pres.abs))  # select everthing except "pres.abs" column

# Upsize the training sample
up_train <- upSample(x = x_col,  # the predictor variables
                     y = train_data$pres.abs)  # the factor variable with class

# Check the proportion of "0"s and "1"s in the upsized training data
table(up_train$Class)

# the "1"s are upsized to 94, equal ratio achieved
```

We now get an equal ratio of the class. The train data set is now ready for fitting into a logistic regression model!


<a name="four"></a>

## 4. Logistic regression model

### A. Visualising our data

Before we actually build our model, let's visualise how our data look like using `ggplot2`.

<a id="Acode08" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code08" markdown="1">

```r
# 3. The logistic regression model ----

# Let's visualise our data first ----
(plot_1 <- 
    ggplot(up_train, 
           aes(x = altitude + distance + NoOfPools + NoOfSites + 
                   avrain + meanmin + meanmax, 
               y = as.numeric(as.character(Class)))) +  # change "Class" to numeric values
   geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +  # plot the classes of each observation
   geom_smooth(method = "glm",  # add the curve
               method.args = list(family = "binomial"),
               colour = "paleturquoise3",  # colour of the curve
               fill = "azure3",
               size = 1) +  # colour of the SE area
   theme_bw() +
   theme(panel.grid.minor = element_blank(),
         panel.grid.major.x= element_blank(),
         plot.margin = unit(c(1,1,1,1), units = , "cm"),
         plot.title = element_text(face = "bold", size = 10, hjust = 0),
         axis.text = element_text(size = 8),
         axis.title = element_text(size = 8)) +
   labs(title = "Logistic Regression Model 1\n",  #  "\n" indicates where the space is added
        y = "Probability of the presence of frogs\n",
        x = "\naltitude + distance + NoOfPools + NoOfSites+
        \navrain + meanmin + meanmax") +
  ggsave("image/plot_1.png", width = 5, height = 4, dpi = 800))
```

<center><img src="{{ site.baseurl }}/image/plot_1.png" alt="Img" style="width: 700px; height: 550px"/></center>
Figure caption

The plot shows how the probability of the presence of frogs varies with all the environmental variables. The curve doesn't touch the top which tells us that the probability of presence is quite low. Also, the direction of the curve implies that the probability of the frogs' presence decreases with increasing values of the explanatory variables.


Fitting a logistic regression model is very close to what we did in the previous tutorial for linear models, except that we're replacing `lm()` with `glm()` as logistic regression is a type of generalised linear model. We also add a family argument `family = "binomial"` where we specify the data distribution.

<a id="Acode09" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code09" markdown="1">

```r
# Using generalised linear model
model <- glm(Class ~. ,  # "." indicated all other variables
             family = "binomial", data = up_train)
```

Before we interpret results from the model, it's important to check if our data meet the 3 assumptions of a logistic regression model!

### B. Checking for assumptions

#### __Assumption 1: Binary outcome structure__

This simply means that the response variable should be binary with two levels only. In this dataset, it should only be either "1" or "0" which stand for "presence" or "absence" respectively.

#### __Assumption 2: No multicollinearity among predictor variables__

This means that the independent or predictor variables should not be highly correlated to one another. Multicollinearity has a significant effect on the fit of the model as it will reduce the precision of the estimates.

To check for multicollinearity, we can use the __variance inflation factor__ (VIF) which quantifies the extent of correlation between each predictor in a model. A VIF of 1 means that the predictor is not correlated to the other predictors at all. The higher the value, the greater the correlation of the predictor with the other predictors. A high VIF values is thus not desirable as it indicates that the contribution of the predictor to the model is difficult to be assessed. VIF higher than 5 is commonly regarded as high.

To calculate VIF, we can use the `vif()` function.

<a id="Acode10" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code10" markdown="1">

```r
# Checking assumptions ----

# Assumption 2: Multicollinearity
vif(model)

# The vif values should be < 5
```
```r
> vif(model)
altitude  distance NoOfPools NoOfSites    avrain   meanmin   meanmax
934.6700    1.3953    1.2581    1.4441   18.6730   35.3180 1083.7000
```
From the results obtained as shown above, we can see that the VIF values of "altitude", "avrain", "meanmin" and "meanmax" variables are all too high. Only that of "distance", "NoOfPools" and "NoOfSites" are smaller than 5. `model` doesn't meet the assumption of no multicollinearity between predictor variables.

This is probably because temperature generally decreases with increasing altitude, so "altitude", "meanmin" and "meanmax" are highly correlated. Let's remove these variables from our old model to minimise overfitting and check for multicollinearity again.

<a id="Acode11" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code11" markdown="1">

```r
# Adjust our model
model_new <- glm(Class ~ distance + NoOfPools + NoOfSites + avrain,
                     family = "binomial", data = up_train)

# Compute the VIF values to check for multicollinearity again
vif(model_new)
```
```r
> vif(model_new)
 distance NoOfPools NoOfSites    avrain
   1.2938    1.0953    1.4221    1.1808
```

We've now obtained VIF values all less than 5, which means that there are nearly no or little multicollinearity between the predictor variables. The assumption is now met for our new model.

#### __Assumption 3: Linearity__

This refers to the required linear relationship between the predictor variables and the link function (logit or log odds).

The simplest way of checking linearity is to plot the residuals vs fitted plot using `plot()`.

<a id="Acode12" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code12" markdown="1">

```r
# Assumption 3: Linearity ----
plot(model_new, which = 1)  # Call for the 1st plot
```
<center><img src="{{ site.baseurl }}/residuals.png" alt="Img"></center>

From the plot we should find no trend in the residuals. Since the red line is generally lying on the grey horizontal line, we can say that the assumption of linearity is met.

#### __Assumption 4: Observation independence__

The observations should be independent of each other i.e they should not come from repeated measurements or matched data. As we didn't collect the data, it's really hard to know if the observations are independent of each other. We can however use a __serial plot of residuals__ to observe if there is the presence of _autocorrelation_, which appears when there is a serial dependence in the measurement of observations. This plot allows us to detect the time trends in the residuals. If there is no autocorrelation, we'll expect to see _no tracking of the residuals_, i.e. the closer observations do not have similar values. The plot is expected to be highly zig-zagged.

<a id="Acode13" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code13" markdown="1">

```r
# Assumption 4: Independence ----
plot(model$residuals, type = "o")
```

+++ INSERT serial plot +++++

The plot is quite zig-zag, although some observations hold similar values. As we didn't collect the data, let's just accept this assumption for now.

#### __Assumption 5: Assumption of a large sample size__

Provided that we are looking at 7 environmental variables, 212 observations are actually not quite enough for us to build an accurate model. Yet, let's just assume this assumption is met for simplicity.


### C. Interpreting results

After ensuring that the assumptions are met and refitting our model, we can finally move on to look at the results the model gave us! We'll first be using `summary()` to obtain a basic idea how the model looks like. Then we'll conduct an ANOVA using `anova()` to confirm our understanding. Sometimes, if the sample size is not large enough, the results obtained can be quite different.

<a id="Acode14" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code14" markdown="1">

```r
# Interpreting the results ----

summary(model_new)
anova(model_new, test = "Chisq")
```
These are the results that we obtained.

```r
> summary(model_new)

Call:
glm(formula = Class ~ distance + NoOfPools + NoOfSites + avrain,
    family = "binomial", data = up_train)

Deviance Residuals:
    Min       1Q   Median       3Q      Max
-1.6440  -1.1267   0.2078   0.9507   2.7837

Coefficients:
              Estimate Std. Error z value Pr(>|z|)
(Intercept)  2.3841265  2.3574047   1.011 0.311856
distance    -0.0006743  0.0001789  -3.769 0.000164 ***
NoOfPools    0.0089039  0.0069443   1.282 0.199774
NoOfSites   -0.0835763  0.1056209  -0.791 0.428777
avrain      -0.0095366  0.0149643  -0.637 0.523935
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 260.62  on 187  degrees of freedom
Residual deviance: 224.03  on 183  degrees of freedom
AIC: 234.03

Number of Fisher Scoring iterations: 6

> anova(model_new, test = "Chisq")
Analysis of Deviance Table

Model: binomial, link: logit

Response: Class

Terms added sequentially (first to last)


          Df Deviance Resid. Df Resid. Dev       Pr(>Chi)
NULL                        187     260.62
distance   1   34.372       186     226.25 0.000000004552 ***
NoOfPools  1    1.423       185     224.83         0.2329
NoOfSites  1    0.389       184     224.44         0.5329
avrain     1    0.407       183     224.03         0.5234
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```
The summary output gives us this logistic equation:

p = exp(2.38 - 0.000674 x distance + 0.00890 x NoOfPools - 0.0836 x NoOfSites - 0.00954 x avrain)/{1 + exp(2.38 - 0.000674 x distance + 0.00890 x NoOfPools - 0.0836 x NoOfSites - 0.00954 x avrain)}

where p is the probability of the presence of the frogs.

However, we can see that only the p-value for the "distance" variable is smaller than 0.05, while others are all larger than 0.05, which means that only the relationship between "distance" and the frogs' presence is statistically significant. The very small p-value also tells us that there is a strong association of the distance to nearest extant population of frogs and probability of their presence.

The logistic equation can hence be considered as:

p = exp(2.38 - 0.000674 x distance)/{1 + exp(2.38 - 0.000674 x distance)}

The probability of the presence of frogs can then be computed at any given "distance to the nearest extant population".

Let's visualise our new model now and compare it with our first plot where we correlate the probability of frogs' presence with all the environmental variables.

<a id="Acode15" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code15" markdown="1">

```r
# Comparing the graph of distance only and the previous plot_1
(plot_2 <- 
    ggplot(up_train, 
           aes(x = distance, 
               y = as.numeric(as.character(Class)))) +  
    geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +  
    geom_smooth(method = "glm",  # add the curve
                method.args = list(family = "binomial"),
                colour = "paleturquoise3", 
                fill = "azure3",
                size = 1) +
    theme_bw() +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x= element_blank(),
          plot.margin = unit(c(1,1,1,1), units = , "cm"),
          plot.title = element_text(face = "bold", size = 10, hjust = 0),
          axis.text = element_text(size = 8),
          axis.title = element_text(size = 8)) +
    labs(title = "Logistic Regression Model 2\n",  
         y = "Probability of the presence of frogs\n",
         x = "\nDistance to the nearest extant population (m)") +
    ggsave("image/plot_2.png", width = 5, height = 4, dpi = 800))
```

<left><img src="{{ site.baseurl }}/image/plot_1.png" alt="Img" style="width: 700px; height: 550px"/></left><right><img src="{{ site.baseurl }}/image/plot_2.png" alt="Img" style="width: 700px; height: 550px"/></right>

We can see that the 2 plots look very similar, even when we take away all other environmental variables except "distance". This confirms our results that the "distance" variable is the most significant one.

### D. Making predictions on test dataset

Now we have the model to make predictions on the probability of the presence of frogs in the test data we created earlier on. Same for linear models, we use `predict()` to make predictions. However, if we use it on a logistic regression model, it will predict the log odds (i.e. the log values of probabilities) of the response variable which is not what we want. To obtain predicted values that lie within the 0 and 1 range, we'll use `predict()` and `plogis()` together. `plogis()` is doing the inverse logarithms where it converts the log odds into probabilities that lie within 0 and 1 range. Alternatively, we can still use `predict()` but adding the `type = "response"` argument in contrast to doing linear regression model prediction.

<a id="Acode16" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code16" markdown="1">

```r
# Predict the probabilities of presence on test data ----

# Add a new column for the predicted probabilities
test_data <- test_data %>%
  mutate(prob = plogis(predict(model_new, newdata = test_data)))

# alternatively, adding the response argument
test_data <- test_data %>%
  mutate(prob_2 = predict(model_new, newdata = test_data,
                        type = "response"))

# they gave the same results
```

### E. Model accuracy

So after doing the predictions, we now have a set of values (i.e. the probabilities of presence) ranging between 0 and 1. The higher the probability means that there is a higher chance of frogs being present. We have to classify these values into two classes, either "present" or "absent" from what the model predicted so that we can assess how much observations are correctly categorised by our model.

By convention, the probability cutoff is 0.5. Nonetheless, tuning the probability cutoff can improve the accuracy. The `optimalCutoff()` function from the package `InformationValue` helps us find the optimal cutoff to improve the prediction of 1’s, 0’s, both 1’s and 0’s and reduce the misclassification error. Lets compute the optimal score that minimises the misclassification error for our model.

<a id="Acode17" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code17" markdown="1">

```r
# Model accuracy ----

# Decide on optimal prediction probability cutoff ----

# The default cutoff prediction probability score is 0.5
# To find the optimal probability cutoff
opt_cut_off <- optimalCutoff(test_data$pres.abs, test_data$prob)

# the p cut-off that gives the minimum misclassification error = 0.519
```

After finding the optimal probability cutoff (0.519), let's categorise the observations based on the predicted probabilities. The predicted probabilities larger than 0.519 will be classed as "1" ("present"). Otherwise, they will be classed as "0" ("absent")

<a id="Acode18" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code18" markdown="1">

```r
# Categorise individuals into 2 classes based on their predicted probabilities
# So if probability of response > 0.519, it will be classified as an event (Present = 1)

# Add a new column for the predicted class
test_data <- test_data %>%
  mutate(predict_class = ifelse(prob > 0.519, 1, 0))
```

We can then calculate the accuracy of the model which is measured as the proportion of observations that have been correctly classified in the test data.

To change a factor variable into numeric values can be quite tricky. If we don't change the factor variable to characters first before converting it to numeric values, then the converted values can be quite funny and not what we want.

<a id="Acode19" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code19" markdown="1">

```r
# Change the pres.abs. to numeric values
test_data$pres.abs <- as.numeric(as.character(test_data$pres.abs))

# How much predicted presence match with the actual data in test data
accuracy <- mean(test_data$predict_class == test_data$pres.abs)

# accuracy of the model = 72.6%
```

The model accuracy is 72.6% which is not bad!

Let's plot the prediction graph to visualise our predictions on the test data!

<a id="Acode20" class="copy" name="copy_pre" href="#"> <i class="fa fa-clipboard"></i> Copy Contents </a><br>
<section id= "code20" markdown="1">

```r
(pred_plot <- ggplot(test_data, aes(x = distance, y = predict_class)) +
    geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +
    stat_smooth(method = "glm", 
                method.args = list(family = "binomial"),
                colour = "indianred",
                fill = "azure3",
                size = 1) +
    theme_bw() +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x= element_blank(),
          plot.margin = unit(c(1,1,1,1), units = , "cm"),
          plot.title = element_text(face = "bold", size = 10, hjust = 0),
          axis.text = element_text(size = 8),
          axis.title = element_text(size = 8)) +
    scale_y_continuous(limits = c(0,1)) +  # Set min and max y values at 0 and 1 respectively
    scale_x_continuous(limits = c(min(test_data$distance), max(test_data$distance))) +
    labs(title = "Predicting frogs' presence on test data\n",
         x = "\nDistance to nearest extant population (m)",  #  "\n" adds space above x-axis title
         y = "Probability of the presence of frogs\n") +
    ggsave("image/pred_plot.png", width = 5, height = 4, dpi = 800))
```

<center><img src="{{ site.baseurl }}/image/pred_plot.png" alt="Img" style="width: 700px; height: 550px"/></center>


<hr>

#### Check out our <a href="https://ourcodingclub.github.io/links/" target="_blank">Useful links</a> page where you can find loads of guides and cheatsheets.

#### If you have any questions about completing this tutorial, please contact us on ourcodingclub@gmail.com

#### <a href="INSERT_SURVEY_LINK" target="_blank">We would love to hear your feedback on the tutorial, whether you did it in the classroom or online!</a>

<ul class="social-icons">
	<li>
		<h3>
			<a href="https://twitter.com/our_codingclub" target="_blank">&nbsp;Follow our coding adventures on Twitter! <i class="fa fa-twitter"></i></a>
		</h3>
	</li>
</ul>

<h3>&nbsp; Related tutorials:</h3>

{% assign posts_thresh = 8 %}

<ul>
  {% assign related_post_count = 0 %}
  {% for post in site.posts %}
    {% if related_post_count == posts_thresh %}
      {% break %}
    {% endif %}
    {% for tag in post.tags %}
      {% if page.tags contains tag %}
        <li>
            <a href="{{ site.url }}{{ post.url }}">
	    &nbsp; - {{ post.title }}
            </a>
        </li>
        {% assign related_post_count = related_post_count | plus: 1 %}
        {% break %}
      {% endif %}
    {% endfor %}
  {% endfor %}
</ul>
<br>

### &nbsp;&nbsp;Subscribe to our mailing list:
<div class="container">
	<div class="block">
        <!-- subscribe form start -->
		<div class="form-group">
			<form action="https://getsimpleform.com/messages?form_api_token=de1ba2f2f947822946fb6e835437ec78" method="post">
			<div class="form-group">
				<input type='text' class="form-control" name='Email' placeholder="Email" required/>
			</div>
			<div>
                        	<button class="btn btn-default" type='submit'>Subscribe</button>
                    	</div>
                	</form>
		</div>
	</div>
</div>
