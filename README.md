## North Chicago Township Home Sale Analysis

### I.	Executive Summary
This project focuses on investigating the determinants of single-family home sale prices in North Chicago, particularly within affluent neighborhoods like Gold Coast, Magnificent Miles, and Lincoln Park, challenging the traditional emphasis on location by employing a linear regression model on various continuous predictors. The study aims to provide in-depth insights beyond the common mantra of "location, location, location," benefiting prospective homebuyers, sellers, and real estate professionals in understanding the nuanced dynamics affecting property prices in the North Chicago Township.

## II.	Data Description

Data Definition	Data Source 	Data Size	Rows/Columns
North Chicago Township Home Sale data	Cook County Assessor 2021 
82 KB	(404, 28)
This data consists of 403 single-family homes sold in the North Chicago township of Cook County between 2018 and 2020.  

III.	Exploratively Data Analysis
Data Distribution
We initially examined the distribution of Sale Price. By drawing a histogram with a bin width of 100 and a horizontal boxplot, it revealed a right-skewed distribution, indicative of a non-normal pattern. Instead, it looks more like a Gamma distribution. Since a linear regression model assumes a normal distribution for the error, we will transform the Sale Price to produce a seemingly normal distribution. Many transformations serve this purpose. For us, we will apply the natural logarithm transformation to Sale Price. 

With this in mind, we will be checking whether the Log Sale Price is normally distributed.  The Normal Q-Q plot, a visual diagnostic tool, did not display the anticipated straight-line pattern indicative of a normal distribution.

Subsequently, formal statistical tests, the Shapiro-Wilks test, and the Anderson-Darling test, were employed for quantitative evaluation. The Shapiro-Wilks test yielded a p-value of 0.9655885, which is considerably high, indicating a lack of evidence to reject the null hypothesis that the data follows a normal distribution. However, the Anderson-Darling test statistic of 4.041554216991244 exceeded the critical values of [0.57, 0.65, 0.779, 0.909, 1.081], leading to the rejection of the null hypothesis.
Shapiro-Wilks test p-value: 0.9655885
Anderson-Darling test statistic: 4.041554216991244
Anderson-Darling test critical values: [0.57  0.65  0.779 0.909 1.081]

However, Even if the Log Sale Price does not perfectly follow a normal distribution, it can still be acceptable for linear regression. Linear regression is robust, and some deviations from normality are often tolerable, especially with larger sample sizes. It's important to consider the overall context, the assumptions of linear regression, and whether the model's performance is satisfactory.

Multiple Linear Regression Model
Next, we transitioned to building a multiple linear regression model incorporating eight predictors, including Age, Bedrooms, Building Square Feet, Full Baths, Garage Size, Half Baths, Land Acre, and Tract Median Income. The model, inclusive of an Intercept term, revealed a commendable Coefficient of Determination (R-squared) of 0.8384, signifying a strong explanatory power.

Diving deeper into the model's intricacies, we examined the regression coefficients along with their standard errors and 95% confidence intervals. The coefficients illuminated the impact of each predictor on Log Sale Price. Notably, variables such as Land Acre and Building Square Feet demonstrated substantial influence, signifying that a larger living space and property lot contribute positively to the property's perceived value. On the other hand, Age exhibited a negative impact, suggesting that older properties, on average, tend to command lower prices in the North Chicago Township real estate market.

These nuanced findings highlight the complex interplay of various factors influencing property prices. Stakeholders, such as homebuyers, sellers, and real estate professionals, can leverage this information to make more informed decisions and refine their strategies in the dynamic North Chicago Township real estate landscape.

Sale Price Prediction
Finally, we will predict the Sale Price of a single-family home whose features are at the median of all predictors. The calculated predicted Sale Price was found to be $886.03. This estimate was derived using a multiple linear regression model, taking into account the medians of all predictor variables. Furthermore, the 95% confidence interval for the predicted Log Sale Price was computed, and upon exponentiation, the confidence interval for Sale Price was obtained. The resulting interval, ranging from $844.95 to $929.10, provides a measure of uncertainty around the predicted Sale Price.

To conclude this project, we will calculate the Shapley values for the predictors included in our regression model.  The Shapley values, representing the average contribution of each predictor to all possible model combinations, were calculated. The sum of Shapley values was found to be 0.8384, indicating the collective influence of all predictors in explaining the variability in sale prices.

Analyzing the individual Shapley values for each predictor, it becomes evident that "Building Square Feet" emerges as the most influential feature, making the highest contribution to predicting sale prices. Following closely are "Land Acre" and "Full Baths," contributing significantly to the overall model. Other predictors, such as "Bedrooms," "Garage Size," "Age," "Half Baths," and "Tract Median Income," also play roles in determining sale prices, albeit to varying degrees.

 




IV.	Insight and Conclusion
The exploration of the North Chicago Township real estate dataset has uncovered pivotal insights guiding our understanding of property prices in this dynamic market:

Initial analyses revealed a right-skewed distribution of Sale Prices, prompting a transformation to Log Sale Price for alignment with regression model assumptions.
The ensuing multiple linear regression model, incorporating eight predictors, demonstrated robust performance with a high Coefficient of Determination (R-squared) of 0.8384.
Building Square Feet emerged as the most influential predictor, followed by Land Acre and Full Baths, as confirmed by both regression coefficients and Shapley values.
Utilizing median values, we predicted a Sale Price of $886.03, supported by a confidence interval providing a nuanced range for anticipated property values.
These findings collectively empower stakeholders with actionable insights, whether optimizing pricing strategies, understanding predictor impacts, or navigating the intricacies of the North Chicago Township real estate market.
![image](https://github.com/kevinkooo/North-Chicago-Township-Home-sale-Analysis/assets/156154849/ca9cb701-9222-4af0-b19a-1f13fbfd2278)
