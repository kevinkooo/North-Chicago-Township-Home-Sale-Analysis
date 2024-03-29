## North Chicago Township Home Sale Analysis

### I.	Executive Summary
This project focuses on investigating the determinants of single-family home sale prices in North Chicago, particularly within affluent neighborhoods like Gold Coast, Magnificent Miles, and Lincoln Park, challenging the traditional emphasis on location by employing a linear regression model on various continuous predictors. The study aims to provide in-depth insights beyond the common mantra of "location, location, location," benefiting prospective homebuyers, sellers, and real estate professionals in understanding the nuanced dynamics affecting property prices in the North Chicago Township.

### II.	Data Description

<img width="814" alt="image" src="https://github.com/kevinkooo/North-Chicago-Township-Home-sale-Analysis/assets/156154849/d8076031-13e9-4ae2-9efc-5c7f256b029d">

This data consists of 403 single-family homes sold in the North Chicago township of Cook County between 2018 and 2020.  

### III.	Exploratively Data Analysis
- Initial analyses revealed a right-skewed distribution of Sale Prices, prompting a transformation to Log Sale Price for alignment with regression model assumptions.
<img width="600" alt="image" src="https://github.com/kevinkooo/North-Chicago-Township-Home-sale-Analysis/assets/156154849/57e765ed-a2b1-4724-9497-dff62984ec8f">

- The ensuing multiple linear regression model, incorporating eight predictors, demonstrated robust performance with a high Coefficient of Determination (R-squared) of 0.8384.
<img width="600" alt="image" src="https://github.com/kevinkooo/North-Chicago-Township-Home-sale-Analysis/assets/156154849/bf2fb9d8-fbee-48b5-81fe-1727178277de">

- Building Square Feet emerged as the most influential predictor, followed by Land Acre and Full Baths, as confirmed by both regression coefficients and Shapley values.
<img width="450" alt="image" src="https://github.com/kevinkooo/North-Chicago-Township-Home-sale-Analysis/assets/156154849/427ea441-07cd-42d4-9455-94ad015a4ad1">

- Utilizing median values, we predicted a Sale Price of $886.03, supported by a confidence interval providing a nuanced range for anticipated property values.

These findings collectively empower stakeholders with actionable insights, whether optimizing pricing strategies, understanding predictor impacts, or navigating the intricacies of the North Chicago Township real estate market.
