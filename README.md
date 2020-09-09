# Heart Disease Prediction Using Feature selection and Machine Learning Ensemble

## About Heart disease

Heart Disease (including Coronary Heart Disease, Hypertension, and Stroke) remains the No. 1 cause of death in the US.The Heart Disease and Stroke Statistics—2019 Update from the **American Heart Association** indicates that:
* 116.4 million, or 46% of US adults are estimated to have hypertension. These are findings related to the new 2017 Hypertension Clinical Practice Guidelines.
* On average, someone dies of CVD every 38 seconds. About 2,303 deaths from CVD each day, based on 2016 data.
* On average, someone dies of a stroke every 3.70 minutes. About 389.4 deaths from stroke each day, based on 2016 data.

## Project Overview

In this Project I will try to unleash useful insights using this heart disease datasets and will perform feature selection to build Soft Voting Ensemble model by combining the power of best performing machine learning algorithms.

This Project is divided into 13 major steps which are as follows:

1. [Data description](#data-desc)
2. [Importing Libraries & setting up environment](#imp-lib)
3. [Loading dataset](#data-load)
4. [Data Cleaning & Preprocessing](#data-prep)
5. [Exploratory Data Analysis](#data-eda)
6. [OUtlier Detection & Removal](#data-out)
7. [Training & Test Split](#data-train)
8. [Cross Validation](#cross-val)
9. [Model Building](#data-model)
10. [Model evaluation & comparison](#model-eval)<br>
11. [Feature Selection](#model-eval)<br>
12. [Model Evaluation](#model-inter)
13. [Conclusion](#data-conc)

## About Data 

This dataset consists of 11 features and a target variable. It has 6 nominal variables and 5 numeric variables. The detailed description of all the features are as follows:

**1. Age:** Patients Age in years (Numeric)<br>
**2. Sex:** Gender of patient (Male - 1, Female - 0) (Nominal)<br>
**3. Chest Pain Type:** Type of chest pain experienced by patient categorized into 1 typical, 2 typical angina, 3 non-anginal pain, 4 asymptomatic (Nominal)<br>
**4. resting bp s:** Level of blood pressure at resting mode in mm/HG (Numerical)<br>
**5. cholestrol:** Serum cholestrol in mg/dl (Numeric)<br>
**6. fasting blood sugar:** Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)<br>
**7. resting ecg:** Result of electrocardiogram while at rest are represented in 3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy (Nominal)<br>
**8. max heart rate:** Maximum heart rate achieved (Numeric)<br>
**9. exercise angina:** Angina induced by exercise 0 depicting NO 1 depicting Yes (Nominal)<br>
**10. oldpeak:** Exercise induced ST-depression in comparison with the state of rest (Numeric)<br>
**11. ST slope:** ST segment measured in terms of slope during peak exercise 0: Normal 1: Upsloping 2: Flat 3: Downsloping (Nominal)<br>

#### Target variable
**12. target:** It is the target variable which we have to predict 1 means patient is suffering from heart risk and 0 means patient is normal.

## Installations :
This project requires Python 3.x and the following Python libraries should be installed to get the project started:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/installing.html)
- [xgboost](https://xgboost.readthedocs.io/en/latest/build.html)

I also reccommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project which also include jupyter notebook to run and execute [IPython Notebook](http://ipython.org/notebook.html).

## Code :
Actual code to get started with the project is provided in two files one is,```heart-disease-classification.ipynb```

## Run :
In a terminal or command window, navigate to the top-level project directory PIMA_Indian_Diabetes/ (that contains this README) and run one of the following commands:

```ipython notebook heart-disease-classification.ipynb```
or

```jupyter notebook heart-disease-classification.ipynb```

This will open the Jupyter Notebook software and project file in your browser.

## Model Evaluation :
I have done model evaluation based on following sklearn metric.
1. [Cross Validation Score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
2. [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
3. [Plotting ROC-AUC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
4. [Plotting Precision recall Curve](https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used)
5. [Sensitivity and Specitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
6. [Classification Error](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
7. [Log Loss](https://www.kaggle.com/dansbecker/what-is-log-loss#:~:text=Log%20Loss%20is%20a%20slight,by%20understanding%20the%20likelihood%20function.)
8. [Mathew Correlation coefficient](https://www.kaggle.com/dansbecker/what-is-log-loss#:~:text=Log%20Loss%20is%20a%20slight,by%20understanding%20the%20likelihood%20function.)

