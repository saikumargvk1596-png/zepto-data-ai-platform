# EDA Report

Missing percentages:

deck           77.216611
age            19.865320
embarked        0.224467
embark_town     0.224467

Fare mean=32.0967, median=14.4542, mode=8.0500; distribution=right-skewed.

Strongest absolute correlations:
1. pclass vs fare = -0.5482
2. sibsp vs parch = 0.4145

## Chart interpretations
### Survival by class and sex
The chart jointly compares class and sex, showing how survival varies across both dimensions. This is more informative than examining either variable independently.

### Fare by survival and class
The chart compares fare distributions across survival outcomes while retaining passenger class. It shows the association between ticket price/class and survival.

### Age vs fare
This chart combines age, fare, survival and class to show where different outcome groups occur in feature space.

### Embarkation and sex
This chart compares survival rates across embarkation points while preserving sex as a second grouping variable.
