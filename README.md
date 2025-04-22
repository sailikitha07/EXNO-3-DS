## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df

```

![Screenshot 2025-04-22 143404](https://github.com/user-attachments/assets/3ea2e876-7cac-4a12-901c-f1f8f1698a29)

```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

![Screenshot 2025-04-22 143549](https://github.com/user-attachments/assets/72f79d40-df50-4134-a285-c9c06e82ba12)

```

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```

![Screenshot 2025-04-22 143725](https://github.com/user-attachments/assets/4e45142c-2cb0-4857-b3c7-0d91d98d6768)

```
le=LabelEncoder()
df=df.copy()
df['ord_2']=le.fit_transform(df['ord_2'])
df

```

![Screenshot 2025-04-22 144319](https://github.com/user-attachments/assets/2c372e08-5ded-4f71-9cf4-346f99199237)

```

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
df2 = df.copy()
ohe = OneHotEncoder(sparse_output=False) 
enc = pd.DataFrame(
ohe.fit_transform(df2[["nom_0"]]),
columns=ohe.get_feature_names_out(["nom_0"]),index=df2.index )
df2 = pd.concat([df2.drop("nom_0", axis=1), enc], axis=1)
df2

```

![Screenshot 2025-04-22 144523](https://github.com/user-attachments/assets/ae1bfb5a-53ef-4ddf-8e4f-45235e722a77)

```
 pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-22 144619](https://github.com/user-attachments/assets/ecea774a-baf0-4b64-a9ff-db63d5a96b8e)
```
 pip install --upgrade category_encoders
```
![Screenshot 2025-04-22 144658](https://github.com/user-attachments/assets/e05fa08a-b1ca-40a7-9198-0a05ad0c1403)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2025-04-22 144740](https://github.com/user-attachments/assets/fbfc7c10-bd0d-468f-b396-7cac0ec56ed9)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```
![Screenshot 2025-04-22 144824](https://github.com/user-attachments/assets/28a7be8a-5ac7-46b8-bf60-fc0fafb2f301)

```
df.skew()
```
![Screenshot 2025-04-22 144923](https://github.com/user-attachments/assets/1560d46e-126e-4a8d-8e98-cc5acdcbe81e)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 145002](https://github.com/user-attachments/assets/13e1ca07-a010-4439-8bbf-91a5e61ab002)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-22 145038](https://github.com/user-attachments/assets/c0593600-f80b-4472-b722-8bd96e37ba64)

```
 np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 145131](https://github.com/user-attachments/assets/408f4726-8576-4ff9-9428-22588245d4ea)

```
 np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-22 145208](https://github.com/user-attachments/assets/673c074d-86a4-4360-88fc-c9ec2a5cc99b)

```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![Screenshot 2025-04-22 145251](https://github.com/user-attachments/assets/52a0f4b4-358f-43dc-9ba3-a51940fcc50b)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

```
![Screenshot 2025-04-22 150559](https://github.com/user-attachments/assets/66e37571-d814-472f-b8a0-42308b0e3c0f)

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-22 151037](https://github.com/user-attachments/assets/fa79f97c-bd0b-493a-a9d4-c2bbcaaa5675)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-22 151151](https://github.com/user-attachments/assets/e0411578-14cd-41c0-86fc-5cefde718611)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
![Screenshot 2025-04-22 151405](https://github.com/user-attachments/assets/c0484bff-9a90-4a64-b3ae-3d5b17a2c636)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
![Screenshot 2025-04-22 151546](https://github.com/user-attachments/assets/fc5a78e4-e75e-4039-a56b-f7f0a398b6be)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```
![Screenshot 2025-04-22 151719](https://github.com/user-attachments/assets/c4624463-326b-4927-aecc-7f1f23983240)

# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file
 was performed successfully.

       
