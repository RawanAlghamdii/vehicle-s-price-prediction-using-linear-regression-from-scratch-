import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

#Load the dataset
df = pd.read_csv(r'C:\Users\96653\Downloads\archive (8)\vehicles.csv')

## clean the Data ##
#1- removing unimportant columns
df.drop(columns=["county","id","url","region_url","VIN","image_url","description","posting_date","lat","long"],
         inplace=True)
print(df.isnull().sum())

#2- handle numerical missing values by filling the null values with the Mean
PMean= df.price.mean()
print("The average price is:" ,PMean )
df.fillna({'price': PMean},inplace=True)
YMean=df.year.mean()
OMean=df.odometer.mean()
df.fillna({'year': YMean,
              'odometer': OMean },inplace=True)
print(df.head(10))

#3- handling categorical missing values
print(df.info())
num_data=df.select_dtypes(["int64","float64"])
cat_data=df.select_dtypes(["object"])
#cat_data=df[['region','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','size','type','paint_color','state']]
for col in cat_data:
    le = LabelEncoder()
    cat_data[col]=le.fit_transform(cat_data[col]) #imtute each column and restore it

#concat cat and num dataframes 
df = pd.concat([num_data,cat_data], axis=1)
print(df.isnull().sum())
# change all the values datatype info float 
df=df.astype(float)
print(df.head(10))

#4- remove outliers 
#its shows like the region , model have outliers
#for col in df:
   # sns.barplot(df[col]) 
   # plt.show()

#Norlization
scalar = MinMaxScaler()
df = pd.DataFrame(scalar.fit_transform(df), columns = df.columns)
print(df.head(10))
# See the relations between the features 
plt.figure(figsize=(20,15))
sns.heatmap(df.corr() , annot=True , linewidths=2)
plt.show()

#Outlier Detection and Removal using the IQR Method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
# Define upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]


## Training ##
#split the data into train and test 
Y= df["price"]
X= df.drop("price", axis=1)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
def cost_function (theta,x,y) :
    m = x.shape[0]
    y_predict =  np.dot(x,theta.T)
    cost = (1/m) * np.sum(np.square(y - y_predict))
    return cost

def update_parameters (theta,x,y,lr):
    m = x.shape[0]
    y_predict =  np.dot(x,theta.T)
    error_diff = (y - y_predict)
    dj = (-2/m)*np.dot(x.T, error_diff)
    new_theta = theta-lr*dj
    return new_theta

def gradient_descent (theta,x,y,iterationNum,lr):
    costs_record=[]
    m = x.shape[0]
    i=0

    for i in range(iterationNum):
        cost = cost_function(theta,x,y)
        costs_record.append(cost)
        theta = update_parameters(theta,x,y,lr)

    return theta, costs_record

theta = np.zeros(x_train.shape[1])
iteration_num= 3
leraning_rate = 0.1
new_thetas, costs_list = gradient_descent(theta, x_train, y_train, iteration_num, leraning_rate)

print("cost record is:" , costs_list)
print("\n")
print("final weights:" , new_thetas)


# plot the cost function
plt.plot( list(range(iteration_num)), costs_list , '-b')
plt.show()
sns.regplot(x= list(range(iteration_num)), y=costs_list, ci=None , color= 'black')
plt.show()


## Testing ##
y_predict = np.dot(x_test, new_thetas)
y_predict.shape
y_test.shape
# calculates MSE
from sklearn.metrics import mean_squared_error as MSE
MSEerror = MSE(y_test,y_predict)
print("MSE equals", MSEerror)

## find the most significat features ##
from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest ( mutual_info_regression, k=1)
selector.fit(x_train,y_train)
cols=selector.get_support(indices=True)

print("selected features",1)
print(df.columns[cols])
