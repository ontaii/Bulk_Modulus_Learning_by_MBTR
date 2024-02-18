#Reading cifs, Converting cifs to MBTR Matrix (csv)
from ase.io import read,write
from dscribe.descriptors import MBTR
import os; import pandas as pd

elements = [
    "Li", "B", "O", "F", "Na", "Mg", "Al", "Si",
    "P", "K", "Ca", "Sc", "Ti", "Zn", "Ga", "Ge",
    "Rb", "Sr", "Y", "Zr", "In", "Sn", "Sb", "Cs",
    "Ba", "La", "Gd", "Lu", "Hf", "Ta"]

MBTR_L = MBTR(species = elements,
               k1={
                   "geometry": {"function": "atomic_number"},
                   "grid": {"min": 1, "max": 73, "n": 10, "sigma": 0.1},},
               k2={
                   "geometry": {"function": "inverse_distance"},
                   "grid": {"min": 0.2, "max": 1, "n": 8, "sigma": 0.05},
                   "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},},
               #k3={
               #"geometry": {"function": "cosine"},
               #"grid": {"min": -1, "max": 1, "n": 10, "sigma": 0.1},
               #"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},},
               periodic=True,
               normalization="l2_each",
              )

def cif2Matrix(input_cif,descriptor):
    cif = read(input_cif)
    output_matrix = descriptor.create(cif)
    return output_matrix

### Input Parameters
input_path = 'Screened_cif'    ### Input cif files.
output_file = 'MBTRMatrix_1K.csv'  ### Generate file name
descriptor = MBTR_L

MBTRMatrix_out = []
index = []
files = os.listdir(input_path)
files.sort()
for filename in files:
    if filename.endswith('.cif'):
        input_cif = os.path.join(input_path,filename)
        MBTRMatrix_out.append(cif2Matrix(input_cif,descriptor))
        index.append(str(filename)[:-4])
df=pd.DataFrame(MBTRMatrix_out)
df.to_csv(output_file)

#ML part
#Read the xy
X_input = 'MBTRMatrix_1K.csv'
Y_file = 'Screened.csv'
Y_property = 'Bulk Modulus'

x = pd.read_csv(X_input)
x = x.drop('Unnamed: 0',axis=1)
y = pd.read_csv(Y_file)[Y_property]

#Training % Testing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn import linear_model; from sklearn import neighbors; from sklearn import tree
from sklearn import svm; from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
'''Data_reading'''
X0 = x; y0 = y #Raw data
'''GridSearch_parameters'''
param_grid_XGB = [{"max_depth":[2,4,6,8]},{"learning_rate":[0.1,0.2,0.3,0.4]},]
'''Method'''
XGB = xgb.XGBRegressor(learning_rate = 0.3, n_estimators=100, verbosity=0,
                       objective='reg:squarederror',booster='gbtree', tree_method='auto',
                       n_jobs=1, gamma=0.0001, min_child_weight=8,max_delta_step=0,
                       num_parallel_tree=1, importance_type='gain',eval_metric='rmse',nthread=16)

GSCV = GridSearchCV(XGB,param_grid_XGB,cv=10) #KNR/RNR/DTR/GBR/XGB
'''Fitting_by StandardScaler()'''
scaler = StandardScaler()
Method = GSCV
X2 = scaler.fit_transform(X0)
Method.fit(X2,y0); y_pred = Method.predict(X2)
'''Parameters'''
#print('Coefficients:',Method.coef_); print('Intercept:',Method.intercept_)
'''Evaluation'''
print('Method:',Method)
print("Mean squared error: %.2f" % mean_squared_error(y0, y_pred)) # The mean squared error
print("Mean absolute error: %.2f" % mean_absolute_error(y0, y_pred)) # The mean squared error
print("Coefficient of determination: %.2f" % r2_score(y0, y_pred)) # The coefficient of determination: 1 is perfect prediction
'''Output xlsx'''

'''File Output'''
Out = pd.DataFrame(); Out['y0']=y0; Out['y_pred']=y_pred
Out.to_excel('Out.xlsx')

#Predicting the Cr3+-doped materials
#Reading cifs, Converting cifs to MBTR Matrix (csv)

elements = [
    "Li", "B", "O", "F", "Na", "Mg", "Al", "Si",
    "P", "K", "Ca", "Sc", "Ti", "Zn", "Ga", "Ge",
    "Rb", "Sr", "Y", "Zr", "In", "Sn", "Sb", "Cs",
    "Ba", "La", "Gd", "Lu", "Hf", "Ta"]

MBTR = MBTR(species = elements,
               k1={
                   "geometry": {"function": "atomic_number"},
                   "grid": {"min": 1, "max": 73, "n": 10, "sigma": 0.1},},
               k2={
                   "geometry": {"function": "inverse_distance"},
                   "grid": {"min": 0.2, "max": 1, "n": 8, "sigma": 0.05},
                   "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},},
               #k3={
               #"geometry": {"function": "cosine"},
               #"grid": {"min": -1, "max": 1, "n": 10, "sigma": 0.1},
               #"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},},
               periodic=True,
               normalization="l2_each",
              )


def Pred_cif(cif_input,descriptor,Method):
    input_x = cif2Matrix(cif_input,descriptor)
    input_x = scaler.transform(input_x.reshape(1,-1))
    prediction = Method.predict(input_x)
    return prediction

def Pred_barch(location):
    cif_files = os.listdir(location)
    cif_files.sort()
    PredictionTable = pd.DataFrame()  #Empty table
    for cif_file in cif_files:
        if cif_file.endswith('.cif'):
            name = str(cif_file)[:-4]
            cif_file = os.path.join(location,cif_file)
            BM = Pred_cif(cif_file,descriptor,Method)
            New_Row = {'Compound':str(name),'Bulk_Modulus(GPa)':float(BM)}
            PredictionTable = PredictionTable.append(New_Row,ignore_index=True)
    PredictionTable.to_excel('Prediction_All_Oxides.xlsx')

def cif2Matrix(input_cif,descriptor):
    cif = read(input_cif)
    output_matrix = descriptor.create(cif)
    return output_matrix        

PATH = 'All_Oxides'
Data = []
descriptor = MBTR
Method = Method
Pred_barch(PATH)

