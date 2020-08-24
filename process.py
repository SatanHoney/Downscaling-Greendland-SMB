import numpy as np
import netCDF4 as nc
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import keras

plt.switch_backend('agg')

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# Process Input
X=[]
X1=[]
X2=[]
model = ['ACCESS1.3', 'CSIRO-Mk3.6', 'IPSL-CM5-MR', 'MIROC5']
filename1 = ['Amon_ACCESS1-3_historical_r2i1p1', 'Amon_CSIRO-Mk3-6-0_historical_r1i1p1', 'Amon_IPSL-CM5A-MR_historical_r3i1p1', 'Amon_MIROC5_historical_r3i1p1']
filename2 = ['Amon_ACCESS1-3_rcp85_r1i1p1', 'Amon_CSIRO-Mk3-6-0_rcp85_r1i1p1', 'Amon_IPSL-CM5A-MR_rcp85_r1i1p1', 'Amon_MIROC5_rcp85_r1i1p1']
para = ['cl', 'hus', 'pr', 'psl', 'rlut', 'rsut',  'ta', 'ts']
# para = ['ta', 'pr', 'ts']

directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/ACCESS1.3-histo_1950_2005/input_years'
for j in range(1970,2006):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename1[0],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] + attri[9] + attri[10] + attri[11])/12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in range(0,70,10):
            for i2 in range(0,48,8):
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X1.append(data)
        else:
            X1[j-1970] = np.concatenate((X1[j-1970],data))
X1 = np.array(X1)
print(X1.shape)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/ACCESS1.3-rcp85_2006_2100/input_years'
for j in range(2006,2101):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename2[0],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] +
                     attri[9] + attri[10] + attri[11]) / 12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in range(0,70,10):
            for i2 in range(0,48,8):
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X2.append(data)
        else:
            X2[j-2006] = np.concatenate((X2[j-2006],data))
X2 = np.array(X2)
print(X2.shape)
X = np.concatenate((X1,X2))
print(X.shape)

X1=[]
X2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-histo_1950_2005/input_years'
for j in range(1950,2006):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename1[1],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] + attri[9] + attri[10] + attri[11])/12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,10,20,30,40,50,60]:
            for i2 in [0,5,10,16,21,26]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X1.append(data)
        else:
            X1[j-1950] = np.concatenate((X1[j-1950],data))
X1 = np.array(X1)
print(X1.shape)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-rcp85_2006_2100/input_years'
for j in range(2006,2101):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename2[1],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] +
                     attri[9] + attri[10] + attri[11]) / 12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,10,20,30,40,50,60]:
            for i2 in [0,5,10,16,21,26]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X2.append(data)
        else:
            X2[j-2006] = np.concatenate((X2[j-2006],data))
X2 = np.array(X2)
print(X2.shape)
X1 = np.concatenate((X1,X2))
print(X1.shape)
X = np.concatenate((X,X1))
print(X.shape)

X1=[]
X2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/IPSL-CM5-MR-histo_1950_2005/input_years'
for j in range(1950,2006):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename1[2],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] + attri[9] + attri[10] + attri[11])/12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,7,15,23,30,37,45]:
            for i2 in [0,8,15,23,31,39]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X1.append(data)
        else:
            X1[j-1950] = np.concatenate((X1[j-1950],data))
X1 = np.array(X1)
print(X1.shape)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/IPSL-CM5-MR-rcp85_2006_2100/input_years'
for j in range(2006,2101):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename2[2],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] +
                     attri[9] + attri[10] + attri[11]) / 12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,7,15,23,30,37,45]:
            for i2 in [0,8,15,23,31,39]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X2.append(data)
        else:
            X2[j-2006] = np.concatenate((X2[j-2006],data))
X2 = np.array(X2)
print(X2.shape)
X1 = np.concatenate((X1,X2))
print(X1.shape)
X = np.concatenate((X,X1))
print(X.shape)

X1=[]
X2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/MIROC5-histo_1950_2005/input_years'
for j in range(1950,2006):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename1[3],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] + attri[9] + attri[10] + attri[11])/12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,12,26,40,53,66,80]:
            for i2 in [0,7,14,21,28,36]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X1.append(data)
        else:
            X1[j-1950] = np.concatenate((X1[j-1950],data))
X1 = np.array(X1)
print(X1.shape)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/MIROC5-rcp85_2006_2100/input_years'
for j in range(2006,2101):
    for k in range(len(para)):
        file = directory + '/%s_%s_%d.nc'%(para[k],filename2[3],j)
        file_obj = nc.Dataset(file)
        attri = file_obj.variables['%s'%para[k]]
        attri = np.array(attri)
        attri_avg = (attri[0] + attri[1] + attri[2] + attri[3] + attri[4] + attri[5] + attri[6] + attri[7] + attri[8] +
                     attri[9] + attri[10] + attri[11]) / 12
        attri_avg = normalization(attri_avg)
        data = []
        for i1 in [0,12,26,40,53,66,80]:
            for i2 in [0,7,14,21,28,36]:
                if para[k] == 'cl' or para[k] == 'ta' or para[k] == 'hus':
                    data.append(attri_avg[0][i2][i1])
                else:
                    data.append(attri_avg[i2][i1])
        data = np.array(data).reshape(-1)
        if k == 0:
            X2.append(data)
        else:
            X2[j-2006] = np.concatenate((X2[j-2006],data))
X2 = np.array(X2)
print(X2.shape)
X1 = np.concatenate((X1,X2))
print(X1.shape)
X = np.concatenate((X,X1))
print(X.shape)
#Process output
Y=[]
Y1=[]
new_Y1=[]
Y2=[]
new_Y2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/ACCESS1.3-histo_1950_2005/outputs'
for i in range(1970,2006):
    file = directory + '/MARv3.9-yearly-ACCESS1.3-histo-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y1.append(SMB[1602560])#2423161
Y1 = np.array(Y1)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/ACCESS1.3-rcp85_2006_2100/outputs'
for i in range(2006,2101):
    file = directory + '/MARv3.9-yearly-ACCESS1.3-rcp85-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y2.append(SMB[1602560])
Y2 = np.array(Y2)
Y = np.concatenate((Y1,Y2))
print(Y.shape)

Y1=[]
Y2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-histo_1950_2005/outputs'
for i in range(1950,2006):
    file = directory + '/MARv3.9-yearly-CSIRO-Mk3.6-histo-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y1.append(SMB[1602560])
Y1 = np.array(Y1)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/CSIRO-Mk3.6-rcp85_2006_2100/outputs'
for i in range(2006,2101):
    file = directory + '/MARv3.9-yearly-CSIRO-Mk3.6-rcp85-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y2.append(SMB[1602560])
Y2 = np.array(Y2)
Y1 = np.concatenate((Y1,Y2))
print(Y1.shape)
Y=np.concatenate((Y,Y1))
print(Y.shape)

Y1=[]
Y2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/IPSL-CM5-MR-histo_1950_2005/outputs'
for i in range(1950,2006):
    file = directory + '/MARv3.9-yearly-IPSL-CM5-MR-histo-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y1.append(SMB[1602560])
Y1 = np.array(Y1)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/IPSL-CM5-MR-rcp85_2006_2100/outputs'
for i in range(2006,2101):
    file = directory + '/MARv3.9-yearly-IPSL-CM5-MR-rcp85-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y2.append(SMB[1602560])
Y2 = np.array(Y2)
Y1 = np.concatenate((Y1,Y2))
print(Y1.shape)
Y=np.concatenate((Y,Y1))
print(Y.shape)

Y1=[]
Y2=[]
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/MIROC5-histo_1950_2005/outputs'
for i in range(1950,2006):
    file = directory + '/MARv3.9-yearly-MIROC5-histo-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y1.append(SMB[1602560])
Y1 = np.array(Y1)
directory = '/nfs/annie/shared/SMB-Gen/ML_downscaling/mar_data/MIROC5-rcp85_2006_2100/outputs'
for i in range(2006,2101):
    file = directory + '/MARv3.9-yearly-MIROC5-rcp85-%d.nc'%i
    file_obj = nc.Dataset(file)
    SMB = file_obj.variables['SMB']
    SMB = np.array(SMB)
    SMB = SMB.reshape(-1)
    Y2.append(SMB[1602560])
Y2 = np.array(Y2)
Y1 = np.concatenate((Y1,Y2))
print(Y1.shape)
Y=np.concatenate((Y,Y1))
print(Y.shape)
mean = np.mean(Y)
median = np.median(Y)

index = [i for i in range(len(Y))]
np.random.seed(1)
np.random.shuffle(index)

X = X[index]
Y = Y[index]
X_train, Y_train = X[:525], Y[:525]
X_test, Y_test = X[525:], Y[525:]

square1 = (Y_test - mean)*(Y_test - mean)
square2 = (Y_test - median)*(Y_test - median)
MSE1 = np.sum(square1)/len(Y_test)
MSE2 = np.sum(square2)/len(Y_test)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=336, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer= keras.regularizers.l2(0.1)))
model.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer= keras.regularizers.l2(0.1)))

model.add(Dense(units=1))

model.compile(loss='mean_squared_error',
              optimizer='adagrad')
print('Training -----------')
history = model.fit(X_train, Y_train, validation_split = 0.1, epochs = 350, batch_size=64, shuffle=False)
print(history.history.keys())
pred_test_y = model.predict(X_test)
pred_test_y = pred_test_y.reshape(-1)

print(pred_test_y)
print(Y_test)
x=np.linspace(np.min(Y),np.max(Y),100)
pdf = PdfPages('loss.pdf')
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot([0,500],[MSE1,MSE1])
plt.plot([0,500],[MSE2,MSE2])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid','mean','median'], loc='upper left')
plt.plot()
pdf.savefig()
plt.close()
pdf.close()
pdf = PdfPages('result.pdf')
plt.figure()
plt.scatter(Y_test,pred_test_y, color = 'r', s=9)
plt.plot(x,x)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.plot()
pdf.savefig()
plt.close()
pdf.close()

MSEP = pred_test_y - Y_test
MSEP = np.sum(MSEP*MSEP)/len(Y_test)

print(MSE1)
print(MSE2)
print(MSEP)





# SMB = 'ACCESS1.3-histo_1950_2005/outputs/MARv3.9-yearly-ACCESS1.3-histo-1950.nc'
# SMB_obj = nc.Dataset(SMB).variables
# def find(arr,min,max):
# 	pos_min = arr>=min
# 	pos_max =  arr<=max
# 	pos_rst = pos_min & pos_max
# 	return np.where(pos_rst == True)
#
# pos1=find(SMB_obj['LAT'][:],66.8,67.2)
# loc=0
# pos2=find(SMB_obj['LON'][:],-48.7,-48.2)
# for i in range(len(pos1[0])):
#     pos3 = np.where(pos2[0]==pos1[0][i])
#     for j in range(len(pos3[0])):
#         if pos2[1][pos3[0][j]] == pos1[1][i]:
#             loc=i
#             break
# y = pos1[0][loc]
# x = pos1[1][loc]
# print(y)
# print(x)
# print(SMB_obj['LAT'][y][x])
# print(SMB_obj['LON'][y][x])

