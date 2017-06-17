import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m
# Part 1: Load the data and separate it into three arrays; one for each class
data = np.loadtxt('MLE1_iris.data')
# ^^ YOUR CODE HERE ^^
data0 = data[0:50, :]
data1 = data[50:100, :]
data2 = data[100:150, :]
#tum = data[1, 5]
# Part 2: Plot each type of data for all classes in 1D (with shifts of 0.1 for better visualization)
fig = plt.figure()
plt.plot(data0[:, 0], np.ones(len(data0[:, 0])) * 0.0, '+r', label='Data 0 Class 0')
plt.plot(data1[:, 0], np.ones(len(data1[:, 0])) * 0.1, '+g', label='Data 0 Class 1')
plt.plot(data2[:, 0], np.ones(len(data2[:, 0])) * 0.2, '+b', label='Data 0 Class 2')

plt.plot(data0[:, 1], np.ones(len(data0[:, 1])) * 1.0, 'xr', label='Data 1 Class 0')
plt.plot(data1[:, 1], np.ones(len(data1[:, 1])) * 1.1, 'xg', label='Data 1 Class 1')
plt.plot(data2[:, 1], np.ones(len(data2[:, 1])) * 1.2, 'xb', label='Data 1 Class 2')

plt.plot(data0[:, 2], np.ones(len(data0[:, 2])) * 2.0, '.r', label='Data 2 Class 0')
plt.plot(data1[:, 2], np.ones(len(data1[:, 2])) * 2.1, '.g', label='Data 2 Class 1')
plt.plot(data2[:, 2], np.ones(len(data2[:, 2])) * 2.2, '.b', label='Data 2 Class 2')

plt.plot(data0[:, 3], np.ones(len(data0[:, 3])) * 3.0, '1r', label='Data 3 Class 0')
plt.plot(data1[:, 3], np.ones(len(data1[:, 3])) * 3.1, '1g', label='Data 3 Class 1')
plt.plot(data2[:, 3], np.ones(len(data2[:, 3])) * 3.2, '1b', label='Data 3 Class 2')

plt.legend(fontsize=9, loc=3)

# Part 3: Examining the plots above select two of the data types and plot them in 2D - one data type for each axis. Let's say you chose ath and bth columns as your data. This means you have to plot dataN[:,a] vs dataN[:,b] for N=0,1,2.
# ^^ YOUR CODE HERE ^^
figxy = plt.figure()
plt.plot(data0[:, 2], data0[:, 3], '+r', label='Data x Class 1')
plt.plot(data1[:, 2], data1[:, 3], '+g', label='Data x Class 2')
plt.plot(data2[:, 2], data2[:, 3], '+b', label='Data x Class 3')

# Part 4: Using the two datatype you have chosen, extract the 2D Gaussian (Normal) distribution parameters. Numpy functions are called here to be used ONLY for validation of your results.
# mx0 = np.mean(data0[:, 2])
# my0 = np.mean(data0[:, 3])
# cov0 = np.cov(data0[:, 2:4].T)
# mx1 = np.mean(data1[:, 2])
# my1 = np.mean(data1[:, 3])
# cov1 = np.cov(data1[:, 2:4].T)
# mx2 = np.mean(data2[:, 2])
# my2 = np.mean(data2[:, 3])
# cov2 = np.cov(data2[:, 2:4].T)
#
# varx0 = np.var(data0[:, 2])
#
# varx1 = np.var(data1[:, 2])
# varx2 = np.var(data2[:, 2])
# vary0 = np.var(data0[:, 3])
# vary1 = np.var(data1[:, 3])
# vary2 = np.var(data2[:, 3])


# ^^ YOUR CODE HERE ^^
# for mean
def mean(x):
    return (np.sum(x) / len(x))
mx0 = mean(data0[:,2])
mx1 = mean(data1[:,2])
mx2 = mean(data2[:,2])
my0 = mean(data0[:,3])
my1 = mean(data1[:,3])
my2 = mean(data2[:,3])


# for variance 
sumx0 = 0
data00_elements = data0[:,2]
for i in range(len(data00_elements)):
    sumx0 += m.pow((mx0 - data00_elements[i]), 2)
varx0 = sumx0/50


sumx1 = 0
data10_elements = data1[:,2]
for i in range(len(data10_elements)):
    sumx1 += m.pow((mx1 - data10_elements[i]), 2)
varx1 = sumx1/50

sumx2 = 0
data20_elements = data2[:,2]
for i in range(len(data20_elements)):
    sumx2 += m.pow((mx2 - data20_elements[i]), 2)
varx2 = sumx2/50


sumy0 = 0
data01_elements = data0[:,3]
for i in range(len(data01_elements)):
    sumy0 += m.pow((my0 - data01_elements[i]), 2)
vary0 = sumy0/50


sumy1 = 0
data11_elements = data1[:,3]
for i in range(len(data11_elements)):
    sumy1 += m.pow((my1 - data11_elements[i]), 2)
vary1 = sumy1/50


sumy2 = 0
data21_elements = data2[:,3]
for i in range(len(data21_elements)):
    sumy2 += m.pow((my2 - data21_elements[i]), 2)
vary2 = sumy2/50


# for covariance
def covariance(x,y):
    sum_covx0 = 0
    for i in range(len(x)):
        sum_covx0 += (x[i] - mean(x)) * (y[i] - mean(y))
    return sum_covx0 /(len(x) -1)

cov0 = []
cov0.append([])
cov0.append([])
cov0[0].append(covariance(data0[:,2],data0[:,2]))
cov0[0].append(covariance(data0[:,2],data0[:,3]))
cov0[1].append(covariance(data0[:,3],data0[:,2]))
cov0[1].append(covariance(data0[:,3],data0[:,3]))

cov1 = []
cov1.append([])
cov1.append([])
cov1[0].append(covariance(data1[:,2],data1[:,2]))
cov1[0].append(covariance(data1[:,2],data1[:,3]))
cov1[1].append(covariance(data1[:,3],data1[:,2]))
cov1[1].append(covariance(data1[:,3],data1[:,3]))

cov2 = []
cov2.append([])
cov2.append([])
cov2[0].append(covariance(data2[:,2],data2[:,2]))
cov2[0].append(covariance(data2[:,2],data2[:,3]))
cov2[1].append(covariance(data2[:,3],data2[:,2]))
cov2[1].append(covariance(data2[:,3],data2[:,3]))

# Part 5: Plot the Gaussian surfaces for each class.
## First, we generate the grid to compute the Gaussian function on.
vals = np.linspace(np.min(data), np.max(data), 500)
x, y = np.meshgrid(vals, vals)


## Next, we define and implement the 2D Gaussian function.
def gaussian_2d(x, y, mx, my, cov):
    ''' x and y are the 2D coordinates to calculate the function value
        mx and my are the mean parameters in x and y axes
        cov is the 2x2 variance-covariance matrix'''
    ret = 0

    # ^^ YOUR CODE HERE ^^
    sigmax = np.sqrt(cov[0][0])
    sigmay = np.sqrt(cov[1][1])
    p = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))
    ret = (1 / (2 * np.pi * sigmax * sigmay * np.sqrt( 1 - np.power(p,2)))) * np.exp((( -1 / ( 2 * ( 1 - np.power(p,2)))) * ( ((np.power((x - mx), 2)) / (np.power(sigmax,2))) + ((np.power((y - my), 2)) / ( np.power(sigmay, 2))) - (( 2 * p * (x - mx) * (y - my)) / (sigmax * sigmay)))))

    return ret

## Finally, we compute the Gaussian function outputs for each entry in our mesh and plot the surface for each class.
z0 = gaussian_2d(x, y, mx0, my0, cov0)
z1 = gaussian_2d(x, y, mx1, my1, cov1)
z2 = gaussian_2d(x, y, mx2, my2, cov2)
fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.plot_surface(x, y, z0, cmap=cm.jet, linewidth=0, antialiased=False)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x, y, z2, cmap=cm.jet, linewidth=0, antialiased=False)

plt.show()


# Part 6: Classify each sample in the dataset based on your findings and assign a class label. Explain your reasoning behind your implementation with few sentences
lbl = []
for d in data:
    label = 0
    # ^^ YOUR CODE HERE ^^
    
    # Each sample can belong to one of three classes which are 0, 1 or 2.
    # First we need to give our samples to gaussian function.
    # Parameters for gaussian are 2nd and 3rd index because we chose them. They are seperated more than the others.
    # And for the other parameters we try for mx0,my0,cov0 and mx1, my1, cov1 and mx2, my2, cov2.
    # From these three we should pick which one is maximum. Because this is MLE.
    # g0 is gaussian for class 0.
    # g1 is gaussian for class 1.
    # g2 is gaussian for class 2.
    g0 = gaussian_2d(d[2],d[3],mx0,my0,cov0)
    g1 = gaussian_2d(d[2],d[3],mx1,my1,cov1)
    g2 = gaussian_2d(d[2],d[3],mx2,my2,cov2)


    class_of_Data = max(g0, g1, g2)
    
    label = 0 if class_of_Data == g0 else 1 if class_of_Data == g1 else 2
    
    lbl.append(label)

# Part 7: Calculate the success rate - the percentage of correctly classified samples    
success_rate = 0
# ^^ YOUR CODE HERE ^^
wrong_guess_number = 0
index = 0
for d in data:
    if lbl[index] != d[4]:
        wrong_guess_number = wrong_guess_number + 1
    index = index + 1

success_rate = ((len(data) - wrong_guess_number) * 100) / (len(data))
print 'Success rate is %4.2f %%' % success_rate

# Part 8: Repeat the same process for non-overlapping training and test sets.
data_test = np.vstack((data[0:25], data[50:75], data[100:125]))
data_train = np.vstack((data[25:50], data[75:100], data[125:150]))
data_test0 = data_test[data_test[:, 4] == 0]
data_test1 = data_test[data_test[:, 4] == 1]
data_test2 = data_test[data_test[:, 4] == 2]
data_train0 = data_train[data_train[:, 4] == 0]
data_train1 = data_train[data_train[:, 4] == 1]
data_train2 = data_train[data_train[:, 4] == 2]

mx00 = mean(data_train0[:,2])
mx11 = mean(data_train1[:,2])
mx22 = mean(data_train2[:,2])
my00 = mean(data_train0[:,3])
my11 = mean(data_train1[:,3])
my22 = mean(data_train2[:,3])

cov00 = []
cov00.append([])
cov00.append([])
cov00[0].append(covariance(data_train0[:,2],data_train0[:,2]))
cov00[0].append(covariance(data_train0[:,2],data_train0[:,3]))
cov00[1].append(covariance(data_train0[:,3],data_train0[:,2]))
cov00[1].append(covariance(data_train0[:,3],data_train0[:,3]))



cov11 = []
cov11.append([])
cov11.append([])
cov11[0].append(covariance(data_train1[:,2],data_train1[:,2]))
cov11[0].append(covariance(data_train1[:,2],data_train1[:,3]))
cov11[1].append(covariance(data_train1[:,3],data_train1[:,2]))
cov11[1].append(covariance(data_train1[:,3],data_train1[:,3]))

cov22 = []
cov22.append([])
cov22.append([])
cov22[0].append(covariance(data_train2[:,2],data_train2[:,2]))
cov22[0].append(covariance(data_train2[:,2],data_train2[:,3]))
cov22[1].append(covariance(data_train2[:,3],data_train2[:,2]))
cov22[1].append(covariance(data_train2[:,3],data_train2[:,3]))

lbl_of_test = []
for t in data_test:
    label_test = 0
   
    g00 = gaussian_2d(t[2],t[3],mx00,my00,cov00)
    g11 = gaussian_2d(t[2],t[3],mx11,my11,cov11)
    g22 = gaussian_2d(t[2],t[3],mx22,my22,cov22)


    class_of_Data = max(g00, g11, g22)
    
    label_test = 0 if class_of_Data == g00 else 1 if class_of_Data == g11 else 2
    
    lbl_of_test.append(label_test)


# Success for test data
success_rate_test = 0
wrong_guess_number_test = 0
index_test = 0
for d in data_test:
    if lbl_of_test[index_test] != d[4]:
        wrong_guess_number_test = wrong_guess_number_test + 1
    index_test = index_test + 1

success_rate_test = ((float(len(data_test)) - wrong_guess_number_test) * 100) / (len(data_test))
print 'Success rate for test data is %4.2f %%' % success_rate_test