import numpy as np

#using the following ndarray
a=np.array([[1,2],[3,4],[5,6]])
#print the number 4
print(a[1][1])
#print [5 6]
print(a[2])
#print [1 4 5]
print(a[[0,1,2],[0,1,0]])

#Create an ndarray of 4 values spaced evenly between, and including, 1 and 2.
a = np.linspace(1,2,4)
print(a)

#Sort the following numpy ndarray in ascending and then in descending order
#After each sort, print the results
a = np.array([6, 1, 4, 2, 18, 9, 3, 4, 2, 8, 11])

a = np.sort(a)
print(a)
a = np.sort(a)[::-1]
print(a)

#Using the following ndarray,
a = np.array([1,4,3,2,5,6,9,2,7,1], ndmin=2)
#reshape the array to 2 rows by 5 columns
#sort in ascending order along the first axis. Then print the result.
#sort in ascending order along the second axis. Then print the result.
a = a.reshape(2,5)
print(a)
a = np.sort(a, axis=0)
print(a)
a = np.array([1,4,3,2,5,6,9,2,7,1], ndmin=2)
a = a.reshape(2,5)
a = np.sort(a, axis=1)
print(a)

#Using the following ndarray, sort it and keep the unique values.  Then, print the array.
#Finally, display a message similar to the following (for each unique item):
#We are currently on the number 1 in our ndarray loop!
a = np.array([6, 1, 4, 2, 18, 9, 3, 4, 2, 8, 11])
a = np.sort(a)
a = np.unique(a)
print(a)
for x in a:
    print(f'We are currently on the number {x} in our ndarray loop!')

#Using arange, create an ndarray from 0 to 47 of floating point numbers.
#Reshape the ndarray to have 2 frames, 8 rows and 3 columns in each frame.
#Print the ndarray and the number of dimensions in the reshaped ndarray.
#Loop over the ndarray and print each item in the ndarray.
#Print the value contained in the second frame, third row and first column!
#Change the value of the last item in the first frame (last row and last column) to 100!

a = np.arange(48, dtype = float)
a = a.reshape(2,8,3)
print(a)
print(a.ndim)

for x in np.nditer(a):
    print(x)

print(a[1][2][0])
a[0][7][2] = 100


#Using the following ndarray, add a row containing 30, 33 and 36.
#Print the updated ndarray
a = np.array([[12,15,18], [21,24,27]])
newarray = np.append(a,[[30,33,36]], axis = 0)
print(newarray)

#Using the following ndarray, delete the 399 and print the ndarray
a = np.array([314,399,409])
newarray = np.delete(a,1,axis = 0)
print(newarray)

#using the following ndarray, delete the first column, then delete the third row.
#Then, print the final ndarray
a = np.array([[325,350,375], [400,425,450], [475,500,525]])
newarray = np.delete(a,0,axis=1)
newarray = np.delete(newarray,2,axis=0)
print(newarray)

#Using the following ndarray, test to see whether the number 300 is in the ndarray.
# Print the result of your search.  Your printed message should be as follows:
#300 is found at index: [] and [].
a = np.array([[325,350,375], [400,425,450], [475,500,525]])
x = 300
index = np.where (a == x)
print(f'{x} is found at index: {index[0]} and {index[1]}.')

#Using the following ndarray, multiply each value by 10.  Then, print the result.
#Array after mult function:  [2110 2120 2130 2140 2150 2160 2170 2180]
a = np.array([211,212,213,214,215,216,217,218])
mult = lambda x: x * 10
print('Array after mult function: ', mult(a))

#Using the following ndarray, divide each value by 2 and then print the results only for the values greater than 107.
a = np.array([211,212,213,214,215,216,217,218], dtype=float)
for x in np.nditer(a, op_flags = ['readwrite']):
    x[...] = x/2

print(a[a>107])

#Using the following ndarray, print the 100th percentile across all of the data
#Print the 75th percentile going down
#Print the 25th percentile going across
a = np.array([[12, 15, 20, 33], [180, 143, 131, 1], [190, 222, 143, 110]])
print(np.percentile(a,100)) #all of the data across both axis 1 and axis 2 together
print('Percentile along axis 0', np.percentile(a, 75, 0))
print('Percentile along axis 1', np.percentile(a, 25, 1))

#Using the following ndarray, print the median, mean, standard deviation and sum across both axis 1 and axis 2
a = np.array([[12, 15, 20, 33], [180, 143, 131, 1], [190, 222, 143, 110]])
print(np.median(a,axis=0))
print(np.median(a,axis=1))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))
print(np.std(a,axis=0))
print(np.std(a,axis=1))
print(np.sum(a,axis=0))
print(np.sum(a,axis=1))

#Create an ndarray of dates between 3/1/2020 (inclusive) and 5/31/2020 (inclusive)
#Print the following message for all dates on Tuesdays:
#2020-03-03 is on a Tuesday!
#2020-03-10 is on a Tuesday!
#2020-03-17 is on a Tuesday!
#....
#....
#2020-05-19 is on a Tuesday!
#2020-05-26 is on a Tuesday!

a = np.arange('2020-03-01', '2020-06-01', dtype='datetime64[D]')
print(a)
from dateutil.parser import *
my_list = a.astype('O').tolist()
print(my_list)
for x in my_list:
    if x.strftime('%A') == 'Tuesday':
        print(f'{x} is on a {x.strftime("%A")}!')

#Create an ndarray of zeros (2 by 2) similar to the following:
#[[(0., 0) (0., 0)]
# [(0., 0) (0., 0)]]
#Name the float numbers as x and the integer zeros as y
#Change the x zeros to a random number
#print off your final ndarray, which should look similar to the following:
#[[(0.556773, 0) (0.556773, 0)]
# [(0.556773, 0) (0.556773, 0)]]
a = np.zeros((2,2), dtype=[('x', np.float32),('y', np.int32)])
print(a)
a['x'] = np.random.random()

#Create an empty (2 by 2) ndarray as follows:
# [[( 0.0000000e+00, 1072693248) ( 1.4660155e+13, 1073042773)]
# [(-3.0316488e-13, 1073392298) ( 0.0000000e+00, 1073741824)]]
##Name the float numbers as x and the integer zeros as y
#NOTE: Using the np.empty will result in different initial values!
#Next, change each value in the ndarray to be as follows:
#[[(1.,  50) (2., 100)]
# [(3., 150) (4., 200)]]
#Finally, print your ndarray
a = np.empty((2,2), dtype=[('x', np.float32),('y', np.int32)])
print(a)
xcounter=0
ycounter=0
for nums in a:
    xcounter+=1
    nums['x'][0] = xcounter
    xcounter += 1
    nums['x'][1] = xcounter
    ycounter +=50
    nums['y'][0] = ycounter
    ycounter += 50
    nums['y'][1] = ycounter


#Create an ndarray of 10 evenly spaced numbers (float data types) between 1 and 20 (including the endpoints).
#Then, reshape the array to be two rows and five columns.
#Finally, print the results.
a = np.linspace (1,20,10, endpoint=True, dtype=np.float64)
print(a)
a = a.reshape(2,5)
print(a)

#Create an ndarray of 20 float numbers between 1 and 10 (including the endpoints) along a logscale
# with a base of 2 (include the endpoints)
#Then, reshape the array to be 4 rows by 5 columns
#Finally, print the final ndarray.
a = np.logspace (1,10,20,endpoint=True, base=2, dtype=np.float64)
print(a)
a = a.reshape(4,5)
print(a)

#Using the following ndarray, create a new array that is the squareroot of each value.
#Print the final ndarray when finished
a = np.array([[1,2,3],[4,5,6]])
b = np.sqrt(a)
print(b)

#Using the arange function, create the following ndarray:
#[[10 11 12 13 14]
# [15 16 17 18 19]]
#Then, print out the result
a = np.array([np.arange(10, 15), np.arange(15, 20)])
print(a)

#Using the following numpy ndarray, print the items that are greater than or equal to 7
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(a[a >= 7])

#Using the following numpy ndarray
a = np.array([[211,212,213,214],[215,216,217,218], [219,220,221,222]])
#Create a new ndarray with the following items and print the result
#[[216 217 218]
#[220 221 222]]
b = a[1:3, 1:4]
print(b)
#Alternatively,
b = a[1:3, [1,2,3]]
print(b)

#using the following numpy ndarray:
a = np.array([[211,212,213,214],[215,216,217,218], [219,220,221,222]])
#Create a new ndarray with the following items and print the results:
#[[211 214]
# [219 222]]
rows = np.array([[0,0], [2,2]])
cols = np.array([[0,3],[0,3]])  #corner items
b = a[rows,cols]
print(b)

#Using the following numpy ndarray
a = np.array([[211,212,213,214],[215,216,217,218], [219,220,221,222]])
#Create a new ndarray with the following items and print the results:
#[[212 213]
# [220 221]]
rows = np.array([[0,0], [2,2]])
cols = np.array([[1,2],[1,2]]) #middle items
b = a[rows,cols]
print(b)

#Create a two by two dimensional np.chararray of byte strings.
#Set each value to be z.
#Then attempt to change the second row, first column to bill
#Then, print the following message:
#We could not change this item to bill because the itemsize is 1.
#Instead, the value has been changed to b.

charar = np.chararray((2, 2))
charar[:] = 'z'
charar[1][0] = 'bill'
print(f'We could not change this item to bill because the itemsize is {charar.itemsize}.\nInstead, the value has '
      f'been changed to {charar[1][0].decode("utf-8")}.')