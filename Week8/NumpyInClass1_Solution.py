#numpy is derived from numeric, which was developed in 1995 and then rebuilt as numpy in 2006
# multidimensional matrices and arrays are the core numpy objects. Along with that, it provides a
# gamut of high-level functions to perform mathematical operations on these structures.
import numpy as np

#base object is an n-dimensional array
a = np.array([8,14,19])
#a = np.array([8,14,19], ndmin=1) #change ndmin from 1 to 2 to 3 to 4
print(a)
print(a.dtype)
print(a.ndim)

a = np.array(range(0,5000), dtype=int) #change the data type to int
print(a)

#Technically, the data types are numpy data types and not core Python classes/data types
#but practically, a float and an np.float will have the same effect for our purposes.
a = np.array(range(4999,-1,-3), dtype=np.float, ndmin=2)  #Notice the np.
print(a)
print(a.dtype)  #Notice the float64
print(a.ndim) #This links back to the ndmin argument of 2 that we passed to our array definition

#One of the benefits of numpy is that it can handle more complex data types beyond just
#int, float, str
a = np.array([1,2,3,4,5], dtype=np.complex, ndmin=2)
print(a)
print(a.dtype)

#However, a business analyst will typically just deal with the basics of int, float and strings
a = np.array([1,2,3,4,5], dtype=str, ndmin=2)  #These look like numbers but they are really strings for this example
print(a)
print(a.dtype)
#What does the > or < sign mean?
#The < and the > refer to the byte encoding ...
# < is little-endian where the least significant is stored in the smallest address.
# > is big-endian where the most significant byte is stored in the smallest address.
# A string is stored as a 'U' string, which stands for unicode string
# The number is the length of the string.  Notice what happens when we change the 5 to a 15 in the above ndarray.

#Technically, strings are unicode strings in a numpy array.
#What will the size of the string data type be in this case?
a = np.array([15,20,30,35,40,45,70,80,95,100,105,110], dtype=np.unicode, ndmin=2)
print(a)
print(a.dtype)

#Here is another example of an ndarray containing strings.
a = np.array(['jane smith','sally jones','james baily', 'mike gimmel'], ndmin=1)
print(a)
print(a.dtype)

#We have the ability to add elements (individually) to the ndarray object
a = np.append(a, 'james thorndike')
print(a)
#if we want to create a newarray by appending to a (but keeping the existing a ndarray object)
newarray = np.append(a, 'testing new person')
print(newarray)
print(a)

#Print the dtypes of both ndarrays.  Why are they different?
print(newarray.dtype)
print(a.dtype)

#We can also append elements and items by passing in a list or a tuple
newarray = np.append(a, ('jane smith-cline', 'william jones'))
print(newarray)
newarray = np.append(a, ['jane smith-cline', 'william jones'])
print(newarray)

#Create an ndarray with two rows and three columns
a = np.array([[10,20,30],[40,50,60]])
print(a)
print(a.shape)
#(2,3)....2 rows and 3 columns

#We can also reshape our ndarray using the reshape function
a = a.reshape(3,2)
print(a)
print(a.shape)

#Add more dimensions!...for this example, let's create an array of zeros with three axises!
#frames, rows, columns
a = np.zeros((2,3,4), dtype=float)
print(a.shape)
print(a) #Notice the two sections (frames)

#Create a 3 by 3 ndarray of random numbers between 0 and 0.99999
a = np.random.random((3,3))
print(a)

#We reference items in the ndarray like we would in a list!
#Extract the 2nd row and 3rd column
print(a[1][2])
print(a[(1),(2)]) #This should be the same!

#Here is another example of reshaping an numpy ndarray
a = np.array([[11,22,33,44],[55,66,77,88]])
print(a)
print(a.shape) ###rows by columns
a.shape=(4,2)  ###the reshape must come out evenly based on the current shape!
#We can set the shape property or we can reshape the ndarray using the reshape method
a = a.reshape(4,2)
print(a)

#The arange function is a useful function to create an ndarray.
#This function is similar to the range function
a = np.arange(24)
#a = np.arange(24, dtype = float)
print(a)
print(type(a))
print(a.ndim)

#Now reshape the array to 2 frames by 4 rows by 3 columns
b = a.reshape(2,4,3)
print(b)
print(b.ndim)

#print the 3rd row, 1st column in the 2nd frame
print(b[1][2][0])

#Again, arange is conceptually similar to the range function
a = np.arange(10,20,2) #non-inclusive of the 2nd argument
print(a)
a = np.arange(20,5,-2)
print(a)

#Let's add a fourth axises to an ndarray (metaframes, frames, rows, columns).
#Four is probably as complex as we would make.
a = np.arange(48)
a = a.reshape(2,2,3,4)
print(a)
print(a.shape)
# which is (metaframes, frames, rows, columns)
print(a)

#print the 23 in this reshaped ndarray
print(a[0][1][2][3]) #go from metaframe all the way to column index

#The linspace function allows us to create evenly spaced elements between two points (inclusive)
a = np.linspace(10,20,5)  #5 evenly spaced elements including the endpoint
#a = np.linspace(10,20,5, endpoint = True, retstep= True) #retstep will store the increment units
print(a)
print(type(a))

#Create a 2 row by 3 column ndarray of ones
a = np.ones((2,3)) #default dtype is a float
print(a)

#Create a 1 row by 2 column ndarray of zeros
a = np.zeros((1,2))
print(a)

#create an identity matrix of 4 rows and 4 columns
a = np.eye(4)
print(a)

#Create a matrix (3 rows by 2 columns) of constants
a = np.full((3,2),7) #Matrix of constants of 7....3 rows by 2 columns
print(a)

#We can slice an array using index referencing
a = np.array([501,502,503,504,505,506,507,508,509,510,511,512,513,514,515])
b = a[0:2] #non-inclusive of the last element
print(b)
b = a[-3:]  #negative indexing should also work
print(b)

a = np.array([[811,812,813], [814,815,816]])
print(a.shape) #a.shape will return a tuple (m, n), where m is the number of rows, and n is the number of columns (in this case).
b = a[1][1:2] #second row....second column
print(b)
b = a[1][1:]  #second row and columns 1 to the end
print(b)

a = np.array([[1,2,3],[3,4,5]])
b = np.array([[40],[50]])
newarray = np.append(a,b, axis = 1)
print(newarray)
print(newarray.shape)

#Notice what happens when I omit the axis argument
#We have essentially flattened the ndarray because it was confused as to where to add the 40 and the 50
b = np.array([[40],[50]])
newarray = np.append(a,b)
print(newarray)
print(newarray.shape)

#Why do we get an error with the following bit of code?
#because the shape of a does not match the shape of b along the 0 axis.
#b = np.array([[40],[50]])
b = np.array([[40,41,42],[50,51,52]]) #add two rows
newarray = np.append(a,b, axis=0) #notice the error initially
print(newarray)
print(newarray[0][1])
#change the 2 to a 147...We can also do this by indexing!
newarray[0][1] = 147
print(newarray[0,1])
print(newarray[0][1])

#change the 5 to a 105
newarray[1][2] = 105
print(newarray)
print(newarray[1,2])
print(newarray[1][2])

#We can also use the insert method to insert an element or a column to a specific position
a = np.array([211,212,213])
#Second argument is the index position
newarray = np.insert(a,1,214)
print(newarray)
print(newarray.shape)

#Notice when the ndarray has multiple axises...we can still do an insert but we have to be careful
# to specify where we want to add the data!
a = np.array([[211,212,213], [214,215,216]])
newarray = np.insert(a,1,222, axis=1)
print(newarray)
newarray = np.insert(a,1,[222,223,224], axis=0)
print(newarray)

#We can delete an element from a single dimensional ndarray
a = np.array([211,212,213])
newarray = np.delete(a,1,axis=0)
print(newarray)
print(newarray.shape)

a = np.array([[211,212,213], [214,215,216], [217,218,219]])
newarray = np.delete(a,1,axis=0) #delete the second row
print(newarray)

#now let's delete a column
a = np.array([[211,212,213], [214,215,216], [217,218,219]])
newarray = np.delete(a,1,axis=1) #delete the second column
print(newarray)

#The size property determines whether the ndarray is empty or not...similar to the len of a list
a = np.array([[211,212,213], [214,215,216]])
#a = np.array([])
if a.size == 0:
    print('The given np array is empty')
else:
    print(f'The np array contains {a} and has a size of {a.size}.')

#We can create an np.ndarray from a list or a tuple
my_list = [11, 12, 13, 14, 15]
a = np.array(my_list)
print(f'The Numpy array from Python list = {a}')

#There is also an asarray function that we can use to create an ndarray from any type of sequence.
my_list = [11, 12, 13, 14, 15]
a = np.asarray(my_list) #convert a sequence to a numpy ndarray using the asarray() function
a = np.asarray(my_list, dtype = float) #convert a sequence to a numpy ndarray using the asarray() function and change the data type
print(f'The Numpy array from Python list = {a}')
my_newlist = a.tolist() #convert an ndarray back to a list
print(my_newlist)

#Notice this list containing two tuples. The tuples are different sizes but we can create an ndarray from this structure.
x = [(1,2,3), (4,5)]
a = np.asarray(x)
print(a)
print(type(a))
print(a.shape)
print(a[0][1])
print(a[1][1])

x = [[(1,2,3), (4,5)]]
a = np.asarray(x)
a = a.reshape(2,1) #Notice how it appears that we have a different number of columns (but we only have a single column)
print(a)

#Another way to create an ndarray from a collection is to use the fromiter function
my_list = [11, 12, 13, 14, 15]
my_list = [11, 12, 'Bill Smith', 14, 15] #We get an error here because Bill Smith cannot be converted to a float!
a = np.fromiter(iter(my_list), dtype = float)
print(a)

#We can also create an ndarray from a string variable using the numpy frombuffer function
s = 'Hello Professor Mattson'
a = np.frombuffer(s.encode('utf-8'), dtype='S1')
print(a) #Notice how these are byte strings and not traditional strings
print(a[1])
print(a[1].decode('utf-8'))
print(a[1].decode('utf-8').upper())

#We can sort a numpy ndarray but there is no Reverse=True option
#Let's first work with a flat numpy array
a = np.array([211,218,213,210,215,216,217,209])
print(f'sorted array = {np.sort(a)}')
print(a)

#Now let's sort a two dimensional ndarray
a = np.array([[211,214,213,212],[215,213,217,211]])
print(f'sorted array = {np.sort(a)}')  #each row is being sorted going across
newArray = np.append(a,[[220,218,219,217]], axis = 0)
print(newArray.shape)
print(newArray)
print(f'sorted array = {np.sort(newArray, axis = 0)}')
print(f'sorted array = {np.sort(newArray, axis = 1)}')
print(f'sorted array = {np.sort(newArray)}')

#np supports all kinds of dtypes, including custom data types!
dt = np.dtype([('tenure', int)])
x = np.array([(10,),(15,),(17)], dtype=dt)
print(x)
#Now print just the values for tenure
print(x['tenure'])

#Another example of a custom data type (dtype) where we specify the datatypes
dt = np.dtype([('id', np.unicode,100), ('name', str, 100), ('score', float)])
print(dt['id'])
print(dt['name'])
print(dt['score'])
x = np.array([('tjones','Tim Jones', 22),('bjohnson', 'Bill Johnson', 33),('tmattson', 'Tom Mattson', 29)], dtype=dt)
print(x)
print(x.shape)
print(x[2][2])
#Loop over similar to how we would loop over a list or a tuple
#Here we will loop over everything in the third position, which is a tuple
for i in x[2]:
    print(i)

#Print everything in each of the named columns
print(x['id'])
print(x['name'])
print(x['score'])

#print everything in a specific column individually
for i in x['id']:
    print(i)

#we can easily export a numpy array to a csv file
import numpy as np
a = np.array([211,212,213,214,215,216,217,218])
np.savetxt('testingoutput.txt', a)
a = np.array([[211,212,213,214],[215,216,217,218]])
print(a)
np.savetxt('testingoutput.txt', a, fmt='%.2f', delimiter=';')

#Read in data from a delimited text file to a numpy array
my_data = np.loadtxt('testingoutput.txt', delimiter=';')
print(my_data)
print(type(my_data))
print(my_data[1][3])
print(my_data[1][2])

#We can also work with the json object
import json
#Notice the format of the string.
json_rings = '{"rings" : [[[-8081441.0, 5685214.0], [-8081446.0, 5685216.0], [-8081442.0, 5685219.0], ' \
             '[-8081440.0, 5685211.0], [-8081441.0, 5685214.0]]]}'
dict_load = json.loads(json_rings)
numpy_2d_arrays = np.array(dict_load["rings"])
print(numpy_2d_arrays)
print(numpy_2d_arrays)


#We can also construct the ndarray in the following manner
numpy_2d_arrays = [np.array(ring) for ring in dict_load["rings"]]
print(numpy_2d_arrays)
print(numpy_2d_arrays[0])

#Now let's work with a dictionary object instead of a string that 'looks' like a dictionary object
dict_rings = {"rings1": np.array([[-8081441.0, 5685214.0], [-8081446.0, 5685216.0], [-8081442.0, 5685219.0], [-8081440.0, 5685211.0], [-8081441.0, 5685214.0]])}
dict_rings['rings2'] = np.array([[-9031441.0, 7655214.0], [-9031446.0, 8755216.0], [-9031442.0, 7645219.0], [-9031440.0, 3125211.0], [-9031441.0, 1235214.0]])
print(dict_rings)
a = dict_rings["rings1"]
print(a)
print(type(a))
b = dict_rings["rings2"]
print(b)
c = np.concatenate((a,b), axis=1) #add columns
print(c)
c = np.concatenate((a,b), axis=0) #add rows
print(c)

#we can search for items in an ndarray
x = -9031442.0
print(f'{x} is found at index: {np.where(c == x)}')

#if we just want the position without the data type...
x = -9031442.0
index = np.where (c == x)
print(type(index))
print(index)
print(index[0])
print(type(index[0]))
print(c)
print(f'{x} is found at index: {index[0]} and {index[1]}')
print(f'{x} is found at index: {int(index[0])} and {int(index[1])}')
print(c[7][0])
print(c[int(index[0])][int(index[1])])

#Apply a function to an array...let's multiple 10 to each element in the ndarray object
multiply = lambda x: x * 10
a = np.array([51,52,53,54,55,56,57,58])
print('Array after multiply function: ', multiply(a))
print(a)
def newmultiply(a):
    return a*10
print('Array after multiply function: ', newmultiply(a))
b = multiply(a)
#or alternatively
b = newmultiply(a)
print ('new array b is ', b)

#Now let's work with index referencing in more detail
a = np.array([[211,212,213,214],[215,216,217,218]])
print(a)
#We have many different ways to reference the items in this ndarray
print(a[1,:])  #print the second row
print(a[1][:]) #also print the second row
print(a[:,1]) #print the second column in both rows
print(a[0,1:]) #first column items 1 (2nd item) until the end
print(a[0:2,1:]) #first and second columns.....2nd item until the end
print(a[0:2,:]) #print everything
print(a[0:2,::-1]) #print both rows but in reverse order!

#we can also use the ... notation here. This is useful if we don't know how many rows or columns are contained in the
#ndarray object.
print(a[...,1])  #print everything in the 2nd column in all rows
print(a[:,1]) #This should be the same as the previous statement
print(a[1,...]) #print everything (all columns) in the 2nd row
print(a[...,1:])  #print everything from the 2nd column to the end...both rows
print(a[:,1:]) #should be identical to the previous statement
print(a[...,::-1]) #should print everything with the rows in reverse order

#more complex integer indexing
a = np.array([[211,212,213,214],[215,216,217,218], [219,220,221,222]])
b = a[[0,1,2], [0,1,0]]
print(b) #0,0   then 1,1 then 2,0   211, 216, 219 should be the result

#Now, let's experiment with the slice function
a = np.arange(10)
print(a)
s = slice(3,5) #non-inclusive of position 5
print(s)
#Then we can feed in the slice to the ndarray object
print (a[s])
print(a[3:5]) #Should give us the same result

s = slice(2, 7, 2) #index position starting at 2 and ending at 7 (non-inclusive) step 2
print(s)
print (a[s])
print(a[2:7:2]) #Should give us the same output

#We can also extract items from an ndarray using boolean indexing
a = np.array([[211,212,213,214],[215,216,217,218], [219,220,221,222]])
print(a)
boolean=(a>215)
print(boolean)  # will indicate true/false values for each item in the ndarray
print(a[boolean])
print(a[a>215])  #This should be identical to the previous statement

#NaN is not a number...let's print just the nan's and just the actual numbers
a = np.array ([np.nan,1,2,np.nan,3,4,5])
print(a)
boolean = np.isnan(a)
print(a[boolean])
print(a[np.isnan(a)])
boolean = ~np.isnan(a) #complement operator
print(a[boolean])
print(a[~np.isnan(a)])  #complement operator should yield just the numbers!

#Activities
#Using the following lists, create an ndarray of floats.
#Loop over the values in the third row and print the values.
row0 = [23, 89,100]
row1 = [10, 51, 99]
row2 = [40, 78, 102]
row3 = [35, 81, 110]
row4 = [50, 75, 95]
row5 = [65, 51, 101]

data = np.array([row0, row1, row2, row3, row4, row5], dtype=float)
print(data)
for i in data[2]:
    print(i)

#Create an ndarray object from 500 to 100 (not including the 100) stepping down 5 each time.
a = np.arange(500,100,-5)
print(a)

#Create an ndarray object of 15 evenly spaced numbers between 20 and 40 (inclusive of both)
#Print the ndarray and the step value
a = np.linspace(20,40,15, retstep=True, endpoint=True)
print(a[0])
print(a[1])

#Using the following ndarray object, ask the user for a value to search for within the ndarray.
#Then, print an appropriate message if the value is or is not found.  For instance,
#214 was found in [211 212 213 214 215 216 217 218 219 220] in position 3
#or if the value was not found
#222 was not found in [211 212 213 214 215 216 217 218 219 220]
#For this question, please handle the situation where the user might enter a non-numeric value.
a = np.array([211,212,213,214,215,216,217,218,219,220])
try:
    x = int(input('Please enter a number to search for:'))
    index = np.where (a == x)
    y=index[0]
    if y.size == 0:
        print(f'{x} was not found in {a}')
    else:
        print(f'{x} was found in {a} in position {int(index[0])}')
        # The following will return the integer value in the position in the ndarray that contains the value contained in the x variable
        #print(a[int(index[0])])
except:
    print('Please enter a number.')

#Add a row [217,218,219] to the following ndarray.
#Print the array when finished.
a = np.array([[211,212,213], [214,215,216]])
newarray = np.append(a,[[217,218,219]], axis = 0)
print(newarray)

#Using the following list, create an ndarray with a single axis.
#Then, print the following message:
#The Numpy array from Python list = [11 list([14, 15])]
#Then, print the second column of the ndarray.
#Then, print the first indexed item in the second column of the ndarray
#Finally, convert the ndarray back to a list and print out the list.
my_list = [11, [14,15]]
a = np.array(my_list, ndmin=1)
print(f'The Numpy array from Python list = {a}')
print(a[1]) #print the second column
print(a[1][1]) #print the first indexed element in the second column
my_newlist = a.tolist()  #convert the ndarray to a list
print(my_newlist) #print the list

#In the following list, create an ndarray and change the dimensions from 1 to 2
#Print the shape of the ndarray
#(1, 2)
#Then, print the following message:
#The Numpy array from Python list = [[11 list([14, 15])]]
#Then, print the following:
#[14, 15]
#15
#Finally, convert the ndarray to a list and print out the list
#[[11, [14, 15]]]
#Notice how the structure of the list is different from the original list

my_list = [11, [14,15]]
a = np.array(my_list, ndmin=2)
print(a.shape)
print(f'The Numpy array from Python list = {a}')
print(a[0][1])
print(a[0][1][1])
my_newlist = a.tolist() #Notice the new list structure has an extra nested list
print(my_newlist)

#Create an ndarray with the following tuple.
#Print out the following message:
#The NumPy array from Python Tuple = [11 22 33 44 55]
#Convert the ndarray back to a tuple and print out its contents
#(11, 22, 33, 44, 55)
my_tuple = (11, 22, 33, 44, 55)
a = np.array(my_tuple)
print(f'The NumPy array from Python Tuple = {a}')
my_newtuple = tuple(a.tolist())
print(my_newtuple)

#Create a two dimensional ndarray from the following tuple
#Print out the following message:
#The NumPy array from Python Tuple = [[11 22 33 (44, 46, 48) 55]]
#Then, convert the ndarray back to a tuple and print out its contents
#([11, 22, 33, (44, 46, 48), 55],)
#Notice how the structure of the tuple is slightly different from the original tuple.
my_tuple = (11, 22, 33, (44,46,48), 55)
a = np.array(my_tuple)
a = np.array(my_tuple, ndmin=2)
print(f'The NumPy array from Python Tuple = {a}')
my_newtuple = tuple(a.tolist())
print(my_newtuple)

#using the following ndarray, print the following:
#['John' (6.0, 7.0, 8.0)]
#John
#(6.0, 7.0, 8.0)
#Then, print the shape of the ndarray
x = np.array([('Sarah', (8.0, 7.0, 5.0)), ('John', (6.0, 7.0, 8.0))])
print(x[1])
print(x[1][0])  # Return John
print(x[1][1]) #Return the (6.0,7.0) tuple of values for John
print(x.shape)