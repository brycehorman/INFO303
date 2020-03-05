#Today, we will continue working with numpy (numerical Python)
import numpy as np

#Let's iterate over the elements in our ndarray using the nditer function.
#This function will allow us to loop over the items across multiple dimensions
a = np.arange(0,90,10)
#reshape the nine elements into three rows and three columns
a = a.reshape(3,3)
#if we wanted to execute the previous two statements/expressions using a single statement, we could
#execute the following:
a = np.arange(0,90,10, dtype=float).reshape(3,3)
print(a)

#if we want to loop over a single row, we could do the following:
for x in a[2]:
    print(x)  #This should print the third row of data in our ndarray

#if we want to loop over a single column, we could do something like the following:
for x in a[:,2]: #This should print the second column in our ndarray object!
    print(x)

#A possibly easier way to loop over all of the data in our ndarray is to use the nditer function
#but we have to be careful of the order of the items/elements during the looping operation.

print(a)
#Let's loop across the rows and then down, which should be the default looping order
for x in np.nditer(a):
    print(x) #Pay careful attention to the order that the data get printed!

#Now, let's loop over the columns (down) and then rows (across)....all of column 0, then all of column 1, then all of column 2...
#This is an f-style or a Fortran style looping order.
print(a)
for x in np.nditer(a, order='F'):
    print(x)

#loop over rows (across) and then columns (down) (i.e., loop over all of row 0, then all of row 1, then all of row 2...)
# This is a c style looping order.  This sort order should be the default.
for x in np.nditer(a, order='C'):
    print(x)

#The nditer does not have a reverse parameter so this makes it a little tricky to loop over a sorted row or column.
print(a.shape) #the shape returns a tuple with rows, columns in this case
rows = a.shape[0]  #The rows should be in the first indexed position of the shapes tuple
print(rows)

#Now let's sort each row and print the results in either reverse or standard sorted order
for x in range(0,rows):
    b = np.sort(a[x])
    print(b) #Considering that each row is already sorted in ascending order, it is difficult to see this working!
    print(b[::-1])  #Now, print each row in reverse order!
    #Now let's print each item individually
    #for y in b: #ascending order
    for y in b[::-1]: #print each element in decending order
        print(y)

print(a)

#Now let's sort each column and print the results in either reverse or standard sorted order
cols = a.shape[1]  #Should give me the number of columns in our ndarray

for x in range(0,cols):
    #loop over each column and sort the data in each column
    b = np.sort(a[:,x])  #Here, the x is the column so this is grabbing all rows (:) in the column x, which will vary depending on the loop.
    print(b)
    print(b[::-1])  # Now, print each row in reverse order!
    #now print each item individually in standard or reverse order
    #for y in b:  #ascending order
    for y in b[::-1]: #descending order loop
        print(y)

#Let's add a third dimension to investigate how the nditer function is working
a = np.arange(0,180,10, dtype=float).reshape(2,3,3)
print(a)

#Loop across each row and then down...
# This should work as we would logically expect.  All of row 1 in frame 1 first, then row 2 in frame 1, etc.
#Then, it will print each row in frame 2.
for x in np.nditer(a, order='C'):
    print(x)

#Looping down and then across with three frames is not as intuitive
#Notice how it is not going straight down across the two frames (both first rows, both second rows, both third rows, etc.)
for x in np.nditer(a, order='F'):
    print(x)

#Let's add a fourth dimension!
a = np.arange(0,360,10, dtype=float).reshape(2,2,3,3)
print(a)

#Loop over each row first.  This works as we would logically expect!
for x in np.nditer(a, order='C'):
    print(x)

#What is the order that is going to displayed with this nditer loop?
#Notice the positions within each frame and metaframe.
print(a)
for x in np.nditer(a, order='F'):
    print(x)


#Let's reset our ndarray back to just rows and columns (3 by 3)
a = np.arange(0,90,10, dtype=float).reshape(3,3)
print(a)

#for x in np.nditer(a): #Notice the error because we are not setting the flag to readwrite during the loop!
for x in np.nditer(a, op_flags = ['readwrite']): #This works because we are setting the readwrite flag to readwrite mode during this loop.
    #x = 2 * x  # Note that this does not work because x is the iterable variable and not part of the a ndarray!
    #print(x)
    #print(x[...])
    x[...] = 2*x  #To change the actual a ndarray, use the following syntax!
print(a)

#We can broadcast two ndarrays of the same or of different sizes
a = np.arange(0,120,10)
a = a.reshape(3,4)
print(a)
b = np.arange(120,240,10).reshape(3,4)
print(b)
for x,y in np.nditer([a,b], order = 'C'): #array b is broadcast to size of a  #looping across and then down
    print('%d:%d' % (x,y))

#Now, let's make the b np ndarray a different size and execute the same loop
b = np.array([1,2,3,4], dtype = int)
#b = np.array([1,2,3], dtype = int) #now with a different number of columns than the a ndarray, we will probably get a broadcast error!
print(b)
for x,y in np.nditer([a,b], order = 'C'): #array b is broadcast to size of a  #loop across and then down
    print('%d:%d' % (x,y)) #Notice how the 1,2,3,4 get repeated because the sizes are different


#Now let's do some mathematical operations with ndarrays!
a = np.array([14.313, 94.13120, 224.121, 29.303, 44.9976])
print(np.around(a, 2))  #round each value to two decimal places
print(np.around(a, -1)) #round each value to the first place to the left of the decimal point!
print(np.floor(a)) #round down to the nearest whole number!
print(np.ceil(a)) #round up to the nearest whole number!

#For the next series of examples, let's use the following two dimensional numpy ndarray object
a = np.array([[12, 9, 40], [90, 53, 13], [29, 53, 11]])
print(a)

#pay careful attention to the axis that gets specified or omitted in these functions
print('The minimum element across both axis 1 and axis 2 is ', np.amin(a))
print('The maximum element across both axis 1 and axis 2 is ', np.amax(a))

#We can use a positional or a keyword argument to specify the axis of interest here
print(a)
print('The minimum element going down each column is ', np.amin(a, axis=0))
print('The maximum element going down each column is ', np.amax(a, 0))

print(a)
print('The minimum element going across is ', np.amin(a, 1))
print('The maximum element going across is ', np.amax(a, 1))

#The name of the function numpy.ptp() is derived from the name peak-to-peak.
# It is used to return the range of values along an axis. Consider the following example.
a = np.array([[12, 9, 40], [90, 53, 13], [29, 53, 11]])
print(a)

print('peak to peak (range) across both axis 1 and axis 2 is ', np.ptp(a))
print('ptp value along axis 1 is ', np.ptp(a, 1))  #difference among each between high and low values across columns within each row
print('ptp value along axis 0 is ', np.ptp(a, 0))  #difference among each between high and low values across rows within each column

#Let's calculate percentiles with this two dimensional numpy ndarray!
#The percentiles function takes three arguments:
#input: It is the input array.
#q: It is the percentile(1 - 100) which is calculated of the array element.
#axis: It is the axis along which the percentile is to be calculated.
a = np.array([[12, 9, 40], [90, 53, 13], [29, 53, 11]])
print(a)

print(np.percentile(a,100)) #all of the data across both axis 1 and axis 2 together
print('Percentile along axis 0', np.percentile(a, 100, 0)) #use the 50th and 100th percentile...going down
print('Percentile along axis 1', np.percentile(a, 100, 1)) #use the 50th and 100th percentile...going across

#We can also easily calculate a panel of descriptive statistics
a = np.array([[12, 9, 40], [90, 53, 13], [29, 53, 11]])
print(a)
print(np.median(a))
print(np.median(a, 0))  #going down
print(np.median(a, 1))  #going across
print(np.mean(a))
print(np.mean(a, 0))  #going down
print(np.mean(a, 1))  #going across
#we can calculate a weighted average using the average function
print(np.average(a))  #all data
print(np.average(a, 1)) #This is a straight average across the first axis, which is going across
wts = np.array ([3,2,1])
print(np.average(a,1,weights=wts)) #going across....
print(np.average(a,0,weights=wts, returned = True))  #going down...returned is the sum of the weights
print(np.std(a))  #across all of the data regardless of the axis
print(np.std(a, axis=1)) #going across
print(np.std(a, axis=0)) #going down
print(np.var(a)) #across all of the data regardless of the axis
print(np.var(a, axis=1)) #going across
print(np.var(a, axis=0)) #going down
print(np.sum(a)) #sums all of the values
print(np.sum(a,axis=0)) #Sum of each column ...going across
print(np.sum(a,axis=1)) #Sum of each row...going down

#take the reciprocal of a number
a = np.array([0.25, 1.33,1,0,100])
print(np.reciprocal(a))

#We can use the power function to raise each item in a numpy ndarray to a specific power.
a = np.array([4,8,12])
print(np.power(a,2))
print(np.power(a,3))
#We can specifiy the pwoers in a separate ndarray
b = np.array ([1,2,3])
print(np.power(a,b)) #4 raised to the 1st power, 8 raised to the 2nd power, 12 raised to the third power

#We can calculate the mod or the remainder using the mod and the remainder functions
a = np.array([10,20,30])
#The following two statements should be identical
print(np.mod(a,4))
print(np.remainder(a,4))

b = np.array([3,5,7])
#The following two statements should be identical
print(np.mod(a,b)) #10 divided by 3, 20 divided by 5, 30 divided by 7....then determine the remainder
print(np.remainder(a,b))

#We can also perform mathematical operations across multiple ndarrays
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])
print(np.add(a,b)) #a+b does the same
print(a+b)

print(a)
print(b)
print(np.subtract(a,b)) #Same as a-b
print(a-b)
#obviously the direction of the subtraction matters!
print(np.subtract(b,a)) #Same as b-a
print(b-a)

print(a)
print(b)
print(np.multiply(a,b)) #a*b works too
print(a*b)

print(a)
print(b)
print(np.divide(a,b)) #Same as a/b
print(np.divide(b,a))

#We must be careful when the numpy ndarrays have different shapes. Consider the following example:
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9]])
print(a)
print(b)
print(a-b)
print(a+b)
print(a*b)
print(a/b)

#We can also perform matrix algebra on matrices.  For this example, let's use the following ndarrays:
x=np.array([[1,2],[3,4]])
y=np.array([[5,6],[7,8]])
v=np.array([9,10])
w=np.array([11,12])
print(v)
print(w)
#dot allows us to multiply matrices
print(v.dot(w)) #Same as np.dot(v,w)
#=9*11 +10*12

print(x)
print(v)
print(x.dot(v)) #same as np.dot(x,v)
#=1*9 + 2*10=29....3*9 + 4*10=67
print(x)
print(y)
print(np.dot(x,y))
#=1*5+2*7 = 19,  1*6+2*8 = 22, 3*5+4*7 = 43, 3*6+4*8 = 50

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(a)
print(b)
print(np.dot(a,b))
#[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]

#Now let's use numpy to make a few plots using matplotlib
#Matplotlib is a plotting library for Python.  It allows to make certain types of plots along multiple axises.
#It integrates well with numpy ndarray objects.
#Matplotlib was originally written by John D. Hunter.
# Michael Droettboom and Tom Caswell are the primary developers now (from ~2012).

#Depending on your Python installation, you will probably have to install it!
#python -m pip install matplotlib --user
#Note: the --user is only necessary on the lab machines if you don't have read/write permissions
#on pythons home directory.
#import matplotlib
#or

plt.savefig('testing2.png')
from matplotlib import pyplot as plt
import numpy as np
#pyplot is the single most important function in matplotlib. It is used for 2d plotting!
#let's plot a basic line!
#y = 4x + 8
x = np.arange(1,11)
y = 4*x + 8
plt.clf() #clear the existing plot (if there is one). This will clear an open window but it will also open up another window if one is not already open.
plt.title('Sample Line Graph')
plt.xlabel('x axis label caption')
plt.ylabel('y axis label caption')
#plt.plot(x,y)
plt.plot(x,y, color='Red', linestyle='dashed') #solid #dotted #dashed #dashdot
plt.legend(('Series line'), loc='lower right') #lower right....upper left
plt.annotate(xy=[2,15], s='Sample Annotation')

plt.show()
x = np.arange(1,11)
y = 4*x + 8
plt.clf()
plt.title('Sample Dot Graph')
plt.xlabel('x axis label caption')
plt.ylabel('y axis label caption')
#plt.annotate(xy=[2,15], s='Sample Annotation')
#plt.text(2,20,'This text starts at point (2,20)')
#plt.text(8,30,'This text ends at point (8,30)',horizontalalignment='right')

plt.xticks(np.arange(2,11,2))
plt.yticks(np.arange(5,50,5))

#plt.plot(x,y,'bo')
#plt.plot(x,y,'r+')
#what does zip do?
print(x)
print(y)

result = list(zip(x, y))
print(result)
resultlist = list(result) #Create a list of tuples containing the x and y coordinates of our data points
print(resultlist)

for i,j in resultlist:

    plt.annotate('(%s, %s)' % (i, j),  # this is the text
                 xy=(i, j),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 color='r')  #Color the labels in red font

plt.plot(x,y, '^g:', linewidth=2, markersize=12)
plt.show()

#Color options in plt.plot
#'b' 	blue
#'g' 	green
#'r' 	red
#'c' 	cyan
#'m' 	magenta
#'y' 	yellow
#'k' 	black
#'w' 	white

#Sample options
#'b'    # blue markers with default shape
#'or'   # red circles
#'-g'   # green solid line
#'--'   # dashed line with default color
#'^k:'  # black triangle_up markers connected by a dotted line

#Line Styles
#character 	description
#'-' 	solid line style
#'--' 	dashed line style
#'-.' 	dash-dot line style
#':' 	dotted line style

#Let's now construct a bar chart
x = np.array([5,8,10])
y = np.array([12,16,8])
plt.clf()
plt.bar(x,y, color='r', align='center', width=0.75)
plt.title('Single Bar Chart Example')
plt.ylabel('Y axis label')
plt.xlabel('X axis label')

#Add the data labels to our chart
for i,j in zip(x, y):
    plt.annotate('%s' % j,  # this is the text
                 xy=(i, j),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 5),  # distance from text to points (x,y)
                 ha='center',  # horizontal alignment can be left, right or center
                 color='k')  # Color the labels in black font

plt.show()

#Now let's add two side by side bar charts with multiple x,y combinations
x = np.array([5,8,10])
y = np.array([12,16,8])
x2 = np.array([6,9,11])
y2 = np.array([6,15,7])
plt.clf()
plt.bar(x,y, color='r', align='center', width=0.75)
plt.bar(x2,y2,color='g', align = 'center', width=0.75)
plt.title('Side by Side Bar Chart Example')
plt.ylabel('Y axis label')
plt.xlabel('X axis label')

#Add the data labels to our chart
for i,j in zip(x, y):
    plt.annotate('%s' % j,  # this is the text
                 xy=(i, j),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 5),  # distance from text to points (x,y)
                 ha='center',  # horizontal alignment can be left, right or center
                 color='k')  # Color the labels in black font

for i,j in zip(x2,y2):
    plt.annotate('%s' % j,  # this is the text
                 xy=(i, j),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 5),  # distance from text to points (x,y)
                 ha='center',  # horizontal alignment can be left, right or center
                 color='k')  # Color the labels in black font

plt.show()

#Let's plot multiple series on line graph
x = np.arange(1, 11)
plt.clf()
plt.plot(x, [xi*1 for xi in x], linestyle='solid', color='Blue')
plt.plot(x, [xi*2 for xi in x], linestyle='dashed', color='Orange')
plt.plot(x, [xi*3 for xi in x], linestyle='dotted', color='Green')
plt.legend(['growth projection 1', 'growth projection 2', 'growth projection 3'], loc='upper right')

plt.grid()
plt.axis([0, 15, 0, 35])
plt.xlabel('Sample X units')
plt.ylabel('Sample Y units')
plt.title('My Beautiful Chart Title')
plt.rcParams['figure.figsize'] = (17,7) #figsize should be in inches!
plt.show()

#Let's build a sin wave curve
x = np.linspace(0, 2 * np.pi, 400)  #Build a new ndarray of 400 numbers between 0 and 2*pi
y = np.sin(x ** 2)  #Build a new ndarray where I square each value in the x ndarray
plt.clf()
plt.plot(x, y)
plt.title('A single wave plot')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.rcParams['figure.figsize'] = (12,6) #figsize should be in inches!
plt.show()


#Let's stack two charts vertically!
plt.clf()
fig, axs = plt.subplots(2)  #The two in this context will result in two rows
fig.suptitle('Vertically stacked subplots')
axs[0].plot(x, y, 'ro')
axs[0].legend(['series 1'], loc='upper left')
axs[1].plot(x, -y, 'b+')
axs[1].legend(['series 2'], loc='upper left')

#The next block of code should display the same result
fig, (ax1, ax2) = plt.subplots(2)  #Notice how the left side of the equation is different.
fig.suptitle('Vertically stacked subplots')
ax1.plot(x, y, 'ro')
ax1.legend(['series 1'], loc='upper left')
ax2.plot(x, -y, 'b+')
ax2.legend(['series 2'], loc='upper left')

#The next block of code should display two charts side by side
fig, (ax1, ax2) = plt.subplots(1, 2)  #1 row and 2 columns, which should put the two charts side by side
fig.suptitle('Horizontally stacked subplots')
ax1.plot(x, y, 'ro')
ax1.legend(['series 1'], loc='upper left')
ax2.plot(x, -y, 'b+')
ax2.legend(['series 2'], loc='upper left')

#The next block of code should display four charts (two rows by two columns)
fig, axs = plt.subplots(2, 2) #2 rows by 2 columns so we will display four charts!
axs[0, 0].plot(x, y, color = 'Orange')
axs[0, 0].set_title('Axis [0,0]')
axs[0, 1].plot(x, y**2, color = 'Blue')
axs[0, 1].set_title('Axis [0,1]')
axs[1, 0].plot(x, -y, color = 'Green')
axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].plot(x, -y**2, color = 'Red')
axs[1, 1].set_title('Axis [1,1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.clf()  #Clear the existing window if it is open

#The following should result in the same result
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y, color = 'Orange')
ax2.plot(x, y**2, color='Blue')
ax3.plot(x, -y, color='Green')
ax4.plot(x, -y**2, color='Red')

ax1.set(xlabel='x-label', ylabel='y-label')
ax2.set(xlabel='x-label', ylabel='y-label')
ax3.set(xlabel='x-label', ylabel='y-label')
ax4.set(xlabel='x-label', ylabel='y-label')

for ax in fig.get_axes():
    ax.label_outer()


#####START HERE

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Axes values are scaled individually by default')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2)
axs[0].legend(['series 1'], loc='upper left')
axs[1].plot(x, 0.3 * y, 'o')
axs[1].legend(['series 2'], loc='upper left')
axs[2].plot(x, y, '+')
axs[2].legend(['series 3'], loc='upper left')


fig, axs = plt.subplots(3, sharex=True, sharey=True, gridspec_kw={'hspace': 1, 'height_ratios':[5,5,4]})
fig.suptitle('Sharing both axes')
axs[0].plot(x, y ** 2, 'r-')
axs[1].plot(x, 0.3 * y, 'bo')
axs[2].plot(x, y, 'c+')



fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 1, 'wspace': 1})
(ax1, ax2), (ax3, ax4) = axs
fig.suptitle('Sharing x per column, y per row')
ax1.plot(x, y, color='Orange')
ax2.plot(x, y**2, color='Blue')
ax3.plot(x + 1, -y, color='Green')
ax4.plot(x + 2, -y**2, color='Red')

for ax in axs.flat:
    ax.label_outer()

#Let's build a histogram and plot it
a = np.array([22,87,5,41,43,56,51,73,55,54,62,11,20,49,51,5,79,31,27,49,69])
np.histogram(a,bins=[0,20,40,60,80,100])
#we can unpack this to two variables
hist,bins=np.histogram(a,bins=[0,20,40,60,80,100])
print(hist)
print(bins)
plt.hist(a, bins=[0,20,40,60,80,100], color='red')
plt.title('Sample Histogram')
my_list = list(zip(hist, bins))
print(my_list)
for i,j in zip(bins, hist):
    plt.annotate('%s' % j, xy=(i,j), xytext=(30,5), textcoords='offset points', color='k')
plt.show()




import matplotlib.pyplot as plt
import numpy as np

plt.clf()

# using some dummy data for this example
xs = np.arange(0,10,1)
ys = np.random.normal(loc=3, scale=0.4, size=10)

plt.xlabel('X Units')
plt.ylabel('Y units')
plt.title('Testing Title')

# 'bo-' means blue color, round points, solid lines
plt.plot(xs,ys,'bo-')

# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = "${:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,7,0.5))

plt.show()


import numpy as np

plt.clf()

# using some dummy data for this example
xs = np.random.normal(loc=4, scale=2.0, size=10)
ys = np.random.normal(loc=2.0, scale=0.8, size=10)

plt.scatter(xs,ys)

# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xticks(np.arange(0,10,1))
plt.yticks(np.arange(0,5,0.5))

plt.show()