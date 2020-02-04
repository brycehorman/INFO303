#In Python, a function is a group of related statements that perform a specific task.

#Functions help break our program into smaller and modular chunks.
# As our scripts grow larger and larger, functions make them more
# organized, readable, and manageable.

#Functions also avoid repetition, which makes code more reusable and less error proned.

#Basically, we can divide functions into the following two types:
#Built-in functions - Functions that are built into Python.
#User-defined functions - Functions defined by the users themselves.

#def marks the start of function header.
#A function name to uniquely identify it. Function naming follows the same rules of writing identifiers in Python.
#Parameters are contained in the ()s and are used to pass values to a function. They are optional.
#A colon(:) is used to mark the end of the function header.
#Optional (but best practice) documentation uses a docstring to describe what the function does.
#Then, we have the body of the function, which contains one or more valid python statements that
#make up the function body. Statements must have the same indentation level(usually 4 spaces).
# An optional return statement will return a value from the function.

#The def statement creates the function, but it is not executed until it is actually called.
#Notice what happens when we fail to pass in a argument.
def hellomsg (name):
	"""This function prints a message
		based on the name parameter"""
	print(f'Hello, {name.title()}. Today is a great day to learn some Python!!')

hellomsg('grant williamson')
#hellomsg()

#The following will print off a description of the function
print(hellomsg.__doc__)

#let's build an absolute value function that takes in a
#num parameter
def cube_val (num):
	"""This function returns the absolute value of the cube
	of the num parameter"""
	if num >= 0:
		return num*num*num
	else:
		return -(num*num*num)

print(cube_val(2))

x = cube_val(-4)
print(x)

#Scope of a variable is the portion of a program where the variable is recognized.
# Parameters and variables defined inside a function are not visible from outside.
# Hence, they have a local scope.

#Lifetime of a variable is the period throughout which the variable exists in the memory.
# The lifetime of variables inside a function only as long as the function executes.

#They are destroyed once we return from the function.
# Hence, a function does not remember the value of a variable
# from its previous calls.

def my_custfunc():
	#i has local scope
	#it is only available for use in the function
	i = 23
	print('Value inside function:{}'.format(i))

#Another x variable but this variable is defined outside of the function
i = 207
my_custfunc()

#The i in the next statement references the i is defined outside of the function
print('Value outside function:{}'.format(i))

def hellomsg2 (name, msg = "Have a restful evening!"):
   """
   This function greets the person with a message but
   If message is not provided, then it will default to
   "Have a restful evening!"
   """
   print(f'Hello {name.title()}, {msg.lower()}')

# 1 positional, 1 keyword argument
hellomsg2 ('Mitt Smith')
hellomsg2 ('sue jones','I hope you are doing well')
hellomsg2 (name = 'graham', msg = 'Have a wonderful day')
hellomsg2 (msg = 'Have a wonderful day', name = 'graham')

def hellomsg3 (*names):
   """This function greets all the people
   in the names tuple parameter...
   This parameter is referred to as an *args parameter with the unpacking operator."""

   # names is a tuple so we need to loop over each element/item
   for name in names:
	   print('Hola {x}, como estas?'.format(x = name.title()))

hellomsg3 ('tim','dane','jj','jillian','amit','violet','dane', 'patrick')

#Recursion is the process of defining something in terms of itself.
#A physical world example would be to place two parallel mirrors
# facing each other. Any object in between them would be reflected recursively.

# An example of a recursive function to
# find the factorial of a number

#Following is an example of recursive function to find the factorial of an integer.
#Factorial of a number is the product of all the integers from 1 to that number.
# For example, the factorial of 6 (denoted as 6!) is 1*2*3*4*5*6 = 720.
def calc_factorial(x):
	"""This is a recursive function
	to find the factorial of an integer"""

	if x == 1:
		return 1
	else:
		return (x * calc_factorial(x-1))

num = 4
print('The factorial of %s is %s' % (num, calc_factorial(num)))

#Our recursion ends when the number reduces to 1. This is called the base condition.
#Every recursive function must have a base condition that stops the recursion
# or else the function calls itself infinitely.

#How is this working
#calc_factorial(4)              # 1st call with 4
#4 * calc_factorial(3)          # 2nd call with 3
#4 * 3 * calc_factorial(2)      # 3rd call with 2
#4 * 3 * 2 * calc_factorial(1)  # 4th call with 1
#4 * 3 * 2 * 1                  # return from 4th call as number=1
#4 * 3 * 2                      # return from 3rd call
#4 * 6                          # return from 2nd call
#24                             # return from 1st call
#Verify that it is working using the built-in factorial function

import math
print(math.factorial(4))

#Advantages of Recursion
#Recursive functions make the code look clean and elegant.
#A complex task can be broken down into simpler sub-problems using recursion.
#Sequence generation is easier with recursion than using some nested iteration.

#Disadvantages of Recursion
#Sometimes the logic behind recursion is hard to follow through.
#Recursive calls are expensive (inefficient) as they take up a lot of memory and time.
# Recursive functions are hard to debug.

#In Python, anonymous function is a function that is defined without a name.
#While normal functions are defined using the def keyword, in Python anonymous
# functions are defined using the lambda keyword.
# Hence, anonymous functions are also called lambda functions.

# Program to show the use of lambda functions
#In the above program, lambda x: x * 2 is the lambda function.
# Here x is the parameter and x * 2 is the expression that gets evaluated and returned.
#This function has no name. It returns a function object which is assigned to the
# identifier double. We can now call it as a normal function.
triple = lambda x: x * 3
print(triple(6))
print(type(triple))
#The following function privides the same functionality
def triple(x):
   return x * 3

#We use lambda functions when we require a nameless function for a short period of time.
#In Python, we generally use it as an argument to a higher-order function
# (a function that takes in other functions as arguments).
# Lambda functions are used along with built-in functions like filter(), map(), etc.

#The filter() function in Python takes in a function and a list as arguments.
#The function is called with all the items in the list and a new list is
# returned which contains items for which the function evaluates to True.
#Here is an example use of filter() function to filter out only even numbers from a list.

# Program to filter out only the even items from a list

my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(filter(lambda x: (x%2 == 0) , my_list))
print(new_list)
#The lambda function is functionally equivalent to the following:
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
def myFunc (x):
  if x%2 == 0:
	return True
  else:
	return False

new_list = list(filter(myFunc, my_list))
print(new_list)

#The map() function in Python takes in a function and a list.
#The function is called with all the items in the list and
# a new list is returned which contains items returned
# by that function for each item.

#Here is an example use of map() function to double all the items in a list.
# Program to double each item in a list using map()

my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(map(lambda x: x * 2, my_list))
print(new_list)

my_list = [1, 5, 4, 6, 8, 11, 3, 12]
def myFunc (x):
	return x*2

new_list = list(map(myFunc, my_list))
print(new_list)

#In Python, a variable declared outside of the function
# or in global scope is known as global variable.
# This means, global variable can be accessed inside or outside of the function.

x = 'global'

def foo():
	print('x inside the function: ' + str(x))

foo()
print('x outside the function: ' + str(x))

#global variable
c = 7
def add():
	global c
	c = c + 2
	print('Inside the add() function:', c)

add()
print('In main script outside of the add() function:', c)

x = 6
def foo():
	x = 20
	def bar():
		global x
		x = 25
	print('Before calling bar: ', x)
	print('Calling bar now')
	bar()
	print('After calling bar: ', x)

foo()
print('x in the main script: ', x)

x = 'global'
def foo():
	global x
	y = 'local'
	x = x * 2
	print(x)
	print(y)

foo()
print('x in the main script is: ', x)

x = 5
def foo():
	x = 10
	print('local x:', x)
foo()
print('global x:', x)

#Nonlocal variables are used in nested function
# whose local scope is not defined.
# This means, the variable can be neither
# in the local nor the global scope.

#Let's see an example on how a global variable is created in Python.
#We use nonlocal keyword to create nonlocal variable.
x = 'global'
def outer():
	x = 'local'
	def inner():
		#non-local: where the variable should not belong to the inner function.
		nonlocal x
		x = 'nonlocal'
		print('inner:', x)
	inner()
	print('outer:', x)

outer()
print('global:', x)

#In the above code there is a nested function inner().
# We use nonlocal keyword to create nonlocal variable.
# The inner() function is defined in the scope of another function outer().
#Note : If we change value of nonlocal variable, the changes appears in the local variable.


#Activity...Let's build a program for a color guessing (mastermind) game.
#The computer will automatically generate four colors from a list of possible colors.
#It should be possible for the computer to randomly select the same color more than once in the list of four colors.
#For instance, the computer may choose 'red','blue','red','green' in that order.
# The specific sequence of the computer's selections should not be displayed to the user
# because the user will attempt to guess the colors in the correct order.
import random
def generateColors():
	colors = ['red','blue','orange','yellow','green','purple','white','black']
	c1 = random.choice(colors)
	c2 = random.choice(colors)
	c3 = random.choice(colors)
	c4 = random.choice(colors)
	col = [c1, c2, c3, c4]
	return col

def userChoice():
	colors = []
	for x in range(4):
		colors.append(input('Enter a color (red, blue, orange, yellow, green, purple, white, black').lower())
	return colors

def compare(computer, user):
	corr = 0
	for x in computer:
		for y in user:
			if y == x:
				corr = corr + 1
	return corr

def comparePos(computer, user):
	corr = 0
	for x in range(4):
		if computer[x] == user[x]:
			corr = corr + 1
	return corr

correctInPos = 0
computer = generateColors()
print(computer)
while correctInPos < 4:
	user = userChoice()
	correctInPos = comparePos(computer, user)
	correctInGen = compare(computer, user)
	print('You got' , correctInGen, 'colors correct in general, and', correctInPos , 'colors correct in position')
print('Congrats, you got all the colors in the right position.')
#After the computer makes its four color selections, the user should enter
# their choice of four colors from the same list that the computer used.
# For instance, the user may choose pink, blue, yellow, and red.  The user should
#be given four inputs to select their four colors.  The sequence of the user's selections matters because
#the user is attempting to match the colors in the correct order.

#After the user has made their four color selections, your program should show how many colors they got
# right in the correct position and how many colors they got right but in the wrong position.

#For instance,
#Computer randomly picks: 'red', 'blue', 'red', 'blue'
#User enters: 'pink', 'blue', 'yellow', 'red'

#In the above example, the program should display messages similar to the following:
# 'Correct color in the correct place: 1' and
#'Correct color but the wrong place: 1'

#The game should continue until the user correctly enters the four colors in the proper order.
#At the end of the game, your program should display a suitable
#message and tell them how many guesses they took to get the correct answer.

#This task will require your to split up the tasks into separate functions,
#use lists, use the random module, use if statements, loops, and other skills!

#The colors for this game are the following:
#red, blue, orange, yellow, pink, green, and white

#Have the user just enter the first letter of the colors instead of the full color names.

#Once the computer randomly picks its four choices, those are the four colors for the
#duration of the game.