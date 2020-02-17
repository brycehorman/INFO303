#Display a random fruit from a list of five fruits
import random
fruits = ['apple', 'pear', 'banana', 'grape', 'peach']
print(random.choice(fruits))


#Randomly choose either heads or tails ('h' or 't').  Ask the user to guess heads or tails.
#if they guess correctly, display the message 'You win!'
#Otherwise, display the message 'Bad luck!'
#At the end of your program, let the user know whether the computer randomly selected
#heads or tails.


import random
ht = ('heads', 'tails')
result = random.choice(ht)
user = input('Choose either heads or tails: ')
user = user.lower()
if user == result:
    print('You Win!')
else:
    print('Bad Luck!')
print('The computer chose',result)


#Randomly choose a number between 1 and 5.
#Ask the user to pick/guess the number.
#If they guess correctly, display the message, 'Well Done!'
#Otherwise, tell the user if they are too high or too low.
#Then, ask them to pick a second number.
#If their second guess is correct, display the message 'Correct!',
#otherwise display 'You lose!'
import random
num = random.randint(1,5)
user = int(input('Choose a number between 1 and 5: '))
if num == user:
    print('Well Done!')
else:
    if user > num:
        print('Too high')
    else:
        print('Too low')
    user = int(input('Choose another number between 1 and 5: '))
    if num == user:
        print('Correct!')
    else:
        print('You lose.')



#Randomly pick a whole number (integer) between 1 and 10.
#Ask the user to enter a number and keep entering
#numbers until they enter the number that was guessed
#When finished, tell the user how many guess attempts it took them to guess.
import random
num = random.randint(1,10)
user = int(input('Enter a number between 1 and 10: '))
count = 1
while num != user:
    user = int(input('Wrong! Enter a number between 1 and 10: '))
    count = count+1
print('Congrats, it only took you',count,'tries to get the right answer.')


#Create a void function called get_data, which will ask the user for their name and age.
#Then, print off a message dsplaying their name and their age.
#Finally, write a loop to call the function 3 times.
def getData():
    name = input('Enter your name: ')
    age = input('Enter your age: ')
    print(name,age)

for x in range(3):
    getData()



#Create a new function that returns a tuple of a users name and users age (as an integer).
#Then print off each element in the tuple on two lines.
def nameAge(name,age):
    info = (name, age)
    return info

name = input('Enter your name.')
age = int(input('Enter your age.'))
info = nameAge(name,age)
for x in info:
    print(x)

#Define a function named message that accepts two arguments (name and age)
#If the user passes in an age less than or equal to 10, print 'Hi {name}'
#Otherwise, print 'Hello {name}'
#Call the function with a few different name and age combinations.
#If the user does not pass an age argument to the function, use the value of 45.
def message(name,age = 45):

    if age <= 10:
        print('Hi',name)
    else:
        print('Hello',name)

message('Bryce', 21)
message('Linda')
message('John', 45)
message('Will', 9)



#Define a function that will ask the user to enter a number and save it as a variable.
#Define another function that will use num and count from 1 to that number.
#During the loop, print off a message.
#Call the function
def nums():
    num = int(input('Enter a number.'))
    return num

def count():
    num = nums()
    for x in range(1, num+1):
        print(x)
count()


#Create a function named msgtime that accepts two integer arguments
#The function should count from 1 to whatever number the user passes in for the first argument
#The second number should be the incrementer, which is the second argument of the function.
#For each loop, the function should print off a message.  If the user calls the function as
#msgtime(15,5) should yield something like the following:
#We are on the number 1
#We are on the number 6
#We are on the number 11
#msgtime(5, 2) should yield something like the following:
#We are on the number 1
#We are on the number 3
#We are on the number 5
def msgtime(num1, num2):
    for x in range(1, num1+1, num2):
        print('We are on the number',x)
msgtime(5,2)


#Create an anonymous lambda function that returns 'old' if the argument being passed to the function
#is greater than or equal to 50 and 'young' if the argument being passed to the function is less than 50.
#Use the lambda function to print off a message with a person's name, age, and age category (old or young).
#For instance,
#Bill Smith is 55 years old, which is categorized by our company as 'old'.
age = lambda x:  'old' if x >= 50 else 'young'
num = 55
print('Bill Smith is',num,'years old, which is categorized by our company as',age(num))



#Create an anonymous lambda function that takes in two numbers.
#The output of the function should be a multiplication of the two numbers.
#Call that lambda function with a couple of different pairs of numbers
nums = lambda x, y: x*y
print(nums(4,5))
print(nums(8,9))



#Import the functools module
#Use the reduce() function in the functools module to reduce a list to a single number
#Create a list object of numbers such as [1,2,3,4]
#Create a lambda function that accepts two numerical arguments and multiplies those
#two numbers together.  Then, print off the result.
#Then, create another lambda reduction function that accepts two numerical arguments and adds
#those two numbers together.  Finally, print off the result.
import functools
num1 = functools.reduce(lambda x,y: x*y, [1,2,3,4])
print(num1)
num2 = functools.reduce(lambda x,y: x+y, [1,2,3,4])
print(num2)

#Without using the csv module,
#create a csv file named salaries.csv in this directory
#The file should have a header row 'name,salary'
#Then, insert five comma separated dummy records on the next five lines of the file
#Then close the file
filename = 'salaries.csv'
handle = open(filename, 'w')
handle.write('Name,Salary\n')
handle.write('Bryce,100000\n')
handle.write('Jack,40000\n')
handle.write('Dustin,50000\n')
handle.write('Ben,70000\n')
handle.write('Tim,1000000\n')
handle.close()



#Now, open that file and append a single record to it. Again, do this without using the csv module
handle1 = open(filename, 'a')
handle1.write('Will,34000\n')
handle1.close()

#Create the following menu:
#Select one of the following numerical options:
#1) Add to the file
#2) View all of the records
#3) Delete a record
#4) Quit the program
import csv

file = 'salaries.csv'
def addtofile(file_name):
    handle = open(mode='a', file=file_name, newline='')
    name = input('Enter a name to add to the csv file:')
    salary = int(input('Enter the salary of the employee you want to add to the file:'))
    writer = csv.writer(handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([name, salary])
    handle.close()

def viewrecords(file_name):
    handle = open(mode='r', file=file_name)
    data = csv.reader(handle)
    for row in data:
        print(f'Employee {row[0]} has a salary of {row[1]}')
    handle.close()

def deleterecord(file_name):
    handle = open(mode='r', file=file_name)
    data = csv.reader(handle)
    tmplist = []
    for row in data:
        tmplist.append(row)
    handle.close()
    prompt = 'Pick a row number of the record that you want to delete: \n'
    x = 0
    for row in tmplist[1:]:
        x += 1
        prompt += str(x) + ') Employee Name: ' + row[0] + ' with a salary of ' + str(row[1]) + '\n'
    rowtodelete = int(input(prompt))
    del tmplist[rowtodelete]
    handle = open(mode='w', file=file_name, newline='')
    writer = csv.writer(handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in tmplist:
       writer.writerow(row)
    handle.close()

tryagain = True
while tryagain == True:
   msg = 'Select one of the following numerical options:\n\n'
   msg += '1) Add to the file\n'
   msg += '2) View all of the records\n'
   msg += '3) Delete a record\n'
   msg += '4) Quit the program\n'
   selection = int(input(msg))
   if selection == 1:
       addtofile(file)
   elif selection == 2:
       viewrecords(file)
   elif selection == 3:
       deleterecord(file)
   elif selection == 4:
       tryagain = False
   else:
       print('Incorrect selection.')


#Enter the number of your selection
#If the user selects 1, allow them to add to a file called salaries.csv, which will
#store the name and salary.  The function for this option should ask the user for the name
#and the salary of the record that should be added to the file.  Add the comma separated record.

# If they select 2, it should display all records in the salaries.csv file.  For each record in
# the file, print a user-friendly message such as 'Employee Tom has a salary of 85000.

# If they select 3, modify the salaries.csv file by deleting one of the records in the file.
#To do this, display all of the records and ask the user which row number they want to delete.
#Then, delete that record!

# If they select 4, your program should stop.

#if they select an incorrect option (not 1, 2, 3 or 4), they should see an error.
#The user should keep returning to the menu until they select option 4.

#Your program should have separate functions for options 1 to 3.

#Use the csv module for any reading, writing, and appending of data to the files for this problem





#Build a function named entry() for the loop to enter the program from the previous step
# and call that function from a namefunction.
def entry():
   tryagain = True
   while tryagain == True:
      msg = 'Select one of the following numerical options:\n\n'
      msg += '1) Add to the file\n'
      msg += '2) View all of the records\n'
      msg += '3) Delete a record\n'
      msg += '4) Quit the program\n'
      selection = int(input(msg))
      if selection == 1:
          addtofile(file)
      elif selection == 2:
          viewrecords(file)
      elif selection == 3:
          deleterecord(file)
      elif selection == 4:
          tryagain = False
      else:
          print('Incorrect selection.')

if __name__ == '__main__': entry()

