#1. Define a variable named x and set its value to 1.
#   Loop over x while x is less than or equal to 10.
#   For each loop, print the current value of x and increment x by 1 before starting the next loop.
x = 1
while(x <= 10):
    print(x)
    x = x+1



#2. Define a variable named x and set its value to 10.
#   Loop over x while x is less than or equal to 100.
#   In each loop, print the value of x and increment x by 10 before starting the next loop.
#   Break out of the loop if/when x equals 30.
x = 10
while(x <= 100):
    print(x)
    x = x + 10
    if(x == 30):
        break



#3. Define a variable named x and set its value to 1000.
#   Loop over x while x is greater than or equal to 100.
#   In each loop, reduce the value of x by 50.
#   If the value of x (after subtracting 50) is between 400 and 600, continue to the next loop.
#   Otherwise, print a message that includes the value of x.
x = 1000
while x >= 100:
    x = x - 50
    if x > 400 and x < 600:
        continue
    else:
        print('The value of x is',x)



#4. Within a loop, have the computer pick a random number between 1 and 20.
#   Loop until the random integer that the computer selected is 17.
#   Each time through the loop, print out how many times the loop has executed.
import random
num = 0
count = 0
while num != 17:
    num = random.randint(0,20)
    count = count + 1
    print('The loop has executed',count,'times. The current guess is',num)


#5. Using the below list, print the min, max, mean, standard deviation, median, and mode of the following list.
import statistics
tm_list = [500,500,500,511,500,532,491,522,522,527,528,529,530,533,577,594,599,587,588,590]
print(max(tm_list))
print(min(tm_list))
print(statistics.mean(tm_list))
print(statistics.stdev(tm_list))
print(statistics.median(tm_list))
print(statistics.mode(tm_list))

#6. Using the below list of professors, loop over each and print a user-friendly message such as the following:
#Hello Professor Johnson. Can I talk to you about my grade in this class?
professors = ['johnson', 'jones', 'samson', 'baily', 'nelson']
for x in professors:
    print('Hello Professor',x.capitalize() + '. Can I talk to you about my grade in this class?')

#7. Using the below list of professors, permanently sort them in reverse order and print out the result.
#   Your printed message should be something like the following:
#   I am looping over Professor Baily now.
professors = ['johnson', 'jones', 'samson', 'xie', 'baily', 'nelson']
professors.reverse()
for x in professors:
    print('I am looping over Professor',x.capitalize(),'now.')


#8.  Using the below list of professors, loop over the professors in reverse order (z to a)
#    without permanently changing the order of the list.
#    Your printed message should be something like the following:
#    I am looping over Professor Xie now.
#    After you are finished looping over each individual professor, print the list to show that the
#    order has not changed.
professors = ['johnson', 'jones', 'samson', 'xie', 'baily', 'nelson']
for x in sorted(professors, reverse=True):
    print('I am looping over Professor',x.capitalize(),'now.')
print(professors)


#9.  Using the below list of professors, loop over samson, xie, and baily.
#    Your printed message should be something like the following:
#    I am looping over Professor Xie now.
professors = ['johnson', 'jones', 'samson', 'xie', 'baily', 'nelson']
for x in range(2,5):
    print('I am looping over Professor',professors[x].capitalize(),'now')


#10.  Using the below list, change the 78 to 99 and the blue to orange.
#     Then, print the list to verify that your change was successful.
various_stuff = ['tom', 'mattson', [2,5,84,78], {'1A': 'red', '2A': 'blue', '3A': 'green'}]
various_stuff[2][3] = 99
various_stuff[3]['2A']= 'orange'
print(various_stuff)

#11.  Using the below tuple, print the index of the 'xie' element
professors = ('johnson', 'jones', 'samson', 'xie', 'baily', 'nelson')
print(professors.index('xie'))

#12.  Using the below tuple, clear the items from the [10, 144, 128, 237] list.
#     Then, print the entire tuple to verify that your change was successful.
#     Next, insert four random real numbers between 0 and 99.99 to the list that you just cleared.
#     Only use four decimal places of precision for the values that you add.
#     Then, print the entire tuple to verify that your change was successful.
tm_tuple = ('help me', [10, 144, 128, 237], (21, 56, 34, 23, 29, 33, 55))
tm_tuple[1].clear()
print(tm_tuple)
for x in range(4):
    rand = round(random.random()*100, 4)
    tm_tuple[1].append(rand)
print(tm_tuple)

#13.  Using the following tuple, check to see if the number 44 is contained in it.
#     If so, print an appropriate user-friendly message. If not, print an appropriate user-friendly message.
#     Your message might be something like:
#     The number 44 is not contained in the tuple, which contains (21, 56, 34, 23, 29, 33, 55)
tm_tuple = (21, 56, 34, 23, 29, 33, 55, 44)
if 44 in tm_tuple:
    print('The number 44 is in this tuple.')
else:
    print('The number 44 is not in this tuple.')


#14.  Using the following two tuples, print a message indicating whether one of the requested toppings
#     is available or not.  Your printed messages should be something like the following:
#     We will be happy to add 'Olives' to your pizza.
#     We are currently out of 'Onions' so we cannot add it to your pizza.
requested_toppings = ('Pepperoni', 'Sausage', 'Olives', 'Onions')
not_available = ('Sausage', 'Onions', 'Ham')
for x in requested_toppings:
    if x in not_available:
        print('We are currently out of',x,'so we cannot add it to your pizza.')
    else:
        print('We will be happy to add',x,'to your pizza')

#15.  Using the below tuple, determine how many toppings the user requested.  If the customer
#     requested more than 3 toppings, inform the customer that there will be an extra $2.00 charge.
requested_toppings = ('Pepperoni', 'Sausage', 'Olives', 'Onions')
num = len(requested_toppings)
if num > 3:
    print('There will be an extra $2 charge.')



#16.   Using the below string, create a tuple containing three items (empid, empname, and email).
#      Then, print the items in the tuple to verify that it worked!
#      Also, print the data type of your collection to ensure that it is a tuple and not a list.
line = 'empid;empname;email'
items = (line[:5], line[6:13], line[14:])
print(items)
print(type(items))


#17.   Using the below tuple, check whether each item is a number or not.
#      Then, print an appropriate user-friendly message such as the following:
#      joe is not a number.
#      72 is a number.
#      HINT: The isinstance function might be helpful here!
tm_tuple = ('tim', 'joe', 72, 33, 47, 47.6, 'Bill')



#18. Using the following tuple:
tm_tuple = (118,[18,12,212],5,[2,31,4])
#Change the 31 to a 33
#Then, calculate the sum of all of these numbers (which is 404) and print out the sum in a user-friendly manner.
#  For instance, The sum of all of these numbers is 404.



#19.  Using the following dictionary, change the shoe_size from 5 to 6.
#     Then, print out the dictionary.
my_dict = {'age':22, 'shoe_size': 5, 'shirt_size':'medium', 'hat_size':7}



#20.   Using the following dictionary, loop over all of the key-value pairs and print the following message:
#The details for this customer are as follows:
#	age: 33
#	shoe_size: 5
#	shirt_size: medium
#	hat_size: 9
my_dict = {'age':33, 'shoe_size': 5, 'shirt_size':'medium', 'hat_size':9}



#21.   Using the following dictionary, loop over just the keys and print each key on a separate line.
#     Age
#     Shoe_Size
#     Shirt_Size
#     Hat_Size
my_dict = {'age':33, 'shoe_size': 5, 'shirt_size':'medium', 'hat_size':9}



#22.   Using the following dictionary, calculate an average of the shirt_sizes.
#     Then, print out the average.
my_dict = {'age':22, 'shoe_size': 5, 'shirt_sizes':[14,18,22,19], 'hat_size':7}



#23.   Create an empty dictionary named game.
#   Add a key-value pair for 'x_pos' and the value of 0
#   Add a key-value pair for 'y_pos' and the value of 10
#   Add a key-value pair for 'z_pos' and the value of 20
#   Then, print the dictionary!
#   Next, ask the user to enter a speed (high, medium or low).
#   Based on the speed that the user enters, add 10 to each pos key-value pair if the user enters low,
#   add 20 to each key-value pair if the user enters medium,
#   or add 30 to each key-value pair if the user enters high.
#   Finally, print the dictionary again!


#24. Using the following dictionary object, remove the speed - medium key value pair.
#    Then, print the dictionary.
my_dict = {'x_pos': 20, 'y_pos': 30, 'z-pos': 40, 'speed': 'medium'}



#25.  Using the following dictionary object, add 0.5 to all INFO201 grades.
#    Print out the dictionary when you are finished
grades = {
    'sally': [{'MGMT375':3.75}, {'INFO201':3.25}],
    'pete': [{'MGMT375':3.75}, {'INFO201':3.25}, {'ART202':2.75}],
    'ally': [{'MGMT375':4.00}, {'ACCT201':3.75}],
    'ed': [{'MGMT375':3.75}, {'INFO201':3.25}, {'ART202':2.75}, {'ACCT201':4.00}]
}



#26.  Using the following dictionary object, print the course and grade combination
#   where the student earned their highest grade.  Your output should be as follows:
#Sally earned his/her highest grade of 3.75 in MGMT375.
#Pete earned his/her highest grade of 3.75 in MGMT375.
#Ally earned his/her highest grade of 4.0 in MGMT375.
#Ed earned his/her highest grade of 4.0 in ACCT201.
grades = {
    'sally': [{'MGMT375':3.75}, {'INFO201':3.25}],
    'pete': [{'MGMT375':3.75}, {'INFO201':3.25}, {'ART202':2.75}],
    'ally': [{'MGMT375':4.00}, {'ACCT201':3.75}],
    'ed': [{'MGMT375':3.75}, {'INFO201':3.25}, {'ART202':2.75}, {'ACCT201':4.00}]
}



#27.  Build a void function named rollercoaster that accepts no arguments.
#    This function should asks the user to enter their height in inches.
#    If the user is greater than or equal to 36 inches tall, then print a message that the user
#    is tall enough to ride the roller coaster!
#    Otherwise, print a different user-friendly message indicating that the user is not tall enough to ride.
#    In this function, print a user-friendly message if the user does not enter a valid numeric number of inches.
#    Finally, call the function so we can see it work!




#28.  Build a void function named visitedcities that accepts a single argument for the user's name.
#     Store this argument in a parameter named name.  If the user does not provide a name, use 'tom' as the default.
#     The function should allow the user to enter as many cities as they would like until they enter the word quit.
#The following would represent a sequence for this function:
#Hello tom

#Please tell me a city you have visited:
#(Enter 'quit' when you are finished.)>? detroit
#I'd love to go to Detroit!

#Please tell me a city you have visited:
#(Enter 'quit' when you are finished.)>? nyc
#I'd love to go to Nyc!

#Please tell me a city you have visited:
#(Enter 'quit' when you are finished.)>? quit

#     Call the function with and without an argument!





#29.  Build a fruitful function that returns a user's full name (formatted).
#     The function should accept three arguments stored in three parameters.  One parameter
#     should store the first name, one for the last name and one for the middle name.
#     If the user does not pass a middle name argument, use '' as the default.
#     Once finished, call the function a couple of times with different arguments and print
#     the output of those function calls.




#30.  Build a void function named greet_users that accepts a list of names as an argument stored in a names
#     parameter.  In the function, loop over each name in the names parameter and print a hello message.
#     For instance,
#     Hello, Hannah!     #might be a good message to print for an item in the list named Hannah
#     After your function is created, call the function a few times.





#31.  Build a fruitful function named userpoll that returns a dictionary object.
#     This function should accept no arguments.
#     In this function, allow the user to enter their name, then ask them what mountain
#     they would like to climb.  Then, add the users response to a dictionary object with the key being the user's name
#     and the value being the name of the mountain they would like to climb.
#     Then, ask the user if they would like another person to take our
#     mountain climbing poll.  If they enter yes, repeat the poll.  If they enter no, end the poll and
#     return the dictionary object to the calling statement/expression.
#     Call the function and print the dictionary object that gets returned from the function.





#For the next few problems, don't use any special modules such as the csv module.
#32.  Create a new file named practicetest.txt in the current directory.
#     Write two lines of text to this file.
#     Line one should be customer,amount
#     Line two should be bill, 2000
#     Make sure the text appears on two separate lines.





#33.  Open the file you just created in the previous step and append the following two lines:
#     joe, 3500
#     bryan, 3250
#     Again, please make sure that these two lines appear on separate lines in the file.




#34. Open the file you just modified in the previous question.
#    Print each line in the file on a separate line in the output window.
#    Handle the possibility that the file does not exist.





#35. Using the csv module, open the file you have been working with in the previous steps for reading.
#    Create the following list of tuples with the data contained in that text file:
#[('customer', 'amount'), ('bill', ' 2000'), ('joe', '3500'), ('bryan', '3250')]
#    Each tuple within the list should be a row in the file.
#    Print the list after you create it and then print the item that contains 'joe'





#36. Using the csv module, open the file you have been working with in the previous steps for reading.
#    Create the following dictionary object with the data contained in that text file:
#[{'customer': 'bill', 'amount': ' 2000'}, {'customer': 'joe', 'amount': '3500'}, {'customer': 'bryan', 'amount': '3250'}]
#    Then, print the list of dictionary objects.
#    Then, print bryan by referencing the key-value pair of the appropriate dictionary object.





#37. Using the csv module, create a new file named practicetest.txt.  This program should overwrite the file
#    if it already exists (instead of throwing an error).
#    Write the following rows to the file:
#"customer"; "amount"
#"bill"; "2000"
#"joe"; "3500"
#"bryan"; "3250"
#    Notice the double quotes around each data element in the file.

