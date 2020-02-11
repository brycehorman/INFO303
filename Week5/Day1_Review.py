#These tasks represents tasks that you should probably be able to do without looking
#up how to do them on the Internet.

#1. Ask the user to enter their name and print a message that says 'Hello, {name}!
name = input("What is your name?")
print("hello",name.title() + "!")

#2. Modify the previous program/statement to convert the users name to title case.


#3 Create a variable and assign the value of 6 to it.
#  Multiply that value by 22 and print out the result with a user-friendly message such as
#  The value of 6 multiplied by 22 is 132.
val = 6
val2 = 6*22
print('The value of',val,'multiplied by 22 is',val2)


#4. Ask the user to enter two integer values.
#   Perform a floor division of the first number (numerator) and the second number (denominator)
#   Then, print out the result
one = int(input('Enter an integer'))
two = int(input('Enter another integer'))
result = one//two
print(result)



#5. Ask the user for two integer values.
#   Divide the first number by the second number.
#   Then, print off a message that reads something like the following:
#   19 divided by 10 equals 1 with a remainder of 9.
try:
    one = int(input('Enter an integer'))
    two = int(input('Enter another integer'))
    result = one//two
    mod = one%two
    print(one,'divided by',two,'equals',result,'with a remainder of',mod)
except:
    print('Please enter integers instead of text.')



#6. In the previous program, if the user enters a string (text) instead of integers,
#   print a message (instead of the traceback message) such as the following:
#   Please enter integers instead of text.





#7. Ask the user to enter a sentence.
#   Print the sentence with just the first word in the sentence capitalized.
#   'Hello my name is tom.'
#   Then, print the sentence with each word capitalized.
#   'Hello My Name Is Tom.'
#   Then, print the sentence with each word in lower case and again with each word in all capital letters.
sent = input('Enter a sentence.')
print(sent.capitalize())
print(sent.title())
print(sent.upper())
print(sent.lower())




#8.  Ask the user to enter a word.
#   Print a message indicating the length of the word.  For instance, enter the following message:
#   You entered 'Hello', which is 5 characters long.
word = input('Enter a word')
length = len(word)
print('You entered',word,'which is',length,'characters long.')




#9.  Set a variable equal to the following string: 'We are learning python to eventually learn machine learning.'
#    Then, print off just the word Python with a capital P.
#    Then, print off the words machine learning with a capital M and capital L.
sent = 'We are learning python to eventually learn machine learning.'
one = sent[0:16]
two = sent[16:]
print(one + two.capitalize())


#10. Create a variable with the following string:
#    I told my friend, "Python is my favorite language!" with the quotation marks
#    Then, print out that variable.
#    Next, reassign the value of your variable to:
#    The language of 'Python' is named after Monty Python, not the snake with Python in single quotations.
#    Then, print out that variable.
#    Next, reassign the value of your variable to:
#    One of Python's strengths is its diverse and supportive community with the apostrophe.
#    Then, print out that variable.
sent = 'I told my friend, \"Python is my favorite language!\"'
print(sent)
sent = 'The language of \'Python\' is named after monty python, not the snake'
print(sent)
sent = 'One of python\'s strengths is its diverse and supportive community'
print(sent)





#11. Create a variable to store my first name ('Tom') and a second variable to store my last name ('Mattson')
#    Then, create a third variable that concatenates the two names together with a space between the names.
#    Then, print the third variable
first = 'Tom'
last = 'Mattson'
full = first + ' ' + last
print(full)




#12. In the previous program, instead of creating a third variable, combine the first two variables in a print
#    statement.  Try to do this multiple ways!
first = 'Tom'
last = 'Mattson'
print(first,last)





#13. Print the following with the tabs and the returns:
#Courses:
#   MGMT325
#   INFO201
#   INFO303
print('\tMGMT325\n\tINFO201\n\tINFO303')

#14.  Using the following variable:
fav_course = '          INFO 303           '
#   First, print the variable with the leading spaces stripped
#   Second, print the variable with the trailing spaces stripped
#   Thirs, print the variable with both the leading and trailing spaces stripped
print(fav_course.lstrip(),'f')
print(fav_course.rstrip())
print(fav_course.strip())



#15. Ask the user to enter two numbers.
#    Use a try/except block to handle the possibility that the user enters text instead of numbers in their response.
#    Then, add the two numbers together and print the result.
#    Then, multiply the two numbers together and print the result.
#    Then, subtract the two numbers (first - second) and multiply that difference by 7 and print out the result.
#    Then, divide the two numbers (first number is the numerator), raise that result to the fourth power
#    and print out the result.
#    For the last calculation, print out the data type.
#    Finally, add a docstring comment at the start of your answer
try:
    """This is a try except block for performing arithmetic calculations"""
    one = int(input('Enter a number'))
    two = int(input('Enter another number'))
    sum = one + two
    print(sum,sum**4)
except:
    print('Invalid data entry')




#16. Ask the user to enter a number.
#    If the number is greater than or equal to 25, print an appropriate message.
num = int(input('Enter a number'))
if num >= 25:

    print('Greater than or equal to 25')



#17. Ask the user to enter a number.
#    If the number is between 15 (inclusive) and 40 (inclusive), print an appropriate message.
num = int(input('Enter a number'))
if num >=15 and num <= 40:
    print('Num is between 15 and 40 (inclusive)')



#18. Ask the user to enter either tom, bill, joe, or sally.
#    If they enter, bill (in any case), then print an appropriate message.
name = input('Enter either tom, bill, joe, or sally')
name = name.lower()
if name == 'bill':
    print('You guessed correctly.')



#19. Ask the user to enter either tom, bill, joe, or sally.
#    If they enter, bill (in any case) or tom (in any case), then print an appropriate message.
name = input('Enter either tom, bill, joe, or sally')
name = name.lower()
if name == 'bill' or name == 'tom':
    print('You guessed correctly.')



#20. Ask the user to enter two numbers.  Divide the first number by the second number.
#   If the result is greater than 4, print an appropriate message.  For instance:
#   16.0 divided by 3.0 is 5.33, which is greater than 4
#   Otherwise, print an appropriate message.  For instance,
#   16.0 divided by 4.0 is 4.0, which is not greater than 4
#   Make sure the result of the division in your message is formatted with a reasonable number of decimal places.
one = int(input('Enter a number'))
two = int(input('Enter another number'))
div = one/two
div = round(div,2)
if div > 4:
    print(one,'divided by',two,'is',div,'which is greater than 4')
else:
    print(one,'divided by',two,'is',div,'which is not greater than 4')





#21. Ask the user to enter two numbers.  Raise the first number to the second number.
#    Using a single if conditional, check for the following:
#    If the result is between 0 and 500 (inclusive of both numbers), print an appropriate message.
#    If the result is greater than 500, print an appropriate message.
#    For all other values (i.e., negative numbers), print an appropriate message.
#A sample message for one of the conditions would be the following:
#-7.0 raised to the power of 3.0 is -343.0, which is negative (less than zero)
one = int(input('Enter a number'))
two = int(input('Enter another number'))
power = one**two
if power >=0 and power <= 500:
    print(one,'raised to the power of',two,'is',power,'which is between 0 and 500')
elif power > 500:
    print(one,'raised to the power of',two,'is',power,'which is greater than 500')
else:
    print(one,'raised to the power of',two,'is',power,'which is negative')






#22. Ask the user to enter two words.
#    If the length of the first word is longer than the second word, print an appropriate message.
#    Otherwise, print an appropriate message.
# A sample message would be the following:
#The first word of 'doggy' is 5 characters long, which is longer than the second word of 'cat' that is 3 characters long.
one = input('Enter a word')
two = input('Enter another word')
if len(one) > len(two):
    print('The first word \'' + one + '\' is',len(one),'characters long, which is longer than the second word of \'' + two + '\', which is',len(two),'characters long' )
else:
    print('Second word is longer')

#23. Ask the user to enter two words.
#    If the two words are equal (do not change the case that the user enters), enter an appropriate message.
#    Otherwise, print an appropriate message.
# A sample message would be the following:
#The first word of 'doggy' is equal to the second word of 'doggy'.
one = input('Enter a word')
two = input('Enter another word')
if one == two:
    print('The first word',one,'is equal to the second word',two)
else:
    print('The words aren\'t equal in length')



#24. Modify the previous question to compare the two words, regardless of case.  Therefore,
#    dog should equal 'Dog' for this question.
one = input('Enter a word')
two = input('Enter another word')
if one.lower() == two.lower():
    print('The first word',one,'is equal to the second word',two)
else:
    print('The words aren\'t equal in length')


#25. Ask the user to enter a word.
#    If the word contains the letter r and the letter a, then print an appropriate message.
#    Otherwise, print an appropriate message.
one = input('Enter a word')
if one.find('r') > -1 and one.find('a') > -1:
    print('This word contains both r and a.')



#26. Check whether two strings are equal even if the order of words or
#    the characters are different.  To do this comparison, ignore the case of the words.
#    For instance, consider this string:
#    Str1 = “Hello and Welcome”
#    Str2 = “welcome and Hello”
#    These two should be the same!
#    If the two strings are the same, then print an appropriate message.
#    Otherwise, print an appropriate message.
#    For this question, use the following two variables!
#    After testing your code with these two strings, modify them so they are not equal.
a = 'Hello and welcome'
b = 'Welcome and Hello'
if sorted(a.lower()) == sorted(b.lower()):
    print('Both strings are equivalent')
else:
    print('These are different')



#For the next series of problems, use the random module.
#27  Print the help for random.random.
#    Generate a random real number number between 0 (inclusive) and 1 (not inclusive)
#    If the number is less than 0.5, print an appropriate message.
#    Otherwise, print a different message.
import random
one = random.random()
if one < 0.5:
    print(one,'is less than .5')
else:
    print(one,'is greater than .5')


#28   Print the help for random.randint.
#     Generate a random integer between -15 and 15.
#     If the number is zero, print an appropriate and user-friendly message with the percent chance
#     that the number would be exactly equal to zero.
#     If the number is negative, print an appropriate and user-friendly message with the percent chance
#     that the number would be negative.
#     If the number is positive, print and appropriate and user-friendly message with the percent chance
#     that the number would be positive.  Within this positive condition, do the following:
#           Pick another random integer between 5 and 10.
#           Then, raise the first random number to the power of the second random integer.
#           Then, print the result with an appropriate and user-friendly message.





#29  Create a list containing four names.
#    Have the computer randomly choose one of those names.
#    Then, print an appropriate and user-friendly message.




#30. Modify the previous example.  If the computer automatically selected the first or third item
#    in the list, print an appropriate and user-friendly message such as 'I was hoping for one of
#    these values was'.  Otherwise, print a different
#    user-appropriate message such as 'I was hoping for one of the other options.'



#31.  Generate a random sample (n=5) from the following list.
#    Then, print out the random sample with an appropriate and user-friendly message.!
my_list = [20,40,80,100,120,33,45,65,85, 97,109,123]



#32.  Modify the previous example, to ask the user to enter the sample size.
#    Check to make sure the sample size is less than the size of the list!
#    If less than the size of the list, generate the random sample.
#    Otherwise, print a message that indicates that your sample size is too big.
my_list = [20,40,80,100,120,33,45,65,85, 97,109,123]



#33.  Have the user enter three numbers.
#     Check whether the first number is greater than the second number
#     and whether the second number is greater than the number.
#     If so, print an appropriate user-friendly message.
#     If not, print an appropriate user-friendly message.




#34.  Ask the user to enter a number.
#     Check whether that number is not equal to zero.
#     If so, check if the number is positive and print an appropriate user-friendly message.
#     If not, print an appropriate user-friendly message.
#     If user enters a zero, then print an appropriate user-friendly message.




#35.  Ask the user to enter a year and a month.
#     Enter the year: 2020
#     Enter the month: 4
#     Then, print a message determining how many days in the month.  For instance,
#     There are 30 days in this month
#     First, check whether we are in a leap year by checking the following:
#     Year MOD 4 equals zero and year MOD 100 not equal to zero and year MOD 400 equals zero.
#           Then, within the leap year, perform a conditional test on the month that the user entered
#           to determine the number of days in the month.
#           Also account for a user entering an invalid month (i.e., a 13 or 14).
#     If we are not in a leap year, perform a conditional test on the month that the user entered
#           to determine the number of days in the month.
#           Also account for a user entering an invalid month (i.e., a 13 or 14).
#     If the user enters an invalid year, print a message indicating that the user entered an invalid year.



#36. Build a list of at least four car manufacturers.
#    Print the third element in the list.
#    Print the last item in the list.  For this task, do this in multiple ways.


#37. Using the previous list, print out the second element in the list in all capital letters.


#38. Using the previous list, modify the fourth item to a different manufacturer.
#    Then, print off the new fourth item in title case.
#    Then, print off the entire list.


#39. Using the previous list, delete the fourth item.


#40. Using the previous list, add a manufacturer to the last position and print the list after you add the item.


#41. Using the previous list, add a manufacturer to the second position and print the list after you add the item.


#42. Using the previous list, delete the last item in your list and print the list after you remove that item.



#43. Using the below list, delete the item with the value of 'tom' without referencing the indexed position.
#    When finished, print the list
names = ['bill','joe','sally','sue','tom', 'tim', 'violet']


#44. Using the below list, reverse the order of the list and then print the list.
#    Next, permanently sort the items in ascending order and then print the list.
#    Next, permanently sort the items in descending order and then print the list.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']

#45.  Using the below list, create a variable to store the length of the list.
#     Then, try to use that variable to reference the last item in the list.  Notice how you get an error!
#     Handle that error by printing a user-friendly and appropriate message.  Then, print the last
#     element/item in the list using two different methods.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']


#46. Using the below list and variable, check if the value stored in the variable is
#    contained in the list.  If so, print an appropriate message.  If not, print an appropriate message.
#    After you find that the value is not contained in the list, build another if statement to handle the
#    difference in case (i.e., capital F instead of lower case f)
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']


#47. Check if the following list is empty!  If so, print an appropriate and user-friendly message.
#    If not, print an appropriate and user-friendly message.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']


#48.  Clear the list from the previous question and perform the is empty check again.


#49.  Create a list of numbers from 1 to 1000 and print the values


#50.  Create a list of odd numbers from 500 to 600.  Then, print the values.
#     Create a list of even numbers from 500 to 600.  Then, print the values.
