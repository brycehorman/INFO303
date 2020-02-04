#Create a tuple containing the names of five countries.
#Then, display the entrie tuple
#Then, ask the user to enter one of those five countries.
#Based on the users selection, display the indxe number (i.e., position in the tuple)
#of that item in the tuple
countries = ('latvia','czech Republic','canada','mexico','usa')
country = input('Enter one of the following: Latvia, Czech Republic, Canada, Mexico, or USA')
country = country.lower()
index = countries.index(country)
print(index)
choice = int(input('Enter an integer between 0 and 4'))
print(countries[choice].title())




#modify the previous program to ask the user to enter an integer value between 0 and 4 after
#you have printed the first message.  Then, display the country associated with that index number!

#added above^



#Create a list of sports.  Ask the user what their favorite sport is
# and append this sport to the end of the list.
#Sort the list and display it.
sports = ['baseball','football','soccer','basketball']
fave = input('What is your favorite sport?')
sports.append(fave)
sports.sort()
print(sports)



#Create a list of subjects that you take in school.  Display those subjects to the user.
#Ask the user which of those subjects that they don't like.
#Then, delete that subject and display the list again.
subs = ['finance','math','science','english']
hate = input('Which of these subjects dont you like: finance, math, science, english')
hate = hate.lower()
subs.remove(hate)
print(subs)




#create a dictionary of foods. Have the keys be integers and the values be the name of the food.
#Then, print out the key & value pairs in the dictionary.
#Then, ask the user which item they want to delete.
#Then, delete it and display the dictionary again (but just the values/not the keys).
# This time sort the items in the dictionary.
foods = {1:'Pizza', 2:'Pasta', 3:'Chicken', 4:'Apple'}
print(foods)
hate = int(input('Which of the above do you want to delete? (enter the key)'))
del foods[hate]
print(sorted(foods.values()))




#Create an array which will store a list of integers.
# Generate five random integers and store those random integers in the array
#Display the array, showing each item on a separate line.
import random
ints = []
for i in range(5):
    ints.append(random.randint(0, 100))

for x in ints:
    print(x)




#Set a total variable to 0 to start with
#while the total is 50 or less, ask the user to input a number.
#Add that number to the total variable and print a message such as
#'The total is ...{total}. Exit the loop when the total is greater
#than 50
total = 0
while total <= 50:
    print('The total is:' , total)
    num = int(input('Enter a number'))
    total = total + num

#Ask the user to enter a number.
#Then, ask the user to enter another number.
#Add those two numbers together and then ask if they want to add another number.
#if they enter 'y', ask them to enter another number.
#Keep adding these numbers until they do not answer y.
#Once the loop is stopped, display the total
num = int(input('Enter a number'))
num2 = int(input('Enter another number'))
ans = num + num2
val = input('Do you want to add another number (y or n)')
while val == 'y':
    num3 = int(input('Enter another number: '))
    ans = ans + num3
    val = input('Do you want to add another number (y or n)')
print(ans)





#Create a variable compnum and set the value to 50.  Ask the user to enter a number.
#While the user's guess is not correct, tell them whether their guess is too high or too low.
#Then, ask them to guess again.  Once they enter the same number as the value stored in the compnum variable,
#display the message, 'Well done, you took {count} attempts to guess the correct number.'
compnum = 50
num = int(input('Enter a number'))
count = 1
while num != compnum:
    if num > compnum:
        print('Too high')
    else:
        print('Too low')
    count = count + 1
    num = int(input('Guess again.'))
print('Well done, you took', count , 'attempts to guess the correct number')



#loop over numbers 7 to 19 and print out that number
for x in range(7,20):
    print(x)



#Loop over the odd numbers between 1 and 10 and print out that number
for x in range(1,11,2):
    print(x)


#Starting with number 10, print out 10, 7 and 4 (negative step using the range function)
for x in range(10,3,-3):
    print(x)



#Ask the user to enter their name and then display their name three times
name = input('enter your name: ')
for x in range(3):
    print(name)


#Modify the previous program to ask the user how many times they want you to
#display their name. Please account for the possibility that the user will enter
#something other than an integer in the 'how many times' input box using a try/except block
try:
    name = input('enter your name: ')
    times = int(input('how many times do you want it printed'))
    for x in range(times):
        print(name)
except:
    print('invalid data entry')


#Ask the user to enter their name. Then, ask the user to enter an integer number
#Display their name (one letter at a time on each line) and repeat this for the number of
#times they entered in the second user input.
# Also, use a try/except block to handle bad/invalid user inputs
try:
    name = input('Enter your name: ')
    num = int(input('Enter an integer number: '))
    for x in range(num):
        for y in range(len(name)):
         print(name[y])
except:
    print('invalid data entry')


#Modify the previous program.  If the number is less than 10, then display their name that number of times.
#Otherwise, display the message, 'Too high of a number!'
try:
    name = input('Enter your name: ')
    num = int(input('Enter an integer number: '))
    if num < 10:
        for x in range(num):
            for y in range(len(name)):
             print(name[y])
    else:
        print('Too high of a number!')
except:
    print('invalid data entry')


#Ask the user to enter a number between 1 and 15 and then display the multiplication table for that number
#Nest your proram in a try/except block
try:
    num = int(input('Enter a number between 1 and 15: '))
    for x in range(num+1):
        mult = x * num
        print(str(x) + ' times ' + str(num) + ' = ' + str(mult))
except:
    print('invalid data entry')



#Ask for a number below 50 and then count down from 50 to that number.
#Display the number that the user entered in the last iteration
num = int(input('Enter a number below 50'))
for x in range(50, num-1, -1):
    print(x)


#Set a variable called total to 0.  Then, ask the user to enter five numbers.
# After each user input, ask them if they want that number included in the total.
# If they do, then add that number to the total.  If they do not want it included,
#then don't add it to the total.  After they have entered all five numbers, display the total
total = 0
for x in range(6):
    num = int(input('Enter a number:'))
    val = input('Do you want that to be included in the total (y or n): ')
    if val == 'y':
        total = total + num
print(total)


#Ask the user if they want to count up or down?
#If they select up, then ask them for the top number and count from 1 to that number
#if they select down, then ask them to enter a number below 20 and then count down from 20
#to that number.  If they entered something other than up or down, display a message
#'I don't understand!'
ans = input('Do you want to count up or down')
ans = ans.lower()
if ans == 'up':
    top = int(input('Enter the top number: '))
    for x in range(1, top+1):
        print(x)
elif ans == 'down':
    low = int(input('Enter a number below 20: '))
    for x in range(20, low-1, -1):
        print(x)
else:
    print('I do not understand')


#In the following list, print out the second indexed list, then print out the first element in that list
#The first print statement should display [3,8,5] and the next statement should display the 3.
my_list = [[2,9,7],[4,7,9],[3,8,5],[7,97,65]]
print(my_list[2])
print(my_list[2][0])



#Using the following dictionary, which contains multiple dictionaries,
#have the user select a sales person and then loop over the regions for that selected sales person
#and display the region and the associated sales volume (only for the selected sales person).
#The message should read something like 'Tom sold 345 units in the S region.'  The message
#should be customized for the sales representative and each region!
sales = {'Jon': {'N':434, 'S':365, 'E':467, 'W':987},
'Tom': {'N':765, 'S':345, 'E':967, 'W':417},
'Sally': {'N':800, 'S':405, 'E':707, 'W':712},
'Sue': {'N':555, 'S':333, 'E':963, 'W':129},
         }
person = input('Select a sales person (Jon, Tom, Sally, or Sue)')
for val in sales[person].items():
    print(person + ' sold ' + str(val[1]) + ' units in the ' + str(val[0]) + ' region.')

