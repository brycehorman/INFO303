my_dict_items = {}
my_dict_items = {1: 'orange', 2: 'banana', 3: 'kiwi'}
my_dict_items = {'first name': 'Joe', 1: [8,12,33]}
print(type(my_dict_items))
print(dir(my_dict_items))

my_dict_items = dict({1: 'orange', 2: 'banana', 3: 'kiwi'})
my_dict_items = dict([(1, 'orange'), (2, 'banana'), (3, 'kiwki')])
print(my_dict_items)

print(my_dict_items[2])
print(my_dict_items.get(3))

house_dict = dict([('color','blue'),('size',2500)])
print(house_dict['color'])
print(house_dict['size'])
print(house_dict.get('color'))
print('My house is the color %s and is %s square feet' % (house_dict.get('color'), house_dict['size']))
house_dict['condition'] = house_dict.get('condition', 'Poor')
print(house_dict)
print('My house is the color %r and is %s square feet and the condition is %r' %(house_dict['color'],
                                                                                 house_dict['size'], house_dict['condition']))

import random
x = random.randint(1,3)
if x == 3:
    house_dict['condition']='Excellent'
elif x == 2:
    house_dict['condition']= 'Average'
else:
    house_dict['condition']= 'Poor'
print(house_dict)
print('My house is the color %r and is %s square feet and the condition is %r' %(house_dict['color'],
                                                                                 house_dict['size'], house_dict['condition']))

employee_dict = {'name': 'Bill', 'age': 55, 'division':'marketing', 'office':'Detroit'}
print(employee_dict)
employee_dict['age']=56
print(employee_dict)

employee_dict['email'] = employee_dict.get('email','Bill@company.com')
print(employee_dict)

squares ={1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64}
print(squares)
squares.pop(4)
print(squares)

squares.pop(9)
try:
    squares.pop(9)
except:
    print('That key value is not in the dictionary so we cannot pop that key!')

print(squares)
squares.popitem()
print(squares)
squares.clear()
print(squares)

squares ={1:1, 2:4, 3:9, 4:16, 5:25, 6:36, 7:49, 8:64}
print(squares)
del squares[5]
print(squares)

del squares[9]
try:
    del squares[9]
except:
    print('That key value is not in the dictionary so we cannot pop that key!')

del squares
try:
    print(squares)
except:
    print('Object has been deleted from memory')

squares = {x: x*x for x in range(9)}
print(squares)

squares = {}
for x in range(9):
    squares[x]= x*x
    squares[x] = squares.get(x, x*x)

print(squares)

odd_squares = {x: x*x for x in range(11) if x%2 == 1}
print(odd_squares)
print(len(odd_squares))

print(1 in odd_squares)
print(2 not in odd_squares)
print(81 in odd_squares)

grades = {}.fromkeys(['management', 'economics', 'marketing', 'accounting', 'finance'],0)
print(grades)

for k, v in grades.items():
    print('The key is: ', v)
    print('The associated value with this key is ', v)

for key, value in grades.items():
    print('\nKey: ' + key)
    print('Value: ', value)

import random
for key, value in grades.items():
    print('\nKey: ' + key)
    print('Value: ', value)
    x = random.randint(75,95)
    grades[key] = x
print(grades)

favs = ['Economics','Accounting']
for dept in grades.keys():
    #print(dept.upper())
    if dept in favs:
        print('I am glad to see that', dept, 'is one of your favorite subjects')
    else:
        print('I also do not like', dept + '!!!!')

if 'English' not in grades.keys():
    print('We need to notify the English department that their data are missing')

print('The following grades have been recorded in the system:')
for grade in grades.values():
    print(grade)

print(sorted(grades.keys()))
print(grades.keys())

house_dict1 = dict(['color','blue'),('size',2500)])
house_dict2 = dict(['color','green'),('size',1500)])
house_dict3 = dict(['color','yellow'),('size',1000)])
print(house_dict1)
print(house_dict2)
print(house_dict3)

homes = [house_dict1, house_dict2, house_dict3]
print(homes)
for home in homes:
    print(home)

print(homes[3])
homes[3] = ('color': 'gray', 'size': 5000)
print(homes[3])
print(homes)

for home in homes:
    if home['color'] == 'gray':
        print('This ' + home['color'] + ' house is ' str(home['size']) + ' square feet.')
    elif home['color'] == 'yellow':
        print(home['color'] + ' is not a nice color. Please repaint your house')
    else:
        print(home)

import random
homes = []
colors = ['red', 'green', 'blue','yellow', 'white']
sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500]
for house in range(30):
    new_house = {'color'}

favorite_subjects = {
    'sally': ['MGMT', 'MIS'],
    'Pete': ['MKT', 'FIN'],
    'Ally': ['ECON', 'ENG','ART'],
    'Ed': ['ACCT']
}

for name, subjects in sorted(favorite_subjects.items()):
    print('\n' + name.title() + '\'s favorite subjects in school are:')
    for subject in subjects:
        print('\t' + subject.title())

users = {
    'tmattson' : {'fname': 'tom','lname': 'mattson', 'location': 'new york'},
    'rmsith' : {'fname': 'rick','lname': 'smith', 'location': 'virginia'},
    'sjones' : {'fname': 'sally','lname': 'jones', 'location': 'maryland'},
}

for name, details in sorted(users.items()):
    print('\n UserID: ' + name)
    fullname = details['fname'] + ' ' + details['lname']
    location = details['location']
    print('\t Full name: ' + fullname.title())
    print('\t Location: ' + location.title())
    
