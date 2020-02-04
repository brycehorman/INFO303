friend1 = 'Joseph'
friend2 = 'Glen'
friend3 = 'Sally'
friends = ['Joseph', 'Glen', 'Sally']
carryon = ['Socks' , 'Shirts', 'Perfume', 54, 100, 75.6, ['brother', 'mother']]

print(friends)
print(carryon)

print(type(friends))
print(dir(friends))

print(friends[2])
print([1,24,77])
print(['red', 'blue', 'yellow'])

print([])
stuff = []
print(stuff)
stuff = ['house', 23, 25.8, 'clothes', [4,7],[4,'bill']]
print(stuff)

for i in [10,9,8,7,6,5,4,3,2,1] :
    print(i, 'seconds to launch')
print('Launch Time')

stuff = ['house', 23, 25.8, 'clothes', [4,7],[4,'bill']]
for thing in stuff :
    print(thing)
print('Finished loop')

friends = ['Joseph', 'Glen', 'Sally']
for friend in friends :
    print('Happy New Year: ', friend.lower())
    print('I am happy that {} are my friend'.format(friend.title()))
    print('\n')
    print('Just skipped a space')
print('Done with loop')

fruit = 'Banana'
print(fruit[0:1].lower())
print(fruit)
fruit = fruit.replace('B','b')
print(fruit)

numbers = [2,14,26,41,63]
print(numbers[:])
numbers[2]=28
print(numbers)

print(len(numbers))

characters = ['hello world', 'thomas', 'sally']
print(len(characters[0]))
print(len(characters))

print(list(range(10)))
print(range(10))
print(type(range(10)))
numbers = list(range(10,20))
for num in numbers:
    print(num)
print(len(numbers))

even = list(range(2,11,2))
print(even)
squares = []
for num in even:
    square=num**2
    squares.append(square)
print(even)
print(squares)

squares = [num**2 for num in list(range(2,11,2))]
squares = [num**2 for num in even]
print(squares)

print(min(squares))
print(max(squares))
print(sum(squares))
print(sum(squares)/len(squares))

squares.insert(0,1)
print(squares)
squares.insert(3,12)
print(squares)
squares.append(1000)
print(squares)

squares.pop(0)
print(squares)
squares.pop(3)
print(squares)

squares.remove(1000)
print(squares)

squares.clear()
print(squares)

a = [1,2,3]
b = [4,5,6]
c = a+b
print(a)
print(b)
print(c)
d=c[:]
print(d)
d.append(47)
c.append(49)
print(c)
print(d)

colors1 = ['red', 'yellow', 'green', 'blue', 'orange']
colors2 = colors1.copy()
colors2.append('dark blue')
colors1.append('light blue')
print('First list of colors: {}'.format(colors1))
print(f'Second list of colors: {colors2}')

a = [1,2,3]
b = [4,5,6]
a.extend(b)
print(a)

a = [1,2,3]
b = [4,5,6]
c = a+b
print(a)
print(b)
print(c)
d =c
c.append(51)
print(c)
print(d)

numbers = [1,1,1,4,7,5,9,9,10,11,1]
print(numbers.count(9))

colors = ['red', 'yellow', 'green', 'blue', 'orange']
print(colors.index('blue'))

employees = ['Joe', 'sally', 'charles', 'mike', 'josh', 'sue']
print(employees[0:3])
print(employees[2:4])
print(employees[:4])
print(employees[2:])

x = 3
print('Here are the first {} employees in my list'.format(x))
for emp in employees[:x]:
    print(emp.upper())

x = 2
print('Here are the first {} employees in my list'.format(x))
for emp in employees[len(employees)-x:]:
    print(emp.upper())

employees.sort()
print(employees)
print(employees[0])
print(employees[1])

numbers = [4,5,3,2,1]
numbers.sort()
print(numbers)
numbers.sort(reverse = True)
print(numbers)

numbers = [4,5,3,2,1]
numbers.reverse()
print(numbers)

x=list(range(30,40))
print(x)
print(25 in x)
print(25 not in x)
print(35 in x)
print(40 not in x)

#split function
line = 'This class is       wonderful'
etc = line.split()
print(etc)
line = 'custid;name;address;email'
etc = line.split()
print(etc)
print(len(etc))

etc = line.split(';')
print(etc)
print(len(etc))
etc.append('phone')
print(etc)




