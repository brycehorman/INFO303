#pip is a utility
#pip is the standard package-management system used to install
# and manage software packages written in the Python environment.
# Many packages are found in the default source for packages
# and their dependencies — Python Package Index (PyPI) (www.pypi.org)

#Most distributions of Python come with pip preinstalled.
#pip is a recursive acronym for "Pip Installs Packages"
#python -m pip --version or python -m pip -V
#python -m pip install requests
#NOTE: Depending on how you have your environment variables configured, you can
#skip the python -m (which looks/searches in the python path for the pip utility).
#However, make sure that you are installing the package to the correct interpreter
#if you have multiple versions of python installed
#pip install requests
#NOTE: You may not have full permissions on the python directory
#python -m pip install --user requests

import requests
url = 'https://facultystaff.richmond.edu/~tmattson'
r = requests.get(url)
print(r.status_code == requests.codes.ok)
print(r.text)
print(r.headers)

#if you want to unistall the requests package
#python -m pip uninstall requests



#Let's review loading data from a text file into a series of data structures in python
import csv
#NOTE: This path might be different depending on where you put the file
file_name = 'Students.txt'
handle = open(file_name,'r')
my_list = []
data = csv.reader(handle, delimiter=';')
for row in data:
    my_list.append(tuple(row))
handle.close()
# List with each row as a tuple
print(my_list)

import csv
handle = open(file_name, 'r')
file_name = 'InClassExample/Students.txt'
#This should put everything EXCEPT the header into a dictionary object
reader = csv.DictReader(handle, delimiter=';')
my_list = []
for raw in reader:
    my_list.append(dict(raw))
handle.close()
#list of dictionary objects with the header row serving as the key for each key,value pair
print(my_list)
print(my_list[0]['Student_Id'])
print(my_list[0]['First_Name'])

#Sometimes data are contained in different structured formats besides csv (or otherwise delimited files)
#Python allows us to relatively easily work with data contained in other structured formats
#Let's now work with some json data. json stands for javascript object notation.
import json
file_name = 'JSON_Sample.json'
handle = open(file_name, 'r')
json_loaded = json.load(handle)
handle.close()
print(json_loaded)
#Reference an element in a similar manner to referencing a dictionary object
print(json_loaded['employees'])
print(json_loaded['employees'][1]['email'])

#Now we reference the employees like we would reference a dictionary object
#We have a list of dictionary objects...loop over each item in the list, which is a dictionary!
#Then loop over each key/value pair in the dictionary
for dict in json_loaded['employees']:
    for key, value in dict.items():
        print(f'Key: {key.title()}\nValue: {value}')

#Notice what happens when we try to load a poorly structured JSON file!

#We can also use the json object library for data that are contained in a dictionary
#using the dumps function!
student = {"name": "Jessie", "semester" : "Fall 2019", "courses":
                             ["MGMT 325", "INFO 301", "MGMT 375", "FIN 360", "MKT 325"]}

student_dumped = json.dumps(student)
print(student_dumped)
#However this is a string object so our base dictionary object is certainly more useful
print(type(student_dumped))
#If you have a string, you can write it to a file
handle = open (file_name, 'w')
handle.write(student_dumped)
handle.close()

#However, we don't have to convert it to a string to create a json file!
#Instead, we can use a similar function (dump) to create a json file from a dictionary object
file_name = 'InClassExample/testing.json'
handle = open (file_name, 'w')
json.dump(student, handle)
#Let's make the file a little more readable
#json.dump(student, handle, indent=4, sort_keys=True)
handle.close()

#Another type of structured data file is an xml file
#Note how we have to do the import here. We can't just import xml based on the way this package is structured
import xml.etree.ElementTree as ET
#XML files are still used frequently.  We see them with RSS (Really Simple Syndication) feeds.
#https://www.techmeme.com/feed.xml
#https://www.espn.com/espn/rss/
#We can make a request.url to the rss feed
#Then, save the file as an xml file to our operating system
#Make sure the file is a well-formed XML file and then parse it.
tree = ET.parse('toptechnewsfeed.xml')
# get root element at the top of the tree, which is the <channel> tag.
root = tree.getroot()
newsitems = []
# iterate news items
for item in root.findall('./channel/item'):
    news = {}
    # iterate child elements of each item, which should be the <title>, <link>,
    # <description>, <pubDate>, and <guid> XML tags
    #Each item should contain an iterable tree of items
    for child in item:
        #In Python 3, byte literals are always prefixed with 'b' or 'B';
        # they produce an instance of the bytes type instead of the str type.
        #This will insert an instance of the bytes class as the value in our dictionary object
        if child.tag != 'description':
            news[child.tag] = child.text.encode('utf-8')
            #The following will insert a standard string key, value pair to our dictionary
            #news[child.tag] = child.text
    newsitems.append(news)
print(newsitems)
print(newsitems[1]['title'])
print(newsitems[1]['title'].decode('utf-8'))
print(len(newsitems))


#You can also traverse the tree without using the findall function!
#This 'should' yield the same result
tree = ET.parse('InClassExample/toptechnewsfeed.xml')
# get root element at the top of the tree, which is the <channel> tag.
root = tree.getroot()
newsitems = []
for elem in root:
    for subelem in elem:
        if subelem.tag == 'item':
            news = {}
            for items in subelem:
                if items.tag != 'description':
                    news[items.tag] = items.text.encode('utf-8')
                    print(items.tag)
                    print(items.text.encode('utf-8'))
            print(news)
            # The following will insert a standard string key, value pair to our dictionary
            # news[child.tag] = child.text
            newsitems.append(news)
print(newsitems)
print(newsitems[1]['title'])
print(newsitems[1]['title'].decode('utf-8'))
print(len(newsitems))

#Write this output to a csv file
import csv
fields = ['title', 'link', 'pubDate', 'guid']
handle = open('testingoutput.txt', 'w', newline='')
writer = csv.DictWriter(handle, fieldnames=fields, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# writing headers (field names)
writer.writeheader()

# writing data rows from our list object
writer.writerows(newsitems)
handle.close()

#We can also construct an xml file by constructing the same tree structure
#<school>
#    <instructors>
#        <instructor name="tmattson">Tom Mattson</instructor>
#        <instructor name="thims">Tim Himson</instructor>
#    </instructors>
#</school>

#As you can see, it's a fairly simple XML example, only containing a few nested
# objects and one attribute. However, it should be enough to demonstrate all of the
# XML operations in this article.

import xml.etree.ElementTree as ET

# create the file structure
school = ET.Element('school')
items = ET.SubElement(school, 'instructors')
#Obviously we would probably have the data that we are outputting to this xml file in a
#list or dictionary so we would loop...but this is how we construct the structure
item1 = ET.SubElement(items, 'instructor')
item2 = ET.SubElement(items, 'instructor')
item1.set('name','tmattson')
item2.set('name','thims')
item1.text = 'Tom Mattson'
item2.text = 'Tim Himson'

# create a new XML file with the results
#Note how we reference the root object here!
mydata = ET.tostring(school)
myfile = open('instructors.xml', 'w')
myfile.write(mydata.decode('utf-8'))
myfile.close()


#Now let's use a real world example where we have data from three sources in three different forms
# (one xml, one json, and one csv file).  Our goal is to do a few calculations and summary statistics using
# bits of information from all three files.
#We will first load the data from all three files and then we will calculate a few statistics
#courses.txt file
#students.xml file
#students.json file.
#With these files, we will then calculate a few different statistics!

#Import the courses csv file
#Built two functions...One will load the data to a list of tuples and the other will load
#the data to a list of dictionaries.
def coursesparsetolist(file_name):
    """ Parsing to a list. Pass in the file_name as a parameter"""
    import csv
    handle = open(file_name,'r')
    my_list = []
    data = csv.reader(handle, delimiter='|')
    for row in data:
        my_list.append(tuple(row))
    handle.close()
    return my_list

def coursesparsetodict(file_name):
    """ Parsing to a dictionary. Pass in the file_name as a parameter"""
    import csv
    handle = open(file_name, 'r') #r should be the default
    #This should put everything EXCEPT the header into a dictionary object
    reader = csv.DictReader(handle, delimiter='|')
    my_list = []
    for raw in reader:
        my_list.append(raw)
    handle.close()
    return my_list

#Which structure is best will depend on how we will have to loop over and/or reference the data contained in the list
listofcoursetuples = coursesparsetolist('InclassExample/Courses.txt')
print(listofcoursetuples)
listofcoursedicts = coursesparsetodict('InclassExample/Courses.txt')
print(listofcoursedicts)

#Now let's load the xml data...here, we either want to load the data into the list object for either the international
#or the domestic students
def loadxmlet (file_name, inter = 'no'):
    import xml.etree.ElementTree as ET
    #Create the tree with all of the associated branches
    tree = ET.parse(file_name)
    # get root element at the top of the tree, which is the <channel> tag.
    root = tree.getroot()
    my_list = [] #Create a list of the international or the non-international students
    for elem in root:
        id = elem.attrib['id']
        for subelem in elem:
            if subelem.tag == 'international':
               if inter == subelem.text:
                   my_list.append(int(id))
                  #my_dict = {'id': id, subelem.tag: subelem.text}
            #If we want/need to do something with the awards
            #if subelem.tag == 'awards':
            #    for subaward in subelem:
            #        print(subaward.tag)
            #        print(subaward.text)
    return my_list

listofstudents = loadxmlet('InclassExample/StudentDetails.xml', 'yes') #Second argument determines whether we want the domestic or the international students.
print (listofstudents)

#open a JSON file
import json, statistics
file_name = 'InclassExample/Students.json'
handle = open(file_name, 'r')
json_loaded = json.load(handle)

#calculate the average for a specific course
cname = input('Please enter a course name such as FIN 462 or MGMT 320:')
my_list = []
for dict in json_loaded['students']:
    if len(dict['classes']) != 0:
        for x in dict['classes']:
            for k, v in x.items():
                if v == cname:
                    my_list.append(x['Grade'])
print(f'The average for {cname} across all students is: {"{0:,.2F}".format(statistics.mean(my_list))}')

#Now calculate the average of the grades for international students in a user entered course name
listofstudents = loadxmlet('InclassExample/StudentDetails.xml', 'yes')
cname = input('Please enter a course name such as FIN 462 or MGMT 320:')
my_list = []
for dict in json_loaded['students']:
    id = dict['studentid']
    if id in listofstudents: #Check to see if the studentid is in the list of international students
        if len(dict['classes']) != 0:
            for x in dict['classes']:
                for k, v in x.items():
                    if v == cname:
                        my_list.append(x['Grade'])
print(f'The average for {cname} across all international students is: {"{0:,.2F}".format(statistics.mean(my_list))}')

#Now for doemstic students
listofstudents = loadxmlet('InclassExample/StudentDetails.xml', 'no')
cname = input('Please enter a course name such as FIN 462 or MGMT 320:')
my_list = []
for dict in json_loaded['students']:
    id = dict['studentid']
    if id in listofstudents: #Check to see if the studentid is in the list of international students
        if len(dict['classes']) != 0:
            for x in dict['classes']:
                for k, v in x.items():
                    if v == cname:
                        my_list.append(x['Grade'])
print(f'The average for {cname} across all non international students is: {"{0:,.2F}".format(statistics.mean(my_list))}')

#Now let's only calculate the average across courses with a rating of 3 or higher
#Let's first build a function to determine the rating of a course
def crating(coursetuples, cname):
    for x in coursetuples:
        if x[0] == cname:
            return int(x[1])

internat = 'yes'
listofstudents = loadxmlet('InclassExample/StudentDetails.xml', internat)
ratingscores = 2  #we want all courses rated at or above this number
my_list = []
for dict in json_loaded['students']:
    id = dict['studentid']
    if id in listofstudents: #Check to see if the studentid is in the list of relevant students
        if len(dict['classes']) != 0: #Has the student taken any classes?
            for x in dict['classes']:  #Loop over the classes
                for k, v in x.items(): #Each class is a dictionary key == Title of the course & value == name of the course
                    if k == 'Title':
                        rating = crating(listofcoursetuples, v)  #v should have the course name
                        if rating >= ratingscores:
                            my_list.append(x['Grade'])
print(f'The average for grades for all courses rated {ratingscores} or higher'
      f' across all students flagged as international equal to {internat} '
      f'is: {"{0:,.2F}".format(statistics.mean(my_list))}')