#Dealing with dates can be frustrating because
#Dates come in all different formats, which can make it difficult to add
# or subtract dates or to compare values against specific dates.

#In a banking context,
#selecting all the accounts that have past due loan payments involve manipulating dates
#
#In a HR context, find all employees hired before or after a specific date
#to help calculate benefit eligibility involve manipulating dates.
#
#In a marketing context, find all customers who made multiple purchases in a given month
# involve manipulating dates
#
#In an accounting context, identify all journal entries that were entered in the last week of a quarter
#and the first week of a quarter.

#Once we have a date in a datetime object, we can manipulate the dates quite easily but
#converting them to a datetime object can be challenging due to the different date formats.

#Let's first work with the datetime module/library
import datetime

#today fetches the current date
print(datetime.date.today())

#Create a date object with a specific year, month, day
dt = datetime.date(2020, 10, 20)
print(type(dt))
print(dt)

#now we can extract the components of the date object
print(dt.day)
print(dt.month)
print(dt.year)

#We can also convert dates to strings using the strftime function
#This function takes the format as an argument (see below for formatting flags).
dt = datetime.date.today()
print(dt)
sdt = dt.strftime('%m-%d-%Y')
print(type(sdt))
print('The current date is ' + sdt)

sdt = dt.strftime('%a, %d-%m-%y')
print(sdt)
sdt = dt.strftime('%A, %B %d, %Y')
print(sdt)

#Common datetime formatting options are as follows:
#%d refers to day of the month. In 20-10-2019, %d returns 20.
#%m refers to month of the year. In 20-10-2019, %m returns 10.
#%Y refers to year. The letter 'Y' is in upper case. In 20-10-2019, %Y returns 2019.
#%y refers to year in two-digit format. In 20-10-2019, %y returns 19.
#%a returns the first three letter of the weekday Sun
#%A returns the complete name of the weekday Sunday
#%b returns the first three letters of the month Oct
#%B returns the complete name of the month October
#Some time flags are as follows:
#%I converts 24 hour time format to 12 hour format.
#%p returns AM PM based on time values.
#%H returns hours of the time value.
#%M returns minutes of the time value.
#%S returns seconds of the time value.

#We can also work with time objects using the datetime class
t = datetime.time(21, 2, 3)
print(t)
print(type(t))
print(t.hour)
print(t.minute)
print(t.second)
#Add in microseconds to our datetime.time object
t = datetime.time(21, 2, 3, 4352)
print(t.microsecond)

#Now that we have a time object, we can format our time however we want/need
#using the strftime function
print(t.strftime('%I:%M %p'))
print(t.strftime('%I:%M:%S %p'))
print(t.strftime('%I:%M:%S.%f %p'))

#Now let's work with both dates and times together by extracting the current date & time using the
#now() function and just a datetime function
dt = datetime.datetime(2020, 3, 20, 11, 33, 14)
print(type(dt))
print(dt)
dt = datetime.datetime.now()
print(type(dt))
print(dt)
#Depending on where you live in the world, the format of the above datetime might be in a different
#format.

#The International Organization for Standardization (ISO) date
# format is a standard way to express a numeric calendar date that eliminates ambiguity.
#Note that the isoformat() function results in a string and not a datetime object.
dt = datetime.datetime.now()
my_date = dt.isoformat()  #is this a date or a string object?
print(type(my_date))
print(my_date)

#my_date = 'Jun 28 2018  7:40AM'

#The first useful function here is the strptime function
parsed_date = datetime.datetime.strptime(my_date, '%Y-%m-%dT%H:%M:%S')
parsed_date = datetime.datetime.strptime(my_date, '%Y-%m-%dT%H:%M:%S.%f')
parsed_date = datetime.datetime.strptime(my_date, '%b %d %Y %I:%M%p')
print(type(parsed_date))
print(parsed_date)

#Now we can print out the date in whatever format we want using the strftime function
print(parsed_date.strftime('%A %B %d %X'))
print(parsed_date.strftime('%A %B %d %H:%M'))
print(parsed_date.strftime('%A %B %d %H:%M:%S'))

#Now we can extract the dates and times from our datetime objects
print('Date:', parsed_date.date(), 'Time:', parsed_date.time())

#The dateutils module allows us another way to parse strings to date objects
#python -m pip install dateutils
#python -m pip install python-dateutil
from dateutil.parser import *

date_time_str1 = 'Jun 28 2018  7:40AM'
date_time_str2 = 'September 23, 2019, 22:19:55'
date_time_str3 = 'Sun,06/14/19 12:30PM'
date_time_str4 = '2020-03-12T10:12:45Z'

#NOTE how the parse function takes a string and converts it to a date object
tmstr = parse(date_time_str1)
print(tmstr)
tmstr = parse(date_time_str2)
print(tmstr)
tmstr = parse(date_time_str3)
print(tmstr)
#It is not perfect...some strings are not recognized
#Notice the error here!
date_time_str3 = 'Sun,05/12/99,12:30PM'
tmstr = parse(date_time_str3)
#If we have a non-standard format that the dateutil.parser can't handle, we will probably just revert to
#using the strptime function where we feed in the specific format or possibly a find and replace...
tmstr = parse(date_time_str3.replace(',', ' '))
print(tmstr)

tmstr = parse(date_time_str4)
print(tmstr)

sdt = 'Today is 29 of October of 2020, exactly ' \
      'at 11:20:11 with timezone -04:00.'
tmstr = parse(sdt, fuzzy=True)
print(tmstr)

#We have to be careful with ambiguous dates
#Where is the year in this string
tmstr = parse('12-08-11', yearfirst=True)
print(tmstr)

tmstr = parse('2020.08.11 AD at 15:08:56 PDT', ignoretz=True)
print(tmstr)

#Now let's do some addition and subtraction of dates
#First, let's use the datetime module

#The datetime library gives us a basic way of adding and subtracting dates
#30 days ahead
import datetime
dt = datetime.datetime.now()
print(type(dt))
delta = datetime.timedelta(days=30)
print(type(delta))
print(dt)
print(delta)
print(dt + delta)

#30 days back
print(dt - delta)

#Ahead by 10 days and fraction thereof
delta = datetime.timedelta(days=10, hours=3, minutes=30, seconds=30)
print(dt + delta)

#Advance by weeks and fractions thereof
delta = datetime.timedelta(weeks= 4, hours=3, minutes=30, seconds=30)
print(dt + delta)

#However, a more efficient way to do these types of date operations is by using the
#python-dateutil module.

#importing the dateutil module in this manner will not work based on how
#this module is coded.

import dateutil
now = datetime.datetime.now()
#What is now plus one month?
ndt = now+dateutil.relativedelta(months=+1)
print(ndt)

from dateutil.relativedelta import *
#The relativedelta library provides us with an easy way to add different date units
# (days, weeks, months, years & so on)
#grab the date and time and place them in two variables

#Let's revert to my current system date/time
now = datetime.datetime.now()
print(now)

#What is now plus one month
ndt = now+relativedelta(months=+1)
print(ndt)

#What is now minus one month
ndt = now+relativedelta(months=-1)
print(ndt)

#What is now plus one month and one week?
ndt = now+relativedelta(months=+1, weeks=+1)
print(ndt)

#Plural adds or subtracts while singular replaces
print(now)
ndt = now+relativedelta(year=1, month=4)
print(ndt)

#notice how the year changes from 2020 to 0001 because of the replacing operation
ndt = now+relativedelta(year=2018, month=4, day=14)
print(ndt)

#One month before one year....This example should add 11 months to the now datetime object
ndt = now + relativedelta(years=+1, months=-1)
print(ndt)

#How does relativedelta handle months with different number of days?
# Notice that adding one month will never cross the month boundary.
#February is a great example because it has a different number of days!
dt1 = datetime.date(2020,2,27)+relativedelta(months=+1)
print(dt1)
dt1 = datetime.date(2020,1,31)+relativedelta(months=+1)
print(dt1)
dt1 = datetime.date(2020,3,31)+relativedelta(months=+1)
print(dt1)

#The logic for adding years is the same, even on leap years.
dt1 = datetime.date(2020,2,28)+relativedelta(years=+1)
print(dt1)
dt1 = datetime.date(2020,2,29)+relativedelta(years=+1)
print(dt1)
#Subtract two years from 2/29/2020
#It will be smart enough to recognize that 2/29/2018 is not actual date
dt1 = datetime.date(2020,2,29)+relativedelta(years=-2)
print(dt1)
#Subtract one year from 3/1/2021 and from 2/28/2021
dt1 = datetime.date(2021,3,1)+relativedelta(years=-1)
print(dt1)
dt1 = datetime.date(2021,2,28)+relativedelta(years=-1)
print(dt1)

#Lets use the date only without the time for the next few examples
today = datetime.date.today()
print(today)

#What is the date for next friday?
ndt = today+relativedelta(weekday=FR)
print(ndt)

#What is the last Wednesday in this month?
#substitute the current day with 31 and find the previous thursday
ndt = today+relativedelta(day=31, weekday=TH(-1))
print(ndt)

#What is the second to last Wednesday in the month
ndt = today+relativedelta(day=31, weekday=WE(-2))
print(ndt)

#What is date for next wednesday
ndt = today+relativedelta(weekday=WE(+1))
print(ndt)

#What is the date for not this coming Wednesday but the Wednesday after that?
ndt = today+relativedelta(weekday=WE(+2))
print(ndt)

#What is the date for next Tuesday so long as it is not today!!!
ndt = today+relativedelta(days=+1, weekday=TU(+1))
print(ndt)

#Find the first day of the 5th week of 2020
#This will give me the sunday of the sixth week
ndt =  datetime.datetime(2020,1,1)+relativedelta(weekday=SU, weeks=+5)
print(ndt)
#If we want the start of the 5th week, subtract one from weekday
ndt =  datetime.datetime(2020,1,1)+relativedelta(weekday=SU(-1), weeks=+5)
print(ndt)

#Now let's work with rrule, which are recurrence rules
#I will also work with the parser to parse portions of dates in the following examples
from dateutil.rrule import *
from dateutil.parser import *
#We can use rrule to create lists with different date units
#Create a list object of datetime objects starting with 1/1/2020 until 5/1/2020
#The until should be non-inclusive
tm_list = list(rrule(DAILY, dtstart = parse('20200101T090000'), until = parse('20200501T000000')))
print(tm_list)
print(tm_list[1])
print(len(tm_list))

#Now, create a list object of weeks between 1/1/2020 until 5/1/2020
tm_list = list(rrule(WEEKLY, dtstart = parse('20200101T090000'),until = parse('20200501T000000')))
print(tm_list)
print(len(tm_list))

#Change the unit from weekly to monthly!
tm_list = list(rrule(MONTHLY, dtstart = parse('20200101T090000'),until = parse('20200501T000000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing the next 10 days from 2/12/2020
tm_list = list(rrule(DAILY, count=10, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing the next 3 months from 2/12/2020
tm_list = list(rrule(MONTHLY, count=3, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing every other day starting on 2/12/2020
# but only create five instances
tm_list = list(rrule(DAILY, interval=2, count=5, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing every third month starting on 1/1/2020 (with four instances)
tm_list = list(rrule(MONTHLY, interval=3, count=4, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list containing the days in months 1,3,5,and 7 in years 2020-2021
tm_list = list(rrule(DAILY, bymonth=(1,3,5,7), dtstart=parse('20200101T090000'),
                     until=parse('20211231T090000')))
print(tm_list)
print(tm_list[0])
print(len(tm_list))

#Create a list of every other week starting on 1/1/2020 (first ten such instances)
tm_list = list(rrule(WEEKLY, count=10, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of every other week, 6 occurrences, starting on 1/1/2020.
tm_list = list(rrule(WEEKLY, interval=2, count=6, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates on Tuesday and Thursday for the first 5 weeks of 2020.
tm_list = list(rrule(WEEKLY, count=10, wkst=SU, byweekday=(TU,TH), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates on Wednesday and Thursday for every other week for the first 8 weeks in 2020.
tm_list = list(rrule(WEEKLY, interval=2, count=8, wkst=SU, byweekday=(WE,FR), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates for the 1st Friday of each month in 2020 for 3 occurrences.
tm_list = list(rrule(MONTHLY, count=3, byweekday=FR(1), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Every other month on the 1st and last Sunday of the month for 10 occurrences.
tm_list = list(rrule(MONTHLY, interval=2, count=10,
             byweekday=(SU(1), SU(-1)),
             dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#rruleset is beneath rrule so no new import is needed.
#These allow us to specify specific dates to exclude from our set.

#Create a list using a rruleset
#for 14 days (each day) starting on 1/1/2020,
# but skip Saturday and Sunday occurrences.
set = rruleset()
set.rrule(rrule(DAILY, count=14, dtstart=parse('20200101T090000')))
set.exrule(rrule(YEARLY, byweekday=(SA,SU), dtstart=parse('20200101T090000')))
tm_list = list(set)
print(tm_list)

#Create a list using a rruleset for 4 weeks starting on 1/1/2020.
# However, make sure 1/13/2020 is included and exclude 1/15/2020.
set = rruleset()
set.rrule(rrule(WEEKLY, count=4, dtstart=parse('20200101T090000')))
set.rdate(datetime.datetime(2020, 1, 13, 9, 0))
set.exdate(datetime.datetime(2020, 1, 15, 9, 0))
tm_list = list(set)
print(tm_list)


#Another way to create a list using recurrence rules is the rrulestr() function
#We will pass the rrule and the start date, which will return an rrule object.
#We can then cast this object as a list!

#Create a list of every 10 days (5 instances only) starting on 1/1/2020 using a rrulestr.
tm_rstr = rrulestr('FREQ=DAILY;INTERVAL=10;COUNT=5',
               dtstart=parse('20200101T090000'))
print(type(tm_rstr))
tm_list = list(tm_rstr)
print(tm_list)


#Activity Exercise #1:
#Using the data contained in the Appointments.txt file,
#find the sum, average, and standard deviation for all appointments held on Sundays.
#The amount of each appointment is the sum of the AdministrationCharge + DoctorCharge + OtherCharge
#Then, print the following message:
#The sum of sales on Sunday is $2,578,013.00
#The average of sales on Sunday is $349.99 with a standard deviation of $158.86
import csv
from dateutil.rrule import *
from dateutil.parser import *
handle = open('Appointments.txt')
mylist = []
data = csv.reader(handle, delimiter = '|')
for row in data:
    mytuple = tuple(row)
    if mytuple[0] != 'Patient_Id':
        mylist.append(mytuple)

handle.close()
for row in mylist:
    parsed_date = parse(row[2])
    if parsed_date.strftime('%A') == 'SU':
        printx("x")

#Activity Exercise #2:
#Do older patients generate more revenue than younger patients?
##Additionally, do older patients pay more and faster of their appointment fees than younger patients?
#For this analysis, older patients are patients who are 50 years of age or older (i.e., >=50).
#Calculate the patients age at the date of the appointment.
#The patients date of birth is contained in the patients.xml file.
#The payments data is contained in the invoices.json file.

#Note that each row in the appointments.txt file is uniquely identified by the combination of the
#patient_id, doctor_id, and appointment date.

#Each invoice is linked back to the appointments using the combination of the
#patient_id, doctor_id, and appointment date fields.

#When finished, print the following messages:
#The sum of revenue for young patients is $11,032,270.
#The sum of invoice collections for young patients is $9,951,044.68.
#The percent of revenue collected for young patients is 90.20%.
#Young patients pay their bills (on average) in 42.75 days.
#
#The sum of revenue for old patients is $6,754,268.
#The sum of invoice collections for old patients is $6,090,475.23.
#The percent of revenue collected for old patients is 90.17%.
#Older patients pay their bills (on average) in 43.04 days.

