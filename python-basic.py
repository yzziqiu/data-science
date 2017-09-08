
#!/usr/bin/env python
# -*- coding: utf-8 -*-
chmod a+x hello.py
./hello,py
name = raw_input('please enter your name: ')
print 'hello,', name

# multiple lines
print '''line1
... line2
... line3'''

# char<->utf-8
u'ABC'.encode('utf-8')
#'ABC'
u'中文'.encode('utf-8')
# '\xe4\xb8\xad\xe6\x96\x87'

print '\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')
# 中文
'Hi, %s, you have $%d.' % ('Michael', 1000000)
# 'Hi, Michael, you have $1000000.'

u'中文'.encode('gb2312')
# '\xd6\xd0\xce\xc4'

# last element of list: list[-1]
classmates.append('Adam')
classmates.insert(1, 'Jack')
classmates.pop(1)

# multiple types or lists in lists

# tuple of one element
t = (1,)

# loop
sum = 0
for x in range(101):
    sum = sum + x
print sum   

# function
cmp(1, 2)
# -1
def my_abs(x):
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')

# return multiple values == return a tuple
