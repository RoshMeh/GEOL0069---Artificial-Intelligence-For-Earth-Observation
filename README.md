# Unsupervised Learning Methods

## Overview
This project explores various unsupervised learning methods for data clustering and classification.

### Methods Covered:
- K-Means Clustering
- Gaussian Mixture Models (GMM)
- Data Preprocessing and Visualization

## Installation
To run this project, install the following dependencies:
```bash
pip install numpy pandas sklearn matplotlib seaborn
```

## Project Details
#
 
U
n
s
u
p
e
r
v
i
s
e
d
 
L
e
a
r
n
i
n
g




T
h
i
s
 
s
e
c
t
i
o
n
 
m
a
r
k
s
 
o
u
r
 
j
o
u
r
n
e
y
 
i
n
t
o
 
a
n
o
t
h
e
r
 
s
i
g
n
i
f
i
c
a
n
t
 
d
o
m
a
i
n
 
o
f
 
m
a
c
h
i
n
e
 
l
e
a
r
n
i
n
g
 
a
n
d
 
A
I
:
 
u
n
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
.
 
R
a
t
h
e
r
 
t
h
a
n
 
d
e
l
v
i
n
g
 
d
e
e
p
 
i
n
t
o
 
t
h
e
o
r
e
t
i
c
a
l
 
i
n
t
r
i
c
a
c
i
e
s
,
 
o
u
r
 
f
o
c
u
s
 
h
e
r
e
 
w
i
l
l
 
b
e
 
o
n
 
o
f
f
e
r
i
n
g
 
a
 
p
r
a
c
t
i
c
a
l
 
g
u
i
d
e
.
 
W
e
 
a
i
m
 
t
o
 
e
q
u
i
p
 
y
o
u
 
w
i
t
h
 
a
 
c
l
e
a
r
 
u
n
d
e
r
s
t
a
n
d
i
n
g
 
a
n
d
 
e
f
f
e
c
t
i
v
e
 
t
o
o
l
s
 
f
o
r
 
e
m
p
l
o
y
i
n
g
 
u
n
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
 
m
e
t
h
o
d
s
 
i
n
 
r
e
a
l
-
w
o
r
l
d
 
(
E
O
)
 
s
c
e
n
a
r
i
o
s
.




I
t
'
s
 
i
m
p
o
r
t
a
n
t
 
t
o
 
n
o
t
e
 
t
h
a
t
,
 
w
h
i
l
e
 
u
n
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
 
e
n
c
o
m
p
a
s
s
e
s
 
a
 
b
r
o
a
d
 
r
a
n
g
e
 
o
f
 
a
p
p
l
i
c
a
t
i
o
n
s
,
 
o
u
r
 
d
i
s
c
u
s
s
i
o
n
 
w
i
l
l
 
p
r
e
d
o
m
i
n
a
n
t
l
y
 
r
e
v
o
l
v
e
 
a
r
o
u
n
d
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
t
a
s
k
s
.
 
T
h
i
s
 
i
s
 
b
e
c
a
u
s
e
 
u
n
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
 
t
e
c
h
n
i
q
u
e
s
 
a
r
e
 
e
x
c
e
p
t
i
o
n
a
l
l
y
 
a
d
e
p
t
 
a
t
 
i
d
e
n
t
i
f
y
i
n
g
 
p
a
t
t
e
r
n
s
 
a
n
d
 
c
a
t
e
g
o
r
i
s
i
n
g
 
d
a
t
a
 
w
h
e
n
 
t
h
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
s
 
a
r
e
 
n
o
t
 
e
x
p
l
i
c
i
t
l
y
 
l
a
b
e
l
e
d
.
 
B
y
 
e
x
p
l
o
r
i
n
g
 
t
h
e
s
e
 
t
e
c
h
n
i
q
u
e
s
,
 
y
o
u
'
l
l
 
g
a
i
n
 
i
n
s
i
g
h
t
s
 
i
n
t
o
 
h
o
w
 
t
o
 
d
i
s
c
e
r
n
 
s
t
r
u
c
t
u
r
e
 
a
n
d
 
r
e
l
a
t
i
o
n
s
h
i
p
s
 
w
i
t
h
i
n
 
y
o
u
r
 
d
a
t
a
s
e
t
s
,
 
e
v
e
n
 
i
n
 
t
h
e
 
a
b
s
e
n
c
e
 
o
f
 
p
r
e
d
e
f
i
n
e
d
 
c
a
t
e
g
o
r
i
e
s
 
o
r
 
l
a
b
e
l
s
.




T
h
e
 
t
a
s
k
s
 
i
n
 
t
h
i
s
 
n
o
t
e
b
o
o
k
 
w
i
l
l
 
b
e
 
m
a
i
n
l
y
 
t
w
o
:


1
.
 
D
i
s
c
r
i
m
i
n
a
t
i
o
n
 
o
f
 
S
e
a
 
i
c
e
 
a
n
d
 
l
e
a
d
 
b
a
s
e
d
 
o
n
 
i
m
a
g
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
b
a
s
e
d
 
o
n
 
S
e
n
t
i
n
e
l
-
2
 
o
p
t
i
c
a
l
 
d
a
t
a
.


2
.
 
D
i
s
c
r
i
m
i
n
a
t
i
o
n
 
o
f
 
S
e
a
 
i
c
e
 
a
n
d
 
l
e
a
d
 
b
a
s
e
d
 
o
n
 
a
l
t
i
m
e
t
r
y
 
d
a
t
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
b
a
s
e
d
 
o
n
 
S
e
n
t
i
n
e
l
-
3
 
a
l
t
i
m
e
t
r
y
 
d
a
t
a
.
#
#
 
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
U
n
s
u
p
e
r
v
i
s
e
d
 
L
e
a
r
n
i
n
g
 
M
e
t
h
o
d
s
 
{
c
i
t
e
}
`
b
i
s
h
o
p
2
0
0
6
p
a
t
t
e
r
n
`




#
#
#
 
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
K
-
m
e
a
n
s
 
C
l
u
s
t
e
r
i
n
g




K
-
m
e
a
n
s
 
c
l
u
s
t
e
r
i
n
g
 
i
s
 
a
 
t
y
p
e
 
o
f
 
u
n
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
 
a
l
g
o
r
i
t
h
m
 
u
s
e
d
 
f
o
r
 
p
a
r
t
i
t
i
o
n
i
n
g
 
a
 
d
a
t
a
s
e
t
 
i
n
t
o
 
a
 
s
e
t
 
o
f
 
k
 
g
r
o
u
p
s
 
(
o
r
 
c
l
u
s
t
e
r
s
)
,
 
w
h
e
r
e
 
k
 
r
e
p
r
e
s
e
n
t
s
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
g
r
o
u
p
s
 
p
r
e
-
s
p
e
c
i
f
i
e
d
 
b
y
 
t
h
e
 
a
n
a
l
y
s
t
.
 
I
t
 
c
l
a
s
s
i
f
i
e
s
 
t
h
e
 
d
a
t
a
 
p
o
i
n
t
s
 
b
a
s
e
d
 
o
n
 
t
h
e
 
s
i
m
i
l
a
r
i
t
y
 
o
f
 
t
h
e
 
f
e
a
t
u
r
e
s
 
o
f
 
t
h
e
 
d
a
t
a
 
{
c
i
t
e
}
`
m
a
c
q
u
e
e
n
1
9
6
7
s
o
m
e
`
.
 
T
h
e
 
b
a
s
i
c
 
i
d
e
a
 
i
s
 
t
o
 
d
e
f
i
n
e
 
k
 
c
e
n
t
r
o
i
d
s
,
 
o
n
e
 
f
o
r
 
e
a
c
h
 
c
l
u
s
t
e
r
,
 
a
n
d
 
t
h
e
n
 
a
s
s
i
g
n
 
e
a
c
h
 
d
a
t
a
 
p
o
i
n
t
 
t
o
 
t
h
e
 
n
e
a
r
e
s
t
 
c
e
n
t
r
o
i
d
,
 
w
h
i
l
e
 
k
e
e
p
i
n
g
 
t
h
e
 
c
e
n
t
r
o
i
d
s
 
a
s
 
s
m
a
l
l
 
a
s
 
p
o
s
s
i
b
l
e
.




#
#
#
 
W
h
y
 
K
-
m
e
a
n
s
 
f
o
r
 
C
l
u
s
t
e
r
i
n
g
?




K
-
m
e
a
n
s
 
c
l
u
s
t
e
r
i
n
g
 
i
s
 
p
a
r
t
i
c
u
l
a
r
l
y
 
w
e
l
l
-
s
u
i
t
e
d
 
f
o
r
 
a
p
p
l
i
c
a
t
i
o
n
s
 
w
h
e
r
e
:




-
 
*
*
T
h
e
 
s
t
r
u
c
t
u
r
e
 
o
f
 
t
h
e
 
d
a
t
a
 
i
s
 
n
o
t
 
k
n
o
w
n
 
b
e
f
o
r
e
h
a
n
d
*
*
:
 
K
-
m
e
a
n
s
 
d
o
e
s
n
’
t
 
r
e
q
u
i
r
e
 
a
n
y
 
p
r
i
o
r
 
k
n
o
w
l
e
d
g
e
 
a
b
o
u
t
 
t
h
e
 
d
a
t
a
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
r
 
s
t
r
u
c
t
u
r
e
,
 
m
a
k
i
n
g
 
i
t
 
i
d
e
a
l
 
f
o
r
 
e
x
p
l
o
r
a
t
o
r
y
 
d
a
t
a
 
a
n
a
l
y
s
i
s
.


-
 
*
*
S
i
m
p
l
i
c
i
t
y
 
a
n
d
 
s
c
a
l
a
b
i
l
i
t
y
*
*
:
 
T
h
e
 
a
l
g
o
r
i
t
h
m
 
i
s
 
s
t
r
a
i
g
h
t
f
o
r
w
a
r
d
 
t
o
 
i
m
p
l
e
m
e
n
t
 
a
n
d
 
c
a
n
 
s
c
a
l
e
 
t
o
 
l
a
r
g
e
 
d
a
t
a
s
e
t
s
 
r
e
l
a
t
i
v
e
l
y
 
e
a
s
i
l
y
.




#
#
#
 
K
e
y
 
C
o
m
p
o
n
e
n
t
s
 
o
f
 
K
-
m
e
a
n
s




1
.
 
*
*
C
h
o
o
s
i
n
g
 
K
*
*
:
 
T
h
e
 
n
u
m
b
e
r
 
o
f
 
c
l
u
s
t
e
r
s
 
(
k
)
 
i
s
 
a
 
p
a
r
a
m
e
t
e
r
 
t
h
a
t
 
n
e
e
d
s
 
t
o
 
b
e
 
s
p
e
c
i
f
i
e
d
 
b
e
f
o
r
e
 
a
p
p
l
y
i
n
g
 
t
h
e
 
a
l
g
o
r
i
t
h
m
.


2
.
 
*
*
C
e
n
t
r
o
i
d
s
 
I
n
i
t
i
a
l
i
z
a
t
i
o
n
*
*
:
 
T
h
e
 
i
n
i
t
i
a
l
 
p
l
a
c
e
m
e
n
t
 
o
f
 
t
h
e
 
c
e
n
t
r
o
i
d
s
 
c
a
n
 
a
f
f
e
c
t
 
t
h
e
 
f
i
n
a
l
 
r
e
s
u
l
t
s
.


3
.
 
*
*
A
s
s
i
g
n
m
e
n
t
 
S
t
e
p
*
*
:
 
E
a
c
h
 
d
a
t
a
 
p
o
i
n
t
 
i
s
 
a
s
s
i
g
n
e
d
 
t
o
 
i
t
s
 
n
e
a
r
e
s
t
 
c
e
n
t
r
o
i
d
,
 
b
a
s
e
d
 
o
n
 
t
h
e
 
s
q
u
a
r
e
d
 
E
u
c
l
i
d
e
a
n
 
d
i
s
t
a
n
c
e
.


4
.
 
*
*
U
p
d
a
t
e
 
S
t
e
p
*
*
:
 
T
h
e
 
c
e
n
t
r
o
i
d
s
 
a
r
e
 
r
e
c
o
m
p
u
t
e
d
 
a
s
 
t
h
e
 
c
e
n
t
e
r
 
o
f
 
a
l
l
 
t
h
e
 
d
a
t
a
 
p
o
i
n
t
s
 
a
s
s
i
g
n
e
d
 
t
o
 
t
h
e
 
r
e
s
p
e
c
t
i
v
e
 
c
l
u
s
t
e
r
.




#
#
#
 
T
h
e
 
I
t
e
r
a
t
i
v
e
 
P
r
o
c
e
s
s
 
o
f
 
K
-
m
e
a
n
s




T
h
e
 
a
s
s
i
g
n
m
e
n
t
 
a
n
d
 
u
p
d
a
t
e
 
s
t
e
p
s
 
a
r
e
 
r
e
p
e
a
t
e
d
 
i
t
e
r
a
t
i
v
e
l
y
 
u
n
t
i
l
 
t
h
e
 
c
e
n
t
r
o
i
d
s
 
n
o
 
l
o
n
g
e
r
 
m
o
v
e
 
s
i
g
n
i
f
i
c
a
n
t
l
y
,
 
m
e
a
n
i
n
g
 
t
h
e
 
w
i
t
h
i
n
-
c
l
u
s
t
e
r
 
v
a
r
i
a
t
i
o
n
 
i
s
 
m
i
n
i
m
i
s
e
d
.
 
T
h
i
s
 
i
t
e
r
a
t
i
v
e
 
p
r
o
c
e
s
s
 
e
n
s
u
r
e
s
 
t
h
a
t
 
t
h
e
 
a
l
g
o
r
i
t
h
m
 
c
o
n
v
e
r
g
e
s
 
t
o
 
a
 
r
e
s
u
l
t
,
 
w
h
i
c
h
 
m
i
g
h
t
 
b
e
 
a
 
l
o
c
a
l
 
o
p
t
i
m
u
m
.




#
#
#
 
A
d
v
a
n
t
a
g
e
s
 
o
f
 
K
-
m
e
a
n
s




-
 
*
*
E
f
f
i
c
i
e
n
c
y
*
*
:
 
K
-
m
e
a
n
s
 
i
s
 
c
o
m
p
u
t
a
t
i
o
n
a
l
l
y
 
e
f
f
i
c
i
e
n
t
.


-
 
*
*
E
a
s
e
 
o
f
 
i
n
t
e
r
p
r
e
t
a
t
i
o
n
*
*
:
 
T
h
e
 
r
e
s
u
l
t
s
 
o
f
 
k
-
m
e
a
n
s
 
c
l
u
s
t
e
r
i
n
g
 
a
r
e
 
e
a
s
y
 
t
o
 
u
n
d
e
r
s
t
a
n
d
 
a
n
d
 
i
n
t
e
r
p
r
e
t
.




#
#
#
 
B
a
s
i
c
 
C
o
d
e
 
I
m
p
l
e
m
e
n
t
a
t
i
o
n




B
e
l
o
w
,
 
y
o
u
'
l
l
 
f
i
n
d
 
a
 
b
a
s
i
c
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
o
f
 
t
h
e
 
K
-
m
e
a
n
s
 
c
l
u
s
t
e
r
i
n
g
 
a
l
g
o
r
i
t
h
m
.
 
T
h
i
s
 
s
e
r
v
e
s
 
a
s
 
a
 
f
o
u
n
d
a
t
i
o
n
a
l
 
u
n
d
e
r
s
t
a
n
d
i
n
g
 
a
n
d
 
a
 
s
t
a
r
t
i
n
g
 
p
o
i
n
t
 
f
o
r
 
a
p
p
l
y
i
n
g
 
t
h
e
 
a
l
g
o
r
i
t
h
m
 
t
o
 
y
o
u
r
 
s
p
e
c
i
f
i
c
 
d
a
t
a
 
a
n
a
l
y
s
i
s
 
t
a
s
k
s
.


#
#
 
G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
s
 
(
G
M
M
)
 
{
c
i
t
e
}
`
b
i
s
h
o
p
2
0
0
6
p
a
t
t
e
r
n
`




#
#
#
 
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
s




G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
s
 
(
G
M
M
)
 
a
r
e
 
a
 
p
r
o
b
a
b
i
l
i
s
t
i
c
 
m
o
d
e
l
 
f
o
r
 
r
e
p
r
e
s
e
n
t
i
n
g
 
n
o
r
m
a
l
l
y
 
d
i
s
t
r
i
b
u
t
e
d
 
s
u
b
p
o
p
u
l
a
t
i
o
n
s
 
w
i
t
h
i
n
 
a
n
 
o
v
e
r
a
l
l
 
p
o
p
u
l
a
t
i
o
n
.
 
T
h
e
 
m
o
d
e
l
 
a
s
s
u
m
e
s
 
t
h
a
t
 
t
h
e
 
d
a
t
a
 
i
s
 
g
e
n
e
r
a
t
e
d
 
f
r
o
m
 
a
 
m
i
x
t
u
r
e
 
o
f
 
s
e
v
e
r
a
l
 
G
a
u
s
s
i
a
n
 
d
i
s
t
r
i
b
u
t
i
o
n
s
,
 
e
a
c
h
 
w
i
t
h
 
i
t
s
 
o
w
n
 
m
e
a
n
 
a
n
d
 
v
a
r
i
a
n
c
e
 
{
c
i
t
e
}
`
r
e
y
n
o
l
d
s
2
0
0
9
g
a
u
s
s
i
a
n
,
 
m
c
l
a
c
h
l
a
n
2
0
0
4
f
i
n
i
t
e
`
.
 
G
M
M
s
 
a
r
e
 
w
i
d
e
l
y
 
u
s
e
d
 
f
o
r
 
c
l
u
s
t
e
r
i
n
g
 
a
n
d
 
d
e
n
s
i
t
y
 
e
s
t
i
m
a
t
i
o
n
,
 
a
s
 
t
h
e
y
 
p
r
o
v
i
d
e
 
a
 
m
e
t
h
o
d
 
f
o
r
 
r
e
p
r
e
s
e
n
t
i
n
g
 
c
o
m
p
l
e
x
 
d
i
s
t
r
i
b
u
t
i
o
n
s
 
t
h
r
o
u
g
h
 
t
h
e
 
c
o
m
b
i
n
a
t
i
o
n
 
o
f
 
s
i
m
p
l
e
r
 
o
n
e
s
.




#
#
#
 
W
h
y
 
G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
s
 
f
o
r
 
C
l
u
s
t
e
r
i
n
g
?




G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
s
 
a
r
e
 
p
a
r
t
i
c
u
l
a
r
l
y
 
p
o
w
e
r
f
u
l
 
i
n
 
s
c
e
n
a
r
i
o
s
 
w
h
e
r
e
:




-
 
*
*
S
o
f
t
 
c
l
u
s
t
e
r
i
n
g
 
i
s
 
n
e
e
d
e
d
*
*
:
 
U
n
l
i
k
e
 
K
-
m
e
a
n
s
,
 
G
M
M
 
p
r
o
v
i
d
e
s
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
e
a
c
h
 
d
a
t
a
 
p
o
i
n
t
 
b
e
l
o
n
g
i
n
g
 
t
o
 
e
a
c
h
 
c
l
u
s
t
e
r
,
 
o
f
f
e
r
i
n
g
 
a
 
s
o
f
t
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
n
d
 
u
n
d
e
r
s
t
a
n
d
i
n
g
 
o
f
 
t
h
e
 
u
n
c
e
r
t
a
i
n
t
i
e
s
 
i
n
 
o
u
r
 
d
a
t
a
.


-
 
*
*
F
l
e
x
i
b
i
l
i
t
y
 
i
n
 
c
l
u
s
t
e
r
 
c
o
v
a
r
i
a
n
c
e
*
*
:
 
G
M
M
 
a
l
l
o
w
s
 
f
o
r
 
c
l
u
s
t
e
r
s
 
t
o
 
h
a
v
e
 
d
i
f
f
e
r
e
n
t
 
s
i
z
e
s
 
a
n
d
 
d
i
f
f
e
r
e
n
t
 
s
h
a
p
e
s
,
 
m
a
k
i
n
g
 
i
t
 
m
o
r
e
 
f
l
e
x
i
b
l
e
 
t
o
 
c
a
p
t
u
r
e
 
t
h
e
 
t
r
u
e
 
v
a
r
i
a
n
c
e
 
i
n
 
t
h
e
 
d
a
t
a
.




#
#
#
 
K
e
y
 
C
o
m
p
o
n
e
n
t
s
 
o
f
 
G
M
M




1
.
 
*
*
N
u
m
b
e
r
 
o
f
 
C
o
m
p
o
n
e
n
t
s
 
(
G
a
u
s
s
i
a
n
s
)
*
*
:
 
S
i
m
i
l
a
r
 
t
o
 
K
 
i
n
 
K
-
m
e
a
n
s
,
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
G
a
u
s
s
i
a
n
s
 
(
c
o
m
p
o
n
e
n
t
s
)
 
i
s
 
a
 
p
a
r
a
m
e
t
e
r
 
t
h
a
t
 
n
e
e
d
s
 
t
o
 
b
e
 
s
e
t
.


2
.
 
*
*
E
x
p
e
c
t
a
t
i
o
n
-
M
a
x
i
m
i
z
a
t
i
o
n
 
(
E
M
)
 
A
l
g
o
r
i
t
h
m
*
*
:
 
G
M
M
s
 
u
s
e
 
t
h
e
 
E
M
 
a
l
g
o
r
i
t
h
m
 
f
o
r
 
f
i
t
t
i
n
g
,
 
i
t
e
r
a
t
i
v
e
l
y
 
i
m
p
r
o
v
i
n
g
 
t
h
e
 
l
i
k
e
l
i
h
o
o
d
 
o
f
 
t
h
e
 
d
a
t
a
 
g
i
v
e
n
 
t
h
e
 
m
o
d
e
l
.


3
.
 
*
*
C
o
v
a
r
i
a
n
c
e
 
T
y
p
e
*
*
:
 
T
h
e
 
s
h
a
p
e
,
 
s
i
z
e
,
 
a
n
d
 
o
r
i
e
n
t
a
t
i
o
n
 
o
f
 
t
h
e
 
c
l
u
s
t
e
r
s
 
a
r
e
 
d
e
t
e
r
m
i
n
e
d
 
b
y
 
t
h
e
 
c
o
v
a
r
i
a
n
c
e
 
t
y
p
e
 
o
f
 
t
h
e
 
G
a
u
s
s
i
a
n
s
 
(
e
.
g
.
,
 
s
p
h
e
r
i
c
a
l
,
 
d
i
a
g
o
n
a
l
,
 
t
i
e
d
,
 
o
r
 
f
u
l
l
 
c
o
v
a
r
i
a
n
c
e
)
.




#
#
#
 
T
h
e
 
E
M
 
A
l
g
o
r
i
t
h
m
 
i
n
 
G
M
M




T
h
e
 
E
x
p
e
c
t
a
t
i
o
n
-
M
a
x
i
m
i
z
a
t
i
o
n
 
(
E
M
)
 
a
l
g
o
r
i
t
h
m
 
i
s
 
a
 
t
w
o
-
s
t
e
p
 
p
r
o
c
e
s
s
:




-
 
*
*
E
x
p
e
c
t
a
t
i
o
n
 
S
t
e
p
 
(
E
-
s
t
e
p
)
*
*
:
 
C
a
l
c
u
l
a
t
e
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
t
h
a
t
 
e
a
c
h
 
d
a
t
a
 
p
o
i
n
t
 
b
e
l
o
n
g
s
 
t
o
 
e
a
c
h
 
c
l
u
s
t
e
r
.


-
 
*
*
M
a
x
i
m
i
z
a
t
i
o
n
 
S
t
e
p
 
(
M
-
s
t
e
p
)
*
*
:
 
U
p
d
a
t
e
 
t
h
e
 
p
a
r
a
m
e
t
e
r
s
 
o
f
 
t
h
e
 
G
a
u
s
s
i
a
n
s
 
(
m
e
a
n
,
 
c
o
v
a
r
i
a
n
c
e
,
 
a
n
d
 
m
i
x
i
n
g
 
c
o
e
f
f
i
c
i
e
n
t
)
 
t
o
 
m
a
x
i
m
i
z
e
 
t
h
e
 
l
i
k
e
l
i
h
o
o
d
 
o
f
 
t
h
e
 
d
a
t
a
 
g
i
v
e
n
 
t
h
e
s
e
 
a
s
s
i
g
n
m
e
n
t
s
.




T
h
i
s
 
p
r
o
c
e
s
s
 
i
s
 
r
e
p
e
a
t
e
d
 
u
n
t
i
l
 
c
o
n
v
e
r
g
e
n
c
e
,
 
m
e
a
n
i
n
g
 
t
h
e
 
p
a
r
a
m
e
t
e
r
s
 
d
o
 
n
o
t
 
s
i
g
n
i
f
i
c
a
n
t
l
y
 
c
h
a
n
g
e
 
f
r
o
m
 
o
n
e
 
i
t
e
r
a
t
i
o
n
 
t
o
 
t
h
e
 
n
e
x
t
.




#
#
#
 
A
d
v
a
n
t
a
g
e
s
 
o
f
 
G
M
M




-
 
*
*
S
o
f
t
 
C
l
u
s
t
e
r
i
n
g
*
*
:
 
P
r
o
v
i
d
e
s
 
a
 
p
r
o
b
a
b
i
l
i
s
t
i
c
 
f
r
a
m
e
w
o
r
k
 
f
o
r
 
s
o
f
t
 
c
l
u
s
t
e
r
i
n
g
,
 
g
i
v
i
n
g
 
m
o
r
e
 
i
n
f
o
r
m
a
t
i
o
n
 
a
b
o
u
t
 
t
h
e
 
u
n
c
e
r
t
a
i
n
t
i
e
s
 
i
n
 
t
h
e
 
d
a
t
a
 
a
s
s
i
g
n
m
e
n
t
s
.


-
 
*
*
C
l
u
s
t
e
r
 
S
h
a
p
e
 
F
l
e
x
i
b
i
l
i
t
y
*
*
:
 
C
a
n
 
a
d
a
p
t
 
t
o
 
e
l
l
i
p
s
o
i
d
a
l
 
c
l
u
s
t
e
r
 
s
h
a
p
e
s
,
 
t
h
a
n
k
s
 
t
o
 
t
h
e
 
f
l
e
x
i
b
l
e
 
c
o
v
a
r
i
a
n
c
e
 
s
t
r
u
c
t
u
r
e
.




#
#
#
 
B
a
s
i
c
 
C
o
d
e
 
I
m
p
l
e
m
e
n
t
a
t
i
o
n




B
e
l
o
w
,
 
y
o
u
'
l
l
 
f
i
n
d
 
a
 
b
a
s
i
c
 
i
m
p
l
e
m
e
n
t
a
t
i
o
n
 
o
f
 
t
h
e
 
G
a
u
s
s
i
a
n
 
M
i
x
t
u
r
e
 
M
o
d
e
l
.
 
T
h
i
s
 
s
h
o
u
l
d
 
s
e
r
v
e
 
a
s
 
a
n
 
i
n
i
t
i
a
l
 
g
u
i
d
e
 
f
o
r
 
u
n
d
e
r
s
t
a
n
d
i
n
g
 
t
h
e
 
m
o
d
e
l
 
a
n
d
 
a
p
p
l
y
i
n
g
 
i
t
 
t
o
 
y
o
u
r
 
d
a
t
a
 
a
n
a
l
y
s
i
s
 
p
r
o
j
e
c
t
s
.


## Example Code
```python
f
r
o
m
 
g
o
o
g
l
e
.
c
o
l
a
b
 
i
m
p
o
r
t
 
d
r
i
v
e


d
r
i
v
e
.
m
o
u
n
t
(
'
/
c
o
n
t
e
n
t
/
d
r
i
v
e
'
)
```
