---
layout:     post
title:      First Class Functions in Python
author:     Ed Henry
tags: [Python]
---

## First Class Functions

Typically first class functions are defined as a programming entity that can be :

* Created at runtime
* Assigned to a variable or element in a data structure
* Passed as an argument
* Returned as the result of a function

By this definition, looking at how Python treats all functions, all functions are first class within Python.

Below we'll see examples of exactly how this looks.

#### Treating a function like an object


```python
def factorial(n):
    """
    Returns n! or n(factorial)
    
    e.g 5! = 5 * 4 * 3 * 2
    """
    return 1 if n < 2 else n * factorial(n-1)

factorial(5)
```




    120



#### First class analysis

We can show the first class nature of this function object using a few examples.

We can assign the function to a variable, which will invoke the function when calling that variable.


```python
fact = factorial
fact(5)
```




    120



We can also use the map function, and pass our function as the first argument, allowing that function to be evaluated against the second argument, which is an iterable. Allowing this function to be applied in a successive fashion to all elements of this iterable.


```python
map(factorial, range(10))
```




    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]



#### Higher-Order Functions

A higher order function is a bit....meta. It can take, as an argument, a function and then returns a function as a result.

The `map()` example used above is a great example of this.

The built-in `sorted()` function is another great example of this, within Python. We can pass it an iterable, along with a `key` that can then be applied in succession to the items in the list. 


```python
food = ['eggplant', 'carrots', 'celery', 
        'potatoes', 'tomatoes', 'rhubarb',
        'strawberry', 'blueberry', 'raspberry',
        'banana', 'cherry']

print(sorted(food, key=len))
```

    ['celery', 'banana', 'cherry', 'carrots', 'rhubarb', 'eggplant', 'potatoes', 'tomatoes', 'blueberry', 'raspberry', 'strawberry']


Any single argument function can be used in the key argument of the `sorted()` method.

as a trivial example, we may want to use the reversed order of the characters to sort of words, as this will cause certain clustering of character strings together, such as -berry, and -toes.


```python
def reverse(word):
    '''
    Reverse the order of the letters in a given string
    '''
    return word[::-1]

print(sorted(food, key=reverse))
```

    ['banana', 'rhubarb', 'tomatoes', 'potatoes', 'carrots', 'eggplant', 'celery', 'blueberry', 'raspberry', 'strawberry', 'cherry']


#### Replacements for map and filter

Map, filter, and reduce are typically offered in functional languages as higher order functions. However, the introduction of list comprehensions and generator expressions have downplayed the value of the map and filter functions, as listcomp's and genexp's combine the job of `map` and `filter`.


```python
# Build a list of factorials from 0! to 5!
list(map(fact, range(6)))
```




    [1, 1, 2, 6, 24, 120]




```python
# Build a list of factorials from 0! to 5!
# but using list comprehension
[fact(n) for n in range(6)]
```




    [1, 1, 2, 6, 24, 120]




```python
# Build a list of factorials of odd numbers up to 5!, using `map` and `filter`
list(map(factorial, filter(lambda n: n % 2, range(6))))
```




    [1, 6, 120]



We can see above that with the `map` and `filter` functions, we needed to use a `lambda` function. 

Using a list comprehension can remove this requirement, and concatenate the operations.


```python
# Build a list of factorials of odd numbers up to 5!, using list comprehension
[factorial(n) for n in range(6) if n % 2]
```




    [1, 6, 120]



#### Anonymous Functions

The example above, where we've utilized `map` and `filter` combined with a `lambda` function leads us into our next example.

The `lambda` keyword created an anonymous function within a Python expression. However the syntax limits the `lambda` to be pure expressions. This means that the body of a `lambda` function can't use other Python statements such as `while` or `try`, etc.

These are typically limited in their use because of the lack of the ability to use more complex control structures within the `lambda` functions. This can lead to unreadable or unworkable results.

However, they can still prove useful in certain contexts, such as list arguments.


```python
food = ['eggplant', 'carrots', 'celery', 
        'potatoes', 'tomatoes', 'rhubarb',
        'strawberry', 'blueberry', 'raspberry',
        'banana', 'cherry']

print(sorted(food, key=lambda word: word[::-1]))
```

    ['banana', 'rhubarb', 'tomatoes', 'potatoes', 'carrots', 'eggplant', 'celery', 'blueberry', 'raspberry', 'strawberry', 'cherry']


#### References

* Fluent Python, Ramalho [Purchase Link](http://shop.oreilly.com/product/0636920032519.do)
