---
layout:     post
title:      Sequential and Binary Search in Python
author:     Ed Henry
---


This notebook will include examples of searching and sorting algorithms implemented in python. It is both for my own learning, and for anyone else who would like to use this notebook for anything they'd like.

## Searching

Finding an item in a collection of items is a pretty typical search problem. Depending on the implementation, a search will tend to return a `True` or `False` boolean answer to the question of "is this item contained within this collection of items?".

An example of this can be seen below, using Pythons `in` operator.


```python
# Finding a single integer in an array of integers using Python's `in` 
# operator

15 in [3,5,6,9,12,11]
```




    False



We can see this returns a boolean answer of `False`, indicating that the integer isn't present in the array.

Below is another example where the answer is `True`.


```python
# Finding a single integer in an array of integers using Python's `in` 
# operator

11 in [3,5,6,9,12,11]
```




    True



Python provides useful abstractions like this for a lot of search and sort functionality, but it's important to understand what's going on 'under the hood' of these functions.

## Sequential Search

### Unordered array

Datum, in arrays such as the ones used in the examples above, are typically stores in a collection such as a list. These datum within these lists have linear, or sequential relationship. They are each stores in a position within the array, relative to the other datum.

When searching for a specific datum within the array, we are able to seqeuntially evaluate each item in the list, or array, to see if it matches the item we're looking for.

Using `sequential_search`, we simply move from item to item in the list, evaluating whether our search expression is `True`, or `False`.


```python
# Search sequentially through a list, incrementing the position counter
# if is_present is not True, otherwise set is_present to True and return

def sequential_search(li, item):
    position = 0
    is_present = False
    
    while position < len(li) and not is_present:
        if li[position] == item:
            is_present = True
        else:
            position = position + 1
    
    return is_present

test_array = [1,31,5,18,7,10,25]
print(sequential_search(test_array, 2))
print(sequential_search(test_array, 25))
```

    False
    True


The example above uses an example of uses an unordered list. Because this list is unordered, we will need to evaluate every item in the list to understand if it is the item that we're searching for. Because this is the case, the computational complexity of our `sequential_search` function is $$O(n)$$.

Here is a table summarizing the cases :

| Case               | Best Case | Worst Case | Average Case |
|--------------------|-----------|------------|--------------|
| item is present    | 1         | $$n$$        | $$\frac{n}{2}$$|
| item isn't present | $$n$$       | $$n$$        | $$n$$          |

This can be seen as such :

For every $$n$$ and every input size of $$n$$, the following is true:

* The while loop is executed at most $$n$$ times
* `position` is incremented on each iteration, so `position` > $$n$$ after $$n$$ iterations.
* Each iteration takes $$c$$ steps for some constant $$c$$
* $$d$$ steps are taken outside of the loop, for some constant $$d$$

Therefore for *all* inputs of size $$n$$, the time needed for the entire search is **at most** $$(cn+d) = O(n)$$.

At worst, the item $$x$$ we're searching for is the _last_ item in the entire list of items. This can be seen as 

$$A[n] = x$$ and $$A[i] \ne x$$ for all $$i$$ s.t. $$1 \le i \lt n$$

### Ordered array

If we assume that the list, or array, that we're searching over is ordered, say from low to high, the chance of the item we're looking for being in any one of the $$n$$ positions is still the same. However, if the item is _not_ present we have a slight advantage in that the item that we're looking for may never be present past another item of greater value.

For example, if we're looking for the number 25, and through the process of searching through the array, we happen upon the number 27, we know that no other integers past number 27 will have the value that we're looking for.


```python
def ordered_sequential_search(li, item):
    position = 0
    found = False
    stop = False
    
    while position < len(li) and not found and not stop:
        if li[position] == item:
            found == True
        else:
            if li[position] > item:
                stop = True
            else:
                position = (position + 1)
    
    return found

test_li = [0,2,3,4,5,6,7,12,15,18,23,27,45]
print(ordered_sequential_search(test_li, 25))
```

    False


We can see that we are able to terminate the execution of the search because we've found a number greater than the number we're searching for with the assumption that the list being passed into the function is ordered, we know we can terminate the computation. 

Modifying the table above, we can see that with the item _not_ present in our array, we save some computational cycles in the negative case.

| Case               | Best Case | Worst Case | Average Case |
|--------------------|-----------|------------|--------------|
| item is present    | 1         | $$n$$        | $$\frac{n}{2}$$|
| item isn't present | $$n$$       | $$n$$        | $$\frac{n}{2}$$|

This can prove really useful if we can somehow, somewhere else in our data structure definitions, that we can guarantee ordering of our arrays. This example is left for future work as it's more abstract to just the search examples we're displaying here.

## Binary Search

With sequential search we start by evaluating the first entry of array for whether or not it matches the the item that we're looking for, and if it does not we proceed through the entire collection, trying to find a match. There are at most, at any time, $$n-1$$ more items to look at if the item we're currently evaluating is not the one we're looking for.

Binary search takes a bit of a different approach to the problem. Instead of searching through the collection, sequentially, starting with the first item in the list or array, the process starts at the middle. If the middle item of the list is _not_ the item that we're looking for, and is larger than the middle value, we can drop the entire bottom half of the list and save ourselves that much computation time.


```python
# Binary search example
def binary_search(li, item):
    first = 0
    last = (len(li) - 1)
    found = False
    
    while first <= last and not found:
        midpoint = ((first + last)//2)
        if li[midpoint] == item:
            found = True
        else:
            if item < li[midpoint]:
                last = (midpoint - 1)
            else:
                first = (midpoint + 1)
    return found

test_li = [0,2,3,4,5,8,10,15,17,21,25,32,42,45]
print(binary_search(test_li, 45))
```

    True


Using our handy table again, we can analyze the complexity of the binary search algorithm.

| Comparisons | Approximate Number of Items Left|
|-------------|---------------------------------|
| 1           | $$\frac{n}{2}$$                   |
| 2           | $$\frac{n}{4}$$                   |
| 3           | $$\frac{n}{8}$$                   |
| ...                                           |
|$$i$$          | $$\frac{n}{2^{i}}$$               |

The number of comparisons necessary to get to this point is $$i$$ where $$\frac{n}{2^{i}} = 1$$. Solving for $$i$$ is $$i = log n$$. Therefore, binary search has a computational complexity of $$O(log n)$$.

#### References

[http://interactivepython.org/courselib/static/pythonds/SortSearch/TheSequentialSearch.html](http://interactivepython.org/courselib/static/pythonds/SortSearch/TheSequentialSearch.html)

[http://www.cs.toronto.edu/~tfowler/csc263/TutorialNotes1.txt](http://www.cs.toronto.edu/~tfowler/csc263/TutorialNotes1.txt)
