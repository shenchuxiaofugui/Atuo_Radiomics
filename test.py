from copy import deepcopy
b = ["label", "feature"]
a = deepcopy(b)
b.insert(0, 1)

print(a)