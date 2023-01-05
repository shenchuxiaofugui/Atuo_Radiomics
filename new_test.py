
#进程调用先循环开始再循环结束再循环拿返回值
#进程可以循环调用

a = {}
a["1.0"] = {"1":1}
a["0.0"] = {"0":0}
b = {}
for key, values in a.items():
    b[key.replace(".0", "")] = values
a = b
print("hahaha")