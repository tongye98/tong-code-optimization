## Step1
编译 compile.ipynb

## Step2
用 gem5_benchmark.py 测试
1. 能否用gem5标上时间测度；
2. 用测试运行测试能否得到正确答案；

其中，train和val只需要用三个测试用例即可，因为只需要进行相对的排序；
test需要用所有的测试用例过一遍，因为最后要算平均的speedup.

同时要删除test中部分测试用例，因为有些测试用例会超时。

原文中：
training set: 82.5 test cases
validation set: 75 test cases
test set: 104 test cases

## Step3
只需要考虑能得到正确答案的二进制程序；并且修正后的测试用例集


