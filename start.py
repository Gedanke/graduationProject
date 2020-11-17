import os

result = os.popen('pip list')  # 执行系统命令，返回值为result
res = result.read()
data = res.split()
demo = []
length = len(data)
i = 4
while (i < length):
    demo.append(data[i])
    i = i + 2

for package in demo:
    os.system('python3 -m pip install --upgrade ' + package)

print("all update")
