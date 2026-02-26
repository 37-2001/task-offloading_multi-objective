import numpy as np
def random_cpu_circles():
    cpu_circles=[]
    for n in range(18):
        n=np.random.uniform(100, 300)
        cpu_circles.append(n)
    return cpu_circles
cpu_circles=random_cpu_circles()
with open("cpu_circles.txt","w+") as fw:
    for i in range(len(cpu_circles)):
        fw.write("{}\n".format(cpu_circles[i]))

# data_size=np.random.uniform(400, 500)
