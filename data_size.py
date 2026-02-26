import numpy as np
def random_datasize():
    data_size=[]
    for n in range(18):
        n=np.random.uniform(200, 500)
        data_size.append(n)
    return data_size
data_size=random_datasize()
with open("data_size.txt","w+") as fw:
    for i in range(len(data_size)):
        fw.write("{}\n".format(data_size[i]))

# data_size=np.random.uniform(400, 500)
