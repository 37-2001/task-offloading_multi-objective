import numpy as np
def random_t():
    random_t=[]
    for n in range(20):
        n=np.random.uniform(0, 0)
        random_t.append(n)
    return random_t
t=random_t()
with open("possion.txt","w+") as fw:
    for i in range(len(t)):
        fw.write("{}\n".format(t[i]))
