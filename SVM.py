import pprint
def readData(filename) :
    #membuka file csv berisi dataset untuk clustering
    f = open(filename, 'r')
    # skip baris pertama
    f.readline()
    line = f.readline()
    data = []
    # Untuk setiap baris pada file, parse dan masukan data ke list lalu hapus kolom pertama
    while (line):
        arr = line.split(",")
        arr = list(map(lambda x : float(x.replace('\"', "").replace("\n", "")), arr))
        data.append(arr)
        line = f.readline()
    return data


data = readData('dataset.csv')

# untuk mencari kernel
def getKernel(x, z):
    length = len(x)
    dotProd = sum([x[i] * z[i] for i in range(length)])
    return (dotProd + 1)**2

# untuk membuat matriks kosong
kernels = [[0] * len(data) for i in range(len(data))]

# untuk mengisi matriks dengan nilai kernel
for i in range(len(kernels)):
    for j in range(len(kernels[i])):
        listi = data[i][0:len(data[0])-1]
        listj = data[j][0:len(data[0])-1]
        kernels[i][j] = getKernel(listi, listj )

# untuk menghitung nilai alpha
def calcAlpha(kernels : list, data : list, lamb):
    alpha = [0] * len(data)
    d = [[0] * len(data) for i in range(len(data))]
    c = 2 # konstanta
    treshold = 1e-10

    # untuk menghitung nilai d
    for i in range(len(d)):
        for j in range(len(d[i])) :
            
            d[i][j] = data[i][-1]*data[j][-1] *(kernels[i][j] + (lamb**2))
    
    maxD = max([max(i) for i in d])
    # set nilai gamma (0 < gamma < (2/maxD))
    gamma = (2/maxD)/2
    
    # selama iterasi masih lebih besar dari treshold maka melakukan algoritma untuk menghitung alpha
    f_old = 0
    while True:
        e = [0] * len(alpha)
        
        for i in range(len(alpha)) :
            e[i] = sum([alpha[j] * d[i][j] for j in range(len(d[i]))])
            newA = min(max(gamma*(1-e[i]), - alpha[i]), c - alpha[i])
            alpha[i] = alpha[i] + newA
        
        print(alpha)
        f = sum(alpha) / len(alpha)
        print("F", f)
        delta = abs(f - f_old)
        print("delta", delta)
        if (delta < treshold) :
            break
        f_old = f
    
    return alpha

alpha = calcAlpha(kernels, data, 2)
print(alpha)
res = 0

# main
for i in range(len(alpha)) :
    row = data[i][0:len(data[i])-1]
    temp = alpha[i] *data[i][-1]* getKernel(row, [40, 175])
    res += temp

print(res)
print(1 if res>0 else -1)