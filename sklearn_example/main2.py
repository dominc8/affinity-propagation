import numpy as np

def similarity(p1, p2):
    return -((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2 + (p1[4] - p2[4])**2)

def main():
    X = np.array([[3, 4, 3, 2, 1], [4, 3, 5, 1, 1], [3, 5, 3, 3, 3], [2, 1, 3, 3, 2], [1, 1, 3, 2, 3]])
    data_size = len(X)
    print("X: ", X)
    print("similarity(", X[0], ", ", X[1], ") = ", similarity(X[0], X[1]))
    r = np.zeros([data_size, data_size])
    a = np.zeros([data_size, data_size])
    s = np.zeros([data_size, data_size])


    for i in range(0, data_size):
        for k in range(0, data_size):
            s[i][k] = similarity(X[i], X[k]) 
    np.fill_diagonal(s, np.median(s))

    print("s: ", s)
    c = 0

    for iter in range (20):
        
        print("iteration ", iter)
        
        for i in range(0, data_size):
            for k in range(0, data_size):
                sai = s[i] + a[i]
                sai[k] = np.amin(sai)
                r[i][k] = s[i][k] - np.amax(sai)

        print("r: ", r)

        
        for k in range(0, data_size):
            temp = np.maximum(r[:,k], np.zeros(data_size))
            temp[k] = 0
            a[k][k] = np.sum(temp)
    
        for i in range(0, data_size):
            for k in range(0, data_size):
                if k == i:
                    continue
                max_rk = np.maximum(r[:,k], np.zeros(data_size))
                max_rk[i] = 0
                max_rk[k] = 0
                a[i][k] = min(0, np.sum(max_rk) + r[k][k])
        print("a: ", a);
        c = a + r
        print("c: ", c)

    max_rows = np.amax(c, axis=1)
    print("max_rows: ", max_rows)
    print("diag: ", np.diagonal(c))

if __name__ == "__main__":
    main()
