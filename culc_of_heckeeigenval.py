import numpy as np
import math
import random
import numpy.linalg as LA
import pickle

def Gauss_elimination(m, show_matrix):
    num_line = len(m[0])
    num_column = len(m)
    div = lambda a, b: 0 if b==0 else a / b
    for i in range(num_column):
        for j in range(num_line):
            if m[i][j]!=0:
                for k in range(j+1, num_line):
                    waru = div(m[i][k], m[i][j])
                    for l in range(i, num_column):
                        m[l][k]=m[l][k]-m[l][j]*waru
                for k in range(i+1, num_column):
                    m[k][j]=0
        
    return m

def inversecalc(m,n):
    if n==0 or n%m == 0:
        return (1,0)
    g=math.gcd(m,n)
    for i in range(abs(n)):
        if i*m%abs(n)==g:
            i2= i
            j = int((i*m-g)/n)
            break
    #im-jn=g
    return (i2,j)

def reduce_symbol(N, firstsymbol):
    que = [firstsymbol]
    reducedsymbol = []
    roop_count = 0
    for symbol in que:
        roop_count = roop_count+1
        if roop_count > 1000:
            return 0
        symbol = np.round(symbol).astype(np.float64)
        former_symbol = np.copy(symbol)
        determinant= round(LA.det(symbol))
        if determinant<0:
            symbol=-symbol
            determinant = -determinant
        if determinant==1:
            reducedsymbol.append(symbol)
        elif determinant!=0:
            while symbol[0][1] !=0 or symbol[0][2]:
                for i in range(1,3):
                    if symbol[0][i] != 0 and(symbol[0][0] == 0 or abs(symbol[0][i]) < abs(symbol[0][0])):
                        symbol = np.insert(symbol,  0, symbol[:,i], 1)
                        symbol = np.delete(symbol, obj=i+1, axis=1)
                for i in range(1,3):
                    symbol[:,i] = symbol[:, i] - symbol[:,0] * int((symbol[0][i]- symbol[0][i]%symbol[0][0])/symbol[0][0])
                
            if symbol[0][0] > 1:
                x = np.array([1,0,0])
                m = symbol[0][0]
            else:
                while symbol[1][2]!=0:
                    if symbol[1][2] != 0 and (symbol[1][1] == 0 or abs(symbol[1][2]) < abs(symbol[1][1])):
                        symbol = np.insert(symbol,  1, symbol[:,2], 1)
                        symbol = np.delete(symbol, obj = 3,axis= 1)
                    symbol[:,2] = symbol[:, 2] - symbol[:,1] * int((symbol[1][2]- symbol[1][2]%symbol[1][1])/symbol[1][1])
                    
                if abs(symbol[1][1])>1:
                    x = np.array([-1* symbol[1][0]* symbol[0][0],1, 0])
                    m = symbol[1][1]
                else:
                    x = np.array([(symbol[2][0] * -1 + symbol[1][0] * symbol[2][1] * symbol[1][1])*symbol[0][0],symbol[2][1]  * symbol[1][1]* -1, 1])
                    m = symbol[2][2]
            v = np.zeros((1,3))
            for i in range(3):
                x[i] = (int(m/2)+x[i])%m - int(m/2)
                v += x[i]/m * former_symbol[i]
            for i in range(3):
                que.append(np.insert(np.delete(former_symbol, obj=i, axis=0),i, v,  axis =0))
    return reducedsymbol

def calc_heckeeigenval(heckerep, eigenvector, basis, eqclass, expression_matrix, representatives): 
    basisnum = len(basis)
    heckematrix = np.zeros((basisnum, basisnum))
    for i, base in enumerate(basis):
        if base[0] == 0:
            Qj = np.array([[base[1], 0,0],[base[2], 1,0], [base[0],0,1]])
        else:
            Qj = np.array([[base[0], 0,0],[base[1], 1,0], [base[2],0,1]])
        for Bs in heckerep:
            for symbol in reduce_symbol(N,Qj@Bs ):
                x = int(symbol[0][0])%N
                y = int(symbol[1][0])%N
                z = int(symbol[2][0])%N
                if (int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2])) in representatives:
                    heckematrix[i] -= eqclass[x][y][z][3] * expression_matrix[representatives[(int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2]))]][-basisnum:]
    for i in range(basisnum):
        if abs(eigenvector[i]) > 0.1:
            return (heckematrix@eigenvector)[i] / eigenvector[i]


if __name__ == "__main__":

    N=128
    Nmim = 1

    inversenum = []
    for i in range(N):
        if math.gcd(i, N) == 1:
            inversenum.append(i)

    eqclass = np.zeros((N,N,N,4), dtype=int)
    representatives = {}
    eqnum = 0

    for i in range(Nmim,N):
        for j in range(Nmim, N):
            for k in range(Nmim, N):
                if eqclass[i][j][k][3] == 0 and (math.gcd(N, i,j, k)==1):
                    representatives[(i, j, k)]=eqnum
                    eqnum = eqnum + 1
                    eqclass[i][j][k][0] = i
                    eqclass[i][j][k][1] = j
                    eqclass[i][j][k][2] = k
                    eqclass[i][j][k][3] = 1
                    

                    computable_elems=[]
                    computable_elems.append((i,j, k))
                    for (x,y,z) in computable_elems:
                        if eqclass[(-y)%N][x][z][3]==0:
                            eqclass[(-y)%N][x][z][0]= i
                            eqclass[(-y)%N][x][z][1]=j
                            eqclass[(-y)%N][x][z][2]=k
                            eqclass[(-y)%N][x][z][3]=-1 * eqclass[x][y][z][3]
                            computable_elems.append(((-y)%N,x, z))
                        if eqclass[z][x][y][3]==0:
                            eqclass[z][x][y][0]=i
                            eqclass[z][x][y][1]=j
                            eqclass[z][x][y][2]=k
                            eqclass[z][x][y][3]=eqclass[x][y][z][3]
                            computable_elems.append((z,x,y))
                        for r in inversenum:
                            eqclass[x*r%N][y*r%N][z*r%N][0] = i
                            eqclass[x*r%N][y*r%N][z*r%N][1] = j
                            eqclass[x*r%N][y*r%N][z*r%N][2] = k
                            eqclass[x*r%N][y*r%N][z*r%N][3] = eqclass[x][y][z][3]


    expression_matrix = []
    
    Is_checked = [[[0 for i in range(N)]for j in range(N)]for k in range(N)]

    for i in range( N):
        for j in range(N):
            for k in range(N):
                if Is_checked[i][j][k]==0 and math.gcd(N, i,j, k)==1:
                    x = i
                    y = j
                    z=k
                    expression_row = [0 for k in range(eqnum)]
                    if (int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2])) in representatives:
                        expression_row[representatives[(int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2]))]] += int(eqclass[x][y][z][3])
                    if (int(eqclass[(y-x)%N][(-x)%N][z][0]), int(eqclass[(y-x)%N][(-x)%N][z][1]), int(eqclass[(y-x)%N][(-x)%N][z][2]) ) in representatives:
                        expression_row[representatives[(int(eqclass[(y-x)%N][(-x)%N][z][0]), int(eqclass[(y-x)%N][(-x)%N][z][1]), int(eqclass[(y-x)%N][(-x)%N][z][2])) ]] += int(eqclass[(y-x)%N][(-x)%N][z][3])
                    if (int(eqclass[(-y)%N][(x-y)%N][z][0]), int(eqclass[y][(x-y)%N][z][1]),int(eqclass[y][(x-y)%N][z][2])) in representatives:
                        expression_row[representatives[int(eqclass[(-y)%N][(x-y)%N][z][0]), int(eqclass[y][(x-y)%N][z][1]),int(eqclass[y][(x-y)%N][z][2])]] += int(eqclass[(-y)%N][(x-y)%N][z][3])
                    for r in inversenum:
                        Is_checked[i*r%N][j*r%N][k*r%N]=1
                        Is_checked[(y-x)*r%N][(-x)*r%N][k*r%N]=1
                        Is_checked[(-y)*r%N][(x-y)*r%N][k*r%N]=1

                    expression_matrix.append(expression_row)


    therank = np.linalg.matrix_rank(np.array(expression_matrix))
    basis = []
    basis_of_128  =basis
    basisnum = len(basis_of_128)

    Isbasis = [False for i in range(len(expression_matrix[0]))]
    for base in basis_of_128:
        Isbasis[representatives[base]] = True
    expression_matrix = [tuple(i) for i in expression_matrix]
    expression_matrix = list(set(expression_matrix))
    expression_matrix = [list(i) for i in expression_matrix]
    expression_matrix = np.array(expression_matrix, dtype="float32")
    
    counting = 0
    for i in range(expression_matrix.shape[1]-len(basis_of_128)):
        for j in range(i,expression_matrix.shape[0]):
            if abs(expression_matrix[j][i]) >0.01:
                counting += 1
                a = expression_matrix[j][i]
                nowrow = np.copy(expression_matrix[j]) / a
                expression_matrix=np.delete(expression_matrix,j, axis=0 )
                expression_matrix=np.insert(expression_matrix, i, nowrow, axis=0)
                for k in range(i+1, expression_matrix.shape[0]):
                    expression_matrix[k] = np.copy(expression_matrix[k] - nowrow * expression_matrix[k][i])
                break


    basisnum = len(basis_of_128)
    for i in range(expression_matrix.shape[1]-basisnum):
        for j in range(expression_matrix.shape[1]-basisnum-i-1):
            expression_matrix[j] = expression_matrix[j] - expression_matrix[expression_matrix.shape[1]-basisnum-i-1] * expression_matrix[j][expression_matrix.shape[1]-basisnum-i-1]
    
    basisplace = []
    for i,row in enumerate(expression_matrix[:len(representatives)]):
        if round(row[i]) != 1:
            basisplace.append(i)
            basis_of_128.append(list(representatives)[i])
            expression_matrix[i][i] = -1
    nowcolumn = np.copy(expression_matrix[:, basisplace])
    expression_matrix=np.delete(expression_matrix, basisplace, axis = 1)
    expression_matrix=np.hstack((expression_matrix, nowcolumn))
    basisnum = len(basisplace)

    for row in expression_matrix[:len(representatives)]:
        i=1
                                 

    Is_checked = [[[0 for i in range(N)]for j in range(N)]for k in range(N)]
    for i in range(Nmim, N):
        for j in range(Nmim,N):
            for k in range(Nmim,N):
                if Is_checked[i][j][k]==0 and math.gcd(N, i,j, k)==1:
                    x = i
                    y = j
                    z=k
                    expression_row = np.zeros((1, basisnum))
                    if (int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2])) in representatives:
                        expression_row += eqclass[x][y][z][3]* expression_matrix[representatives[(int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2]))]][-basisnum:]
                    if (int(eqclass[(y-x)%N][(-x)%N][z][0]), int(eqclass[(y-x)%N][(-x)%N][z][1]), int(eqclass[(y-x)%N][(-x)%N][z][2]) ) in representatives:
                        expression_row+= eqclass[(y-x)%N][(-x)%N][z][3]* expression_matrix[representatives[(int(eqclass[(y-x)%N][(-x)%N][z][0]), int(eqclass[(y-x)%N][(-x)%N][z][1]), int(eqclass[(y-x)%N][(-x)%N][z][2]) )]][-basisnum:]
                    if (int(eqclass[(-y)%N][(x-y)%N][z][0]), int(eqclass[y][(x-y)%N][z][1]),int(eqclass[y][(x-y)%N][z][2])) in representatives:
                        expression_row+= eqclass[(-y)%N][(x-y)%N][z][3]* expression_matrix[representatives[(int(eqclass[(-y)%N][(x-y)%N][z][0]), int(eqclass[y][(x-y)%N][z][1]),int(eqclass[y][(x-y)%N][z][2]))]][-basisnum:]
                    for r in inversenum:
                        Is_checked[i*r%N][j*r%N][k*r%N]=1
                        Is_checked[(y-x)*r%N][(-x)*r%N][k*r%N]=1
                        Is_checked[(-y)*r%N][(x-y)*r%N][k*r%N]=1
                    for elements in expression_row:
                        if abs(elements[0]) > 0.001:
                            break
                        
                   
                   
    p=3
    heckerep = []
    heckematrix = np.zeros((basisnum, basisnum))
    for i in range(p):
        for j in range(p):
            heckerep.append(np.array([[1,0,0],[0,1,0],[i*N, j, p]]))
        heckerep.append(np.array([[1,0,0],[i*N, p,0],[0,0,1]]))
    heckerep.append(np.array([[p,0,0],[0,1,0],[0,0,1]]))
    for i, base in enumerate(basis_of_128):
        if base[0] == 0:
            Qj = np.array([[base[1], 0,0],[base[2], 1,0], [base[0],0,1]])
        else:
            Qj = np.array([[base[0], 0,0],[base[1], 1,0], [base[2],0,1]])
        for Bs in heckerep:
            for symbol in reduce_symbol(N,Qj@Bs ):
                x = int(symbol[0][0])%N
                y = int(symbol[1][0])%N
                z = int(symbol[2][0])%N
                if (int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2])) in representatives:
                    heckematrix[i] -= eqclass[x][y][z][3] * expression_matrix[representatives[(int(eqclass[x][y][z][0]), int(eqclass[x][y][z][1]), int(eqclass[x][y][z][2]))]][-basisnum:]


    np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

    theeigenvector = np.copy(LA.eig(heckematrix).eigenvectors[:,4])
    x=1
    print(calc_heckeeigenval(heckerep, theeigenvector, basis_of_128, eqclass))


    heckerep = []
    for i in range(2):
        for j in range(2):
            heckerep.append(np.array([[2,0,i],[0,2,j],[0, 0, 1]]))
        heckerep.append(np.array([[2,i,0],[0, 1,0],[0,0,2]]))

    
    print(calc_heckeeigenval(heckerep, theeigenvector, basis, eqclass, expression_matrix, representatives))

    heckerep = [np.array([[1,0,0],[64,1,0],[0,0,1]]), np.array([[1,0,0],[64,1,0],[64,0,1]]), np.array([[1,0,0],[0,1,0],[64,0,1]])]
    print(calc_heckeeigenval(heckerep, theeigenvector, basis, eqclass, expression_matrix, representatives))


    mod4mats = []
    for i in range(-1,3):
        for j in range(-1,3):
            for k in range(-1,3):
                for l in range(-1,3):
                    A = np.array([[i,j],[k,l]], dtype='int64')
                    if int(i*l-j*k)%4 == 1:
                        mod4mats.append(A)
    heckerep = []
    for A in mod4mats:
        detA=1
        heckerep.append(np.array([[4+32*LA.det(A)*detA,A[1,1]*detA ,-A[0,1]*detA] ,[128* A[0,0],4,0],[128*A[1,0],0,4]]))
    
    print(calc_heckeeigenval(heckerep, theeigenvector, basis, eqclass, expression_matrix, representatives))

    heckerep = []
    for i in range(128):
        heckerep.append(np.array([[0,1,0] ,[-128,0,i],[0,0,1]]))
        if i%2==0:
            heckerep.append(np.array([[0,1,0] ,[0,0,1],[-128,0,i]]))

    print(calc_heckeeigenval(heckerep, theeigenvector, basis, eqclass, expression_matrix, representatives))


