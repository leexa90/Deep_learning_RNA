import sys,os
sys.path.append('/home/leexa/pymol/RNA/ENTIRE_RNA_ONLY/')
import numpy as np
import alignment
import matplotlib.pyplot as plt
models = [x for x in os.listdir('.') if ('solution' not in x and 'pdb' in x)]
dictt_model = {}
for i in models[0:1]:
    dictt_model[i] = []
    f1= open(i,'r')
    seq = ''
    temp_model = {}
    for line in f1:
        if line[17:20].strip() in ['A','U','G','C']:
            if line[12:16].strip() == 'P':
                seq  += line[17:20].strip()

solution = [x for x in os.listdir('.') if ('solution' in x and 'pdb' in x)]
dictt_sol = {}

def get_mat(i):
    dictt_sol[i] = []
    f1= open(i,'r')
    seq = ''
    temp_sol = {}
    for line in f1:
        if line[17:20].strip() in ['A','U','G','C']:
            if line[12:16].strip() == 'P':
                seq  += line[17:20].strip()
                cord1 = np.float(line[30:38].strip())
                cord2 = np.float(line[38:46].strip())
                cord3 = np.float(line[46:54].strip())
                temp_sol [(int(line[22:26].strip()),line[17:20].strip())]=\
                     np.array([cord1,cord2,cord3])
    length = sorted(temp_sol)[-1][0] 
    first_resi =  1
    mat = np.zeros((length,length))
    for i in range(0,len(sorted(temp_sol))):
        key_i = sorted(temp_sol)[i]
        for j in range(i+1,len(sorted(temp_sol))):
            key_j = sorted(temp_sol)[j]
            dist = (temp_sol[key_i] - temp_sol[key_j])**2
            dist = np.sum(dist)**.5
            mat[i,j] = dist
            mat[j,i] = dist
            
    data1 = mat
    a,b,c = (data1 < 8)*1,(data1 <= 15) & (data1 >= 8)*1,(data1 > 15)*1
    temp_resi_map = np.stack((a,b,c),axis=2)
    print seq
    #plt.imshow(temp_resi_map.astype(np.float32));plt.show()
    return temp_resi_map.astype(np.float32)   
answer = get_mat(i)            
for i in models:
    mat_model = get_mat(i)
    score = []
    for i in range(0,answer.shape[0]):
        for j in range(i+1,answer.shape[0]):
            if np.sum(answer[i,j] == mat_model[i,j])==3:
                score += [1,]
            else:
                score += [0,]
    print i,np.mean(score)
