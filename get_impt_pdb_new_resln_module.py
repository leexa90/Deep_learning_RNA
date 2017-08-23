def get_clusters():
    result,result2,result3 =[],[],[]
    for i in open('out','r'):
        #prfasta_chain.npyint i
        if 'ZZZ' in  i:
            if i[-1] == '\n':
                i = i[0:-1]
            result += [ int(i.split('ZZZ')[1].split('(')[0]),]
            if result[-1] > 35:
                if result[-1] < 1000:
                    #print i.split('|')[0][-4:],i.split('Chain(s): ')[1].split(';')[0].strip()
                    result2 += [i.split('ZZZ')[1].split(')')[1].split(','),]

    temp = {}
    for i in result2:
        for j in i:
            pdb, chain = j.strip().split('|')[0].lower(),j.strip().split('|')[2]
            if j[1:5] not in temp:
                temp[j[1:5]] = [pdb+'_'+chain.lower(),]
            else:
                temp[j[1:5]] += [pdb+'_'+chain.lower(),]

    import pandas as pd
    rsln_data = pd.read_csv('resln.csv')
    pdb_map_rsn = {}
    for i in range(len(rsln_data)):
        if 'NMR' in rsln_data.iloc[i]['Exp. Method']:
            temp2 = rsln_data.set_value(i,'Resolution',20)
        pdb_map_rsn[rsln_data.iloc[i]['PDB ID']] = (round(rsln_data.iloc[i]['Resolution'],1),\
                                                    rsln_data.iloc[i]['Exp. Method'])
    result2_resln = []
    for group in result2:
        temp2 = []
        for j in group:
            pdb = j.strip().split('|')[0]
            temp2 +=  [pdb_map_rsn[pdb.upper()] + (j,),]
        result2_resln += [temp2,]
    for i in result2_resln:
        None#print sorted(i)[0:5]
    return result2_resln,result,result2,result3,temp
#result2_resln,result,result2,result3,temp = get_clusters()
import numpy as np
def get_pdb_seq(ii):
    f1 = open(ii.lower()+'.pdb','r')
    #full_seq0 = pdb_seq_chain[ii]
    # get c4i #
    first_resi = True
    cord = {}
    prev_resi, curr_resi = 0, 0
    temp = ''
    for line in f1:
        if 'ATOM' in line and line[12:16].strip() == "P" :
            #print line
            if line[17:20].strip() in ['A','U','G','C']:
                if first_resi is True :
                    first_resi = int(line[22:26].strip()) -1
                #print line
                resNum,resiType = (int(line[22:26].strip())- first_resi,line[17:20].strip())
                X = np.array(float(line[30:38].strip()))
                Y = np.array(float(line[38:46].strip()))
                Z = np.array(float(line[46:54].strip()))
                cord[resNum] = [resiType,np.array([X,Y,Z])]
            else: None#print line[17:20].strip()
    Resi_map = {}
    pdb_seq0 = ''
    for i in range(1,1+max(cord.keys())):
        if i in cord.keys():
            pdb_seq0 += cord[i][0]
        else:
            pdb_seq0 += 'X'
    return pdb_seq0
pdb_seq_chain =  np.load('fasta_chain.npy').item()
def get_cif( ii ='1b23_r'):
    chain = ii[5:].lower()
    #print chain
    f1 = open('../cif/'+ii.lower()[0:4]+'.cif','r')
    full_seq0 = pdb_seq_chain[ii]
    # get c4i #
    first_resi = True
    cord = {}
    prev_resi, curr_resi = 0, 0
    temp = ''
    for line in f1:
        if 'ATOM' in line and len(line.split())>=19 :#'"C1\'"' \
                if line.split()[5] in ['A','U','G','C'] and line.split()[-2] == 'P'\
                   and line.split()[-3].upper() == chain.upper():
                    #print line.split()
                    if first_resi is True :
                        first_resi = int(line.split()[8]) -1
                    #print line
                    resNum,resiType = (int(line.split()[8])- first_resi,line.split()[5])
                    X = np.array(float(line.split()[10]))
                    Y = np.array(float(line.split()[11]))
                    Z = np.array(float(line.split()[12]))
                    cord[(line.split()[-3].lower(),resNum)] = [resiType,np.array([X,Y,Z])]
    Resi_map = {}
    pdb_seq0 = ''
    chain_resi = []
    cord_chain ={}
    for i in sorted(cord.keys()):
        if i[0] == chain:
            chain_resi += [i,]
            cord_chain[i] = cord[i]
    for i in range(1,1+max(chain_resi)[1]):
        if (chain,i) in cord.keys():
            pdb_seq0 += cord[(chain,i)][0]
        else:
            pdb_seq0 += 'X'
    return pdb_seq0, full_seq0, cord_chain
get_cif('2zjr_y')
