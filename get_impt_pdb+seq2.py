import numpy as np
import sys
sys.path.append('/home/leexa/pymol/RNA/ENTIRE_RNA_ONLY/')
import alignment
pdb_seq_chain =  np.load('fasta_chain.npy').item()
from get_impt_pdb_new_resln_module import get_clusters

result2_resln = get_clusters()[0]

def score(a,b):
    a = a.upper()
    b = b.upper()
    assert len(a) == len(b)
    for i in range(0,len(a)):
        if a[i] != 'X' and b[i] != 'X':
            if a[i] != b[i] :
                return False
    else:
        return True
        
def cheap_alignment(pdb_seq0,full_seq0):
    for i in range(0,len(full_seq0)-len(pdb_seq0)+1):
        if score(full_seq0[i:i+len(pdb_seq0)],pdb_seq0):
            #print full_seq0[i:i+len(pdb_seq0)]
            #print pdb_seq0,'\n',
            return pdb_seq0, full_seq0[i:i+len(pdb_seq0)]
        return '',full_seq0
DATA = {}
for ii in sorted(pdb_seq_chain):
    try:
        # get linking residues #
        f1 = open(ii.lower()+'.pdb','r')
        full_seq0 = pdb_seq_chain[ii]
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
                else: print line[17:20].strip()
        Resi_map = {}
        pdb_seq0 = ''
        for i in range(1,1+max(cord.keys())):
            if i in cord.keys():
                pdb_seq0 += cord[i][0]
            else:
                pdb_seq0 += 'X'
        #if cheap_alignment(pdb_seq0,full_seq0):
        if ii in ['4ADX_9','4ADX_8','4V8T_1']:
            None#no atoms
        if ii.upper() == '1B23_R':
            pdb_seq = 'GGCGCGUXAACAAAGCXGGXXAUGUAGCGGAXUGCAXAXCCGUCUAGUCCGGXXCGACUCCGGAACGCGCCUCCA'
            full_seq = full_seq0
        elif ii.upper() == '3JAH_2':
            full_seq = 'GUCUCCGUAGUGUAGCUGGUAUCACGUUCGCCUAACACGCGAAAGGUCCUCGGUUCGAAACCGGGCGGAAACACCA'
            pdb_seq = pdb_seq0
        else:
            pdb_seq,full_seq = cheap_alignment(pdb_seq0,full_seq0)
        matrix = np.zeros((len(pdb_seq),len(pdb_seq)))
        for i in range(1,1+len(pdb_seq)):
            if i in cord.keys():
                i_cord = cord[i][1]
                for j in range(i+1,1+len(pdb_seq)):
                    if j in cord.keys():
                        j_cord = cord[j][1]
                        dist = sum((i_cord - j_cord)**2)**.5
                        matrix[i-1,j-1] = dist
                        matrix[j-1,i-1] = dist
        DATA[ii] = [pdb_seq,full_seq,matrix]
    except TypeError:
        print ii, 'TYPE_ERROR' #cannot align
        print pdb_seq0
        print full_seq0
    except IOError:
        None#print ii#None#DATA[ii] = 'IO_ERROR' #pdb dont exist
    except ValueError:
        None#DATA[ii] = 'index_error' #no cordinates
        print ii
        print pdb_seq0
        print full_seq0
               
    
