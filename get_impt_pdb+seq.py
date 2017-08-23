import numpy as np
import sys
import alignment
pdb_seq_chain =  np.load('fasta_chain.npy').item()

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
        
def cheap_alignment(pdb_seq,full_seq):
    for i in range(0,len(full_seq)-len(pdb_seq)):
        if score(full_seq[i:i+len(pdb_seq)],pdb_seq):
            print full_seq[i:i+len(pdb_seq)]
            print pdb_seq,'\n',
            return pdb_seq, full_seq[i:i+len(pdb_seq)]
DATA = {}
f_align = open('align.out','w')
for ii in sorted(pdb_seq_chain):
    try:
        # get linking residues #
        f1 = open(ii.lower()+'.pdb','r')
        full_seq0 = pdb_seq_chain[ii]
        cordO3_P = {}
        for line in f1:
            if 'ATOM' in line and line[12:16].strip() == "O3'"  :
                if line[17:20].strip() in ['A','U','G','C']:
                    resNum,resiType = (int(line[22:26].strip()),line[12:16].strip())
                    X = np.array(float(line[30:38].strip()))
                    Y = np.array(float(line[38:46].strip()))
                    Z = np.array(float(line[46:54].strip()))
                    cordO3_P[(resNum,"O3'")] = np.array([X,Y,Z])
            if 'ATOM' in line and line[12:16].strip() ==  "P" :
                if line[17:20].strip() in ['A','U','G','C']:
                    resNum,resiType = (int(line[22:26].strip()),line[12:16].strip())
                    X = np.array(float(line[30:38].strip()))
                    Y = np.array(float(line[38:46].strip()))
                    Z = np.array(float(line[46:54].strip()))
                    cordO3_P[(resNum,'P')] = np.array([X,Y,Z])
        # get c4i #
        f1 = open(ii.lower()+'.pdb','r')
        full_seq0 = pdb_seq_chain[ii]
        first_resi = True
        cord = {}
        prev_resi, curr_resi = 0, 0
        temp = ''
        for line in f1:
            if 'ATOM' in line and line[12:16].strip() == "C4'" :
                if line[17:20].strip() in ['A','U','G','C']:
                    curr_resi = int(line[22:26].strip())
                    if prev_resi > 0 :
                        if sum((cordO3_P[(curr_resi,'P')] - cordO3_P[(prev_resi,"O3'")])**2)**.5 < 2 :
                            temp += line[17:20].strip()
                        else:
                            print temp
                            print full_seq0
                            cheap_alignment(temp,full_seq0)
                            temp = ''
                    prev_resi = curr_resi
                if line[17:20].strip() in ['A','U','G','C']:
                    #print line
                    resNum,resiType = (int(line[22:26].strip()),line[17:20].strip())
                    X = np.array(float(line[30:38].strip()))
                    Y = np.array(float(line[38:46].strip()))
                    Z = np.array(float(line[46:54].strip()))
                    cord[resNum] = [resiType,np.array([X,Y,Z])]
                else: print line[17:20].strip()
        Resi_map = {}
        pdb_seq0 = ''
        first, last = sorted(cord)[0], sorted(cord)[-1]
        for i in range(first,last+1):
            if i in [x for x in cord]:
                Resi_map[i] = cord[i][0]
                pdb_seq0 += cord[i][0]
            else:
                pdb_seq0 += 'X' #'X' for missing residues
        f_align.write(ii+'\n')
        temp = alignment.needle(pdb_seq0,full_seq0)
        f_align.write(temp[0]+'\n')
        f_align.write(temp[1]+'\n')
        
        temp2 = ['','']
        Resi_map_new = {}
        pdbCounter = 1
        seqCounter = 1
        for char in range(len(temp[0])):
            # find cases where pdb_seq[0] has gap (X) while
            # sequence has no gap ('-')
            if temp[0][char] == 'X' and temp[0] == '-':
                 pdbCounter += 1
            else:
                if pdbCounter in Resi_map:
                    Resi_map_new[seqCounter] = Resi_map[pdbCounter]
                    pdbCounter += 1
                    seqCounter += 1
        #Resi_map = Resi_map_new
        pdb_seq,full_seq = temp[0], temp[1]
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
        print ii#DATA[ii] = 'TYPE_ERROR' #cannot align
        die
    except IOError:
        None#DATA[ii] = 'IO_ERROR' #pdb dont exist
    except IndexError:
        None#DATA[ii] = 'index_error' #no cordinates
f_align.close()                
    
