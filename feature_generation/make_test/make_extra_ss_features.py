import matplotlib.pyplot as plt

#for i in `ls -d fasta*pdb`;do RNAfold -p < $i > $i.dbn ;done
#for i in `ls -d *dp.ps `;do /home/leexa/Downloads/ViennaRNA-2.4.1/src/Utils/mountain.pl $i > $i.three_features ;done

x = '(((((....((....))(((((((..((.((((((...)))))).)))))))))))))).'

import  numpy as  np

def get_bp_dbn(x,result=[],debug=False):
    prev = 0
    got_bracket = False
    for i in range(len(x)):
        if x[i] == '(' :
            got_bracket = True
            prev = i
        elif x[prev] == '(' and x[i] == ')':
            got_bracket = True
            result += [[prev,i],]
            x = x[0:prev]+':'+x[prev+1:i]+':'+x[i+1:]
            if debug is True:
                print x
            return get_bp_dbn(x,result)
    if got_bracket == False:
        return result
                  
import os,sys

names = {}
dictt = {}
all_fasta = [x for x in os.listdir('.') if x.startswith('fasta') and x.endswith('.dbn')]
for ii in sorted(all_fasta):
    unimportant = ''
    f1 = open(ii,'r')
    for line in f1:
        if '>' in line or 'A' in line or 'U' in line or 'G' in line or 'C' in line:
            unimportant += line
        else:
            if line[-1] == '\n':
                line= line.split()[0]
                x = line
                
            if len(x) < 500:
                y = np.zeros((len(x),len(x)))
                name_y = np.zeros(len(x))
                if len(get_bp_dbn(x,[],False)) == 0:
                    print ii,x#,unimportant
                for i in get_bp_dbn(x,[],False):
                    y[i[0],i[1]] = 1
                    y[i[1],i[0]] = 1
                    name_y[i[0]],name_y[i[1]] = 1,-1
                dictt[ii.split('fasta_')[1].split('.pdb.dbn')[0]] = [y,name_y]
                
                if len(x) > 399:
                    None#sprint x
                    #plt.imshow(y);plt.show()
                break

np.save('../data_test_ss.npy',dictt)
dictt2 = {}
all_three_features = [x for x in os.listdir('.') if  x.endswith('_dp.ps.three_features')]
for ii in sorted(all_three_features):
    unimportant = ''
    f1 = open(ii,'r')
    dictt2[ii.split('_dp.ps.three_features')[0]] = []
    temp = []
    for line in f1:
        if len(line.split()) ==2 :
            temp += [np.float(line.split()[1]),]
        if '&' in line:
            dictt2[ii.split('_dp.ps.three_features')[0]] += [temp,]
            temp = []
    dictt2[ii.split('_dp.ps.three_features')[0]] += [temp,]
    dictt2[ii.split('_dp.ps.three_features')[0]] = np.array(dictt2[ii.split('_dp.ps.three_features')[0]])
np.save('../data_test_extra.npy',dictt2)
