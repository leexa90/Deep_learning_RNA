f1 = open('RNA.cmscan','r')
f2 = open('RNA.fa','r')
RNA_dictt = {}
for line in f2:
    if '>' in line:
        RNA_dictt[line[1:-1]] = ''
        temp = line[1:-1]
    else:
        RNA_dictt[temp] = [line[0:-1]]
RNA_dictt[temp] = [line[0:-1]]
dictt_all = {}       
temp = []
name = ''
for line in f1:
    if 'Query:' in line:
        if name == '':
            temp = []
        prev_name = name
        name = line.split()[1]
    #print line
    if 'Hit alignments:' in line:
        if prev_name != '':
            RNA_dictt[prev_name] += [temp,]
        temp = []
    else:
        temp += [line,]
RNA_dictt[name] += [temp,]
final= {}

def num_white_space(string):
    return [num_white_space_start(string),num_white_space_end(string)]

def num_white_space_start(string):
    num = 0
    for i in string:
        if i == ' ':
            num += 1 
        else: return num
def num_white_space_end(string):
    num = 0
    A = False
    for i in range(1,len(string)+1):
        if string[-i] == ' ' and A == False:
            A = True
        elif A == True and string[-i] == ' ': 
            A= True
        elif A== True and string[-i] != ' ' :
            return len(string)-i+1 
def get_conseus (con,ori):
    assert len(con) == len(ori)
    result = ''
    for i in range(0,len(con)):
        if ori[i].upper() in ['A','U','G','C','.','-','N']: # 5jte_ay has N residue conserved
            if ori[i] == '-' or ori[i] == '.':
               None
            elif con[i] != '-' and con[i] != '.' :
                result += con[i]
            else:
                result += 'X'
        else:
            result += ori[i]
    return result
def final_seq(start,get_conseus,end,full_length,unwanted = ['*','-','.','[',']','>','<']):
    result = ''
    result_ss = ''
    for i in range(0,start-1):
        result += 'X'
    result += get_impt_braket_bw(get_conseus,unwanted = unwanted)
    for i in range(end+1,full_length+1):
        result += 'X'
    return result
def remove_star_confidence(string):
    result = ''
    precceding_is_bracket = False 
    for i in string:
        if i ==']' :
            result += i
            precceding_is_bracket = True
        elif len(result) >0 and result[-1] == ']' and i == '*' and precceding_is_bracket is True:
            precceding_is_bracket = False #this takes care of these cases ]****** or else will entirely disapepar
        elif i == '[' and len(result) >0 and  result[-1] == '*':
            result = result[0:-1]
            result += i
            precceding_is_bracket =  False
        else:
            result += i
            precceding_is_bracket = False
    return result
        
    
def get_impt_braket_bw(string,unwanted ):
    impt = ''
    result = ''
    A = False
    for id in range(len(string)):
        i = string[id]
        if i =='[':            
            A= True
        elif i == ']':
            A = False
            result += int(impt)* 'X'
            impt = ''
        elif A is True :
            impt += i
        else:
            if i not in unwanted:
                result += i
    return result

dictt_result = {}

import re
for i in RNA_dictt:
    #print len(RNA_dictt[i])
    if len(RNA_dictt[i])!=2:
        print i
    else:
        dictt_result[i]  = []
        if 'assing' not in RNA_dictt[i][1][9]: # ensure an alignment is found
            print i
            num = num_white_space(RNA_dictt[i][1][6])
            start = int(RNA_dictt[i][1][9].split()[1])
            end = int(RNA_dictt[i][1][9].split()[-1])
            #print  get_impt_braket_bw(RNA_dictt[i][1][7])
            string_ss = RNA_dictt[i][1][6][num[0]:num[1]]
            confidence = RNA_dictt[i][1][10][num[0]:num[1]]
            con = RNA_dictt[i][1][7][num[0]:num[1]]
            ori = RNA_dictt[i][1][9][num[0]:num[1]]
            #print con
            #print ori
            #print get_conseus (con,ori)
            #print start,get_conseus (con,ori),end
            
            x0 =  RNA_dictt[i][0]
            x1 = final_seq(start,get_conseus(con,ori),end,len(RNA_dictt[i][0])) #full sequence
            remove_star = remove_star_confidence(get_conseus(confidence,ori))
            x2 = final_seq(start,remove_star,end,len(RNA_dictt[i][0]),
                            unwanted = ['-','.','[',']','>','<']) # cannot distingush between * from resi confidence and that of large gaps
            if i == '5tpy_a': #hard coded, alignment file had bug whereby blank space is taken as one residue. 
                x1 = 'XXGUCAGGCCAGCAAAaGCuGCcACXXXXXXXgGUaGACGGUGCUGCCUGCGuCXXXXXXXXXXXXXXXXX'
                x2 = 'XX*********************98XXXXXXX88888899**************XXXXXXXXXXXXXXXXX'
            dictt_result[i]  = [x0,x1,x2]
            print x0
            print x1
            print x2
            print 
            #print final_seq(start,get_conseus(con,ori),end,len(RNA_dictt[i][0]),string_ss)[1]
            #print final_seq(start,get_conseus(ori,confidence),end,len(RNA_dictt[i][0]),confidence)[1],'\n'
            for j in RNA_dictt[i][1][6:11]:
                None#print get_impt_braket_bw(j[num[0]:num[1]])
    ##        print RNA_dictt[i][1][6].split()[0]
    ##        print RNA_dictt[i][1][7].split()[2]
    ##        print RNA_dictt[i][1][9].split()[2]
    ##        print RNA_dictt[i][1][10].split()[0]
                #re.findall('\[.*\]',x)

        else :
            dictt_result[i]  = [RNA_dictt[i][0],RNA_dictt[i][0],'0'*len(RNA_dictt[i][0])]
            #print dictt_result[i]
import numpy as np
np.save('data_484_MSA.npy',dictt_result)
