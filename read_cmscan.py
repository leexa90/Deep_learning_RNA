f1 = open('RNA.cmscan','r')
temp = []
for line in f1:
    print line
    if 'Hit alignments:' in line:
        print temp
        temp = []
    else:
        temp += [line,]
