con = '<[1]*ACuuUuAgCGAUGGAUguCUuGGCUCccGuaUCGAUGAAgaaCGCaGCaAAa..uGCGAUAcGUaguGUGAAuuGCAGaaUuccgUgAAUCacCGAAucuucGAACGCaaaUuGCGcccccgg*[32]>'
ori = '<[0]*ACUCUUAGCGGUGGAUCACUCGGCUCGUGCGUCGAUGAAGAACGCAGCGCUAgcUGCGAGAAUUAAUGUGAAUUGCAGGAC-ACAUUGAUCAUCGACACUUCGAACGCACU-UGCGUACGCCU*[ 0]>'

def get_conseus (con,ori):
    result = ''
    for i in range(0,len(con)):
        if i.upper() in ['A','U','G','C']:
            if ori[i] == '-' or ori[i] == '.':
               None
            elif con[i] != '-' and con[i] != '.' :
                result += con[i]
            else:
                result += 'X'
        else:
            result += ori[i]
    return result
