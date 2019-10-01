import os
from collections import Counter

path = '/home/maxtelll/Desktop/ic_novo_backup/posicionamento/analise_palavras_selecao/'
files = os.listdir(path)
# print(files)
saida = open('/home/maxtelll/Desktop/ic_novo_backup/posicionamento/saida.csv', 'w+')
l = list(range(1,26))
saida.write('QTD,'+','.join(map(str, l)))
saida.write('\n')
for arq in files:
    print(arq)
    saida.write(arq+',')
    f = open(path+arq, 'r').read()
    f = f.split('\n')
    print(len(f))
    # print([i for i in f if len(i) <2])
    f = [len(i) for i in f]
    c = Counter(f)
    print(c)
    print(c[6])
    print(len(c))
    for i in range(1, 26):
        saida.write(str(c[i])+',')
    saida.write('\n')
saida.close()