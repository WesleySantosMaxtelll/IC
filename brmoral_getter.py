import csv


class brmoral_data:

    def __init__(self):
        posicao = {'aborto':1, 'armas':0, 'cotas':5, 'maconha':3, 'maioridade':4, 'pena':2}
        self.dados = {'aborto':[[], []], 'armas':[[], []], 'cotas':[[], []], 'maconha':[[], []], 'maioridade':[[], []], 'pena':[[], []]}
        lista = ['aborto', 'armas', 'cotas', 'maconha', 'maioridade', 'pena']
        self.dic_translate = {'neutral':0, 'for':2, 'against':1}

        with open('/home/maxtelll/Downloads/brmoral.csv', encoding='ISO-8859-1') as csv_file:
            f = csv.reader(csv_file, delimiter = ';')
            next(f, None)
            for i in f:
                linha = [g.replace("\x94", "").replace("\x93", "").replace("\x96", "") for g in i]
                for l in lista:
                    p = posicao[l]
                    self.dados[l][0].append(linha[p])
                    self.dados[l][1].append(self.dic_translate[linha[p+7]])
        # for d in self.dados:
        #     print(d)
        #     print(len(self.dados[d][0]))
        with open('/home/maxtelll/Documents/arquivos/ic_novo_backup/posicionamento/MTwitter-dev-unverified.csv', encoding='ISO-8859-1') as csv_file:
            f = csv.reader(csv_file, delimiter = ';')
            next(f, None)
            for i in f:
                # print(i)
                self.dados[i[0]][0].append(i[2])
                self.dados[i[0]][1].append(int(i[3]))
                # linha = [g.replace("\x94", "").replace("\x93", "").replace("\x96", "") for g in i]
                # for l in lista:
                #     print(l)
                #     p = posicao[l]
                #     self.dados[l][0].append(linha[p])
                #     self.dados[l][1].append(linha[p+7])

        # print(self.dados)
        # for d in self.dados:
        #     print(d)
        #     print(len(self.dados[d][0]))
    def posicionamento(self, topico):
        ts, tg = [], []
        textos, tags = self.dados[topico]
        for t, p in zip(textos, tags):
            ts.append(t)
            if p == 0:
                tg.append(0)
            else:
                tg.append(1)
        return ts, tg

    def ternaria(self, topico):
        ts, tg = [], []
        textos, tags = self.dados[topico]
        for t, p in zip(textos, tags):
            ts.append(t)
            if p == 'neutral':
                tg.append(0)
            elif p == 'for':
                tg.append(2)
            else:
                tg.append(1)
        return ts, tg


    def polaridade(self, topico):
        ts, tg = [], []
        textos, tags = self.dados[topico]
        for t, p in zip(textos, tags):
            # print(type(p))
            if p > 0:
                ts.append(t)
                tg.append(p-1)
        return ts, tg
            # print(self.dados['cotas'][0][d] + '       '+self.dados['cotas'][1][d])


brmoral_data()