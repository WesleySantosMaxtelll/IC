import csv


class brmoral_genero_data:

    def __init__(self):
        # t.gay - marriage;
        # t.gun - control;
        # t.abortion;
        # t.death - penalty;
        # t.drugs;
        # t.criminal - age;
        # t.racial - quotas;
        # t.church - tax;
        # concat;
        # gender;
        # st.gun - control;
        # st.abortion;
        # st.death - penalty;
        # st.drugs;
        # st.criminal - age;
        # st.racial - quotas
        posicao = {'casamento_gay':0, 'armas':1, 'aborto':2, 'pena':3, 'maconha':4, 'maioridade':5, 'cotas':6, 'igreja':7, 'all':8}
        self.dados = {'casamento_gay':[[], []], 'igreja':[[], []], 'aborto':[[], []], 'armas':[[], []], 'cotas':[[], []],
                      'maconha':[[], []], 'maioridade':[[], []], 'pena':[[], []], 'all':[[], []]}
        # lista = ['aborto', 'armas', 'cotas', 'maconha', 'maioridade', 'pena']
        self.dic_translate = {'m':0, 'f':1}
        self.dic_gender = {'m':[[],[],[],[],[],[]], 'f':[[],[],[],[],[],[]]}

        # st.gun-control;st.abortion;st.death-penalty;st.drugs;st.criminal-age;st.racial-quotas
        with open('/home/maxtelll/Documents/arquivos/ic_novo_backup/dados/brmoral.csv', encoding='ISO-8859-1') as csv_file:
            f = csv.reader(csv_file, delimiter = ';')
            next(f, None)
            for i in f:
                # print(i)
                linha = [g.replace("\x94", "").replace("\x93", "").replace("\x96", "").replace("\xa0","") for g in i]
                # print(linha)
                # exit()
                for l in posicao:
                    p = posicao[l]
                    self.dados[l][0].append(linha[p])
                    self.dados[l][1].append(self.dic_translate[linha[9]])

                for i in range(6):
                    self.dic_gender[linha[9]][i].append(linha[10+i])

        # print(self.dic_gender)
        from collections import Counter
        for i in self.dic_gender:
            print(i)
            for j in self.dic_gender[i]:
                c =Counter(j)
                for h in c:
                    print('{} {}'.format(h, c[h]/len(j)))
                print('\n')
            print('\n\n')
        # for d in self.dados:
            # print(self.dados[d])
            # print(len(self.dados[d][0]))
    def get_data(self, topico):
        textos, tags = self.dados[topico]
        return textos, tags

brmoral_genero_data()