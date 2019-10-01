
import csv

def get_files(num_samples):
    path = '/home/maxtelll/Documents/arquivos/base-artigos/articles.csv'
    resp = []

    def len_first(elemento):
        return len(elemento[0])

    with open(path, encoding='utf-8') as csv_file:
        f = csv.reader(csv_file, delimiter=',')
        next(f, None)
        # textos, titulos = [], []
        for i in f:
            if len(i[0]) > 20 and len(i[1]) > 700:
                resp.append([i[1], i[0]])
        print(len(resp))
        return sorted(resp, key=len_first)[:num_samples+1]
        # return resp[:num_samples+1]
        # return [textos[t] for t in range(num_samples)], [titulos[t] for t in range(num_samples)]

# print(get_files(40))