import csv


class Getter:
    
    def __init__(self):
        path = '/home/maxtelll/Documents/arquivos/ic_novo_backup/posicionamento/MTwitter-dev-unverified.csv'

        with open(path, encoding='latin-1') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            self.data = []
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    self.data.append(row)
                    line_count += 1

    def get_label(self, label):
        X, Y = [], []
        for row in self.data:
            if row[0] == label:
                X.append(row[2])
                Y.append(row[3])
        return X, Y

X, Y = Getter().get_label('cotas')
# print(X)
# print(len(Y))