from rouge import Rouge 
import ast


path = '/home/maxtelll/PycharmProjects/text_generator/gerador_words-20-2.txt'

f = open(path, 'r+').read()
for i in ast.literal_eval(f):
	print(i)

hypothesis, reference = [], []

for i in f:
	# print(i.split('\n')[2:4])
	try:
		at, gr = i.split('\n')[2:4]
	except ValueError:
		print(i)
	# print(at)
	# print(gr)

	at, gr = at.split(":")[1], gr.split(":")[1]
	# print(at.strip())
	# print(gr.strip())
	hypothesis.append(gr.strip())
	reference.append(at.strip())

# hypothesis = ["this page includes the", "show transcript use the transcript", " to help students with reading"]

# reference = ["this page includes the", "show transcript use the transcript", " to help students with reading"]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference, avg=True)

print(scores)