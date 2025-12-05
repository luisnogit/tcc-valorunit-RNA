import csv
from random import shuffle,seed

genlist = []

with open("./dnn/dataset.csv", "r") as f:
	reader = csv.reader(f)
	i = 0
	for line in reader:
		if i > 0:
			genlist.append(line)
		else:
			header = line
		i+=1

f.close()

seed()
for i in range(4):
	shuffle(genlist)
size = len(genlist)
i = 0

markings = open('./dnn/test.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

j = 0
while j < int(0.2*size):
	writer.writerow(genlist[i])
	i+=1
	j+=1

markings.close()
print('Test done!')

markings = open('./dnn/validation.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

j = 0
while j < int(0.2*size):
	writer.writerow(genlist[i])
	i+=1
	j+=1

markings.close()
print('Validation done!')

markings = open('./dnn/train.csv','w', encoding="utf-8")
writer = csv.writer(markings)
writer.writerow(header)

while i < size:
	if (float(genlist[i][-1]) == 1):
		for j in range(6):
			writer.writerow(genlist[i])
	else:
		writer.writerow(genlist[i])
	i+=1

markings.close()
print('Train done!')