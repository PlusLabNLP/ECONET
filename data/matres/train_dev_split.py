import random

files = [f.strip() for f in open("train_docs.txt")]

random.seed(7)
random.shuffle(files)

trn = int(len(files) * 0.8)

train = files[:trn]
dev = files[trn:]

with open("trainIds.txt", "w") as outfile:
    for txt in train:
        outfile.write(txt)
        outfile.write('\n')

outfile.close()

with open("devIds.txt", "w") as outfile:
    for txt in dev:
        outfile.write(txt)
        outfile.write('\n')
outfile.close()
