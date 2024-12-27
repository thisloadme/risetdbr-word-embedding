dataset = []
with open('dataset.txt', 'r') as f:
    dataset = f.read().split('\n\n')

dataset = list(set([artikel for artikel in dataset if len(artikel.split('\n')) > 1 and len(artikel.split('\n')[0]) > 0]))

def replace_junk(textdata):
    return textdata.replace('HR. ', 'HR ').replace('QS. ', 'QS ').replace('no. ', 'no ').replace(') ', ')').replace(')', ') ').replace(' .', '.').replace(',.', '')

corpus = [replace_junk(artikel.split('\n')[1]) for artikel in dataset]

print(corpus[0])