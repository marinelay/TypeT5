import pickle

def load_data():
    with open('data/repos_split.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_data()
print(data)