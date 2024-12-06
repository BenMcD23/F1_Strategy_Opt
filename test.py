import pickle


with open(r"cache\2019\2019-03-17_Australian_Grand_Prix\2019-03-15_Practice_1\_extended_timing_data.ff1pkl", 'rb') as f:
    data = pickle.load(f)
    print(data)