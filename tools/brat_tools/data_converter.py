import os




init_data_path = '/home/m.domrachev/repos/competitions/MEDDOCAN/data/'


files = []
for f in os.listdir(init_data_path):
    if f.endswith('.txt'):
        files.append([os.path.join(init_data_path, f), f.split('.')[0]])

print(files)