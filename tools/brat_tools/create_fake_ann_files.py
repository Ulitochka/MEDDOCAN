import os


init_data_path = '/home/m.domrachev/repos/competitions/MEDDOCAN/data/init_data/test/'

for f in os.listdir(init_data_path):
    with open(os.path.join(init_data_path, f.split('.')[0] + '.ann'), 'w') as outf:
        pass
    outf.close()
