import pandas as pd
import glob
import os

files = glob.glob('C:\\Users\\rikua\\Documents\\fd_ajstd_wl_0705\\*')
for file in files :
    path, ext = os.path.splitext(file)
    filelist = path.split('\\')
    name = filelist[5]
    read_file = pd.read_excel (file)
    read_file.to_csv ("C:\\Users\\rikua\\Documents\\fd_ajstd_wl_0705_csv\\" + name + ".csv", index = None, header=True)