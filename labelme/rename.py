# 为数据集重命名 -- 用数字命名
import os
path = '../labelme/pictures'
files = os.listdir(path)
for i, file in enumerate(files):
    NewFileName = os.path.join(path, str(i)+'.jpg')
    OldFileName = os.path.join(path, file)
    os.rename(OldFileName, NewFileName)
