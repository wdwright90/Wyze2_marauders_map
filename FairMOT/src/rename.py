import os
import re
import os.path as osp
# change the below path to the images you want to rename
path='/content/drive/MyDrive/William_house/Cam0/Seq4/im'
f=os.listdir(path)

for i in f:
    oldname=osp.join(path,i)
    num = re.findall("\d+", i)
    newname=osp.join(path,'{:06d}.jpg'.format(int(num[2])))
    os.rename(oldname,newname)
    print(oldname,'======>',newname)