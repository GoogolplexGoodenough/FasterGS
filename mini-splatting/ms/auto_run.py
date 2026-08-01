import os


m360 = r'/home/tuf/datasets/nerfbaselines/mipnerf360'
tat = r'/home/tuf/datasets/tandt'
db = r'/home/tuf/datasets/db'


cmd = "python full_eval.py --fastergs -m360 {} -tat {} -db {} --output_path eval/Mini_faster".format(m360, tat, db)

print(cmd)
os.system(cmd)



cmd = "python full_eval.py -m360 {} -tat {} -db {} --output_path eval/Mini".format(m360, tat, db)

print(cmd)
os.system(cmd)