import os


m360 = r'/home/tuf/datasets/nerfbaselines/mipnerf360'
tat = r'/home/tuf/datasets/tandt'
db = r'/home/tuf/datasets/db'

cmd = "python full_eval.py --fastergs --gpu 0 -m360 {} -tat {} -db {} --output_path eval/Dash_faster".format(
    m360, tat, db
    )

print(cmd)
os.system(cmd)

cmd = "python full_eval.py --gpu 0 -m360 {} -tat {} -db {} --output_path eval/Dash_faster".format(
    m360, tat, db
    )

print(cmd)
os.system(cmd)

