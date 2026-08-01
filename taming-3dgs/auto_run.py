import os


m360 = r'/home/tuf/datasets/nerfbaselines/mipnerf360'
tat = r'/home/tuf/datasets/tandt'
db = r'/home/tuf/datasets/db'
# cmd = "python full_eval.py -m360 {} -tat {} -db {} --sh_lower --mode budget --output_path eval/Taming > Taming.log".format(m360, tat, db)

# print(cmd)
# os.system(cmd)


cmd = "python full_eval.py --fastergs -m360 {} -tat {} -db {} --sh_lower --mode budget --output_path eval/Taming_faster ".format(m360, tat, db)

print(cmd)
os.system(cmd)


cmd = "python full_eval.py -m360 {} -tat {} -db {} --sh_lower --mode budget --output_path eval/Taming ".format(m360, tat, db)

print(cmd)
os.system(cmd)



# cmd = "python full_eval_faster.py -m360 {} -tat {} -db {} --sh_lower --mode budget --skip_training --output_path eval/Taming_faster ".format(m360, tat, db)

# print(cmd)
# os.system(cmd)