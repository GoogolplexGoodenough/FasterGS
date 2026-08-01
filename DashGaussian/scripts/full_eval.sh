GPU=0
m360_path=data/360_v2
tnt_path=data/tandt_db/tandt
db_path=data/tandt_db/db
# Vanilla 3DGS with SparseAdam optimizer
python full_eval.py -m360 ${m360_path} -tat ${tnt_path} -db ${db_path} \
	--output_path output/official_fast \
	--gpu ${GPU} --fast
# Vanilla 3DGS + DashGaussian with SparseAdam optimizer
python full_eval.py -m360 ${m360_path} -tat ${tnt_path} -db ${db_path} \
	--output_path output/official_fast_dash \
	--gpu ${GPU} --fast --dash 
# Vanilla 3DGS with Adam optimizer
python full_eval.py -m360 ${m360_path} -tat ${tnt_path} -db ${db_path} \
	--output_path output/official \
	--gpu ${GPU}
# Vanilla 3DGS + DashGaussian with Adam optimizer
python full_eval.py -m360 ${m360_path} -tat ${tnt_path} -db ${db_path} \
	--output_path output/official_dash \
	--gpu ${GPU} --dash 