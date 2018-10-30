from surprise import Dataset
from surprise import Reader
from surprise import evaluate, print_perf
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
import os

# 3
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# 5
data.split(n_folds=3)

algo = SVD()
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 6
algo = SVD(biased=False) #PMF
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 7
algo = NMF()
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 8
algo = KNNBasic(sim_options = {
    'user_based': True
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 9
algo = KNNBasic(sim_options = {
    'user_based': False
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 10
# Test each 5 algos w respect on RMSE and MAE on folds of 1

# 11
# Test each 5 algos w respect on RMSE and MAE on folds of 2

# 12
# Test each 5 algos w respect on RMSE and MAE on folds of 3

# 13
# Report the mean for all folds, for each of the 5 algos

# 14
algo = KNNBasic(sim_options = {
'name':’MSD’,
'user_based': True
})

algo = KNNBasic(sim_options = {
'name':’cosine’,
'user_based': True
})

algo = KNNBasic(sim_options = {
'name':’pearson’,
'user_based': True
})

# do and compare each of these for: 1) user  based filtering 2) item based filtering
# 15
