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

print('\n#{} SVD -------------------------------\n'.format(5))
data.split(n_folds=3)

algo = SVD()
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 6
print('\n#{} PMF-------------------------------\n'.format(6))


algo = SVD(biased=False) #PMF
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 7
print('\n#{} NMF-------------------------------\n'.format(7))


algo = NMF()
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 8
print('\n#{} User Based-------------------------------\n'.format(8))


algo = KNNBasic(sim_options = {
    'user_based': True
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 9
print('\n#{} Item Based-------------------------------\n'.format(9))

algo = KNNBasic(sim_options = {
    'user_based': False
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)


# 10 - 13
# for f in range(1,4):

# 10
# Test each 5 algos w respect on RMSE and MAE on folds of 1

# 11
# Test each 5 algos w respect on RMSE and MAE on folds of 2

# 12
# Test each 5 algos w respect on RMSE and MAE on folds of 3

# 13
# Report the mean for all folds, for each of the 5 algos

# 14
print('\n#{}-------------------------------\n'.format(14))

print('\n User MSD-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'MSD',
    'user_based': True
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

print('\n User Cosine-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'cosine',
    'user_based': True
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

print('\n User Pearson-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'pearson',
    'user_based': True  
    })
    ####################
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

print('\n Item MSD-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'MSD',
    'user_based': False
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

print('\n Item Cosine-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'cosine',
    'user_based': False
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

print('\n Item Pearson-------------------------------\n'.format(14))
algo = KNNBasic(sim_options = {
    'name':'pearson',
    'user_based': False  
    })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)

# 15
# plot: x - number of neighbors, y - line (user, collab)

print('\n#{}-------------------------------\n'.format(15))
for k in range(1, 20):
    print('\n k={} User------------\n'.format(k))
    algo = KNNBasic(k=20, sim_options = {'name':'MSD', 'user_based': True })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    print('\n k={} Item------------\n'.format(k))
    algo = KNNBasic(k=20, sim_options = {'name':'MSD', 'user_based': False })
    perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)