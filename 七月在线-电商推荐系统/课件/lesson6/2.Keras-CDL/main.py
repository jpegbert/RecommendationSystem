import logging
from utils import read_rating, read_feature
from CDL import CollaborativeDeepLearning

def main():
    logging.info('reading data')
    ratings = read_rating('ratings.dat')
    data_num = len(ratings)
    train_mat = ratings[:int(data_num*0.8)]
    test_mat = ratings[int(data_num*0.8):]
    item_mat = read_feature('movies.dat')
    num_item_feat = item_mat.shape[1]

    model = CollaborativeDeepLearning(item_mat, [num_item_feat, 16, 8])
    model.pretrain(lamda_w=0.001, encoder_noise=0.3, epochs=10)
    model_history = model.fineture(train_mat, test_mat, lamda_u=0.01, lamda_v=0.1, lamda_n=0.1, lr=0.01, epochs=3)
    testing_rmse = model.getRMSE(test_mat)
    print('Testing RMSE = {}'.format(testing_rmse))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    main()