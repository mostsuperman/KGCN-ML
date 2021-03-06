import tensorflow as tf
import numpy as np
from model import KGCN


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    test_auc_list = []
    test_f1_list = []
    test_acc_list = []

    test_topk_precision_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_topk_recall_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_topk_f1_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_acc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_acc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_acc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            test_auc_list.append(test_auc)
            test_f1_list.append(test_f1)
            test_acc_list.append(test_acc)

            print('epoch %d    train auc: %.4f  acc: %.4f  f1: %.4f    eval auc: %.4f  acc: %.4f  f1: %.4f    test auc: %.4f  acc: %.4f  f1: %.4f'
                  % (step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall, f1 = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)

                test_topk_precision_list = np.array(test_topk_precision_list) + np.array(precision)
                test_topk_recall_list = np.array(test_topk_recall_list) + np.array(recall)
                test_topk_f1_list = np.array(test_topk_f1_list) + np.array(f1)

                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()

                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()

                print('f1@K: ', end='')
                for i in f1:
                    print('%.4f\t' % i, end='')
                print('\n')

        avg_top1_acc = float(test_topk_precision_list[0] / float(args.n_epochs))
        avg_top2_acc = float(test_topk_precision_list[1] / float(args.n_epochs))
        avg_top5_acc = float(test_topk_precision_list[2] / float(args.n_epochs))
        avg_top10_acc = float(test_topk_precision_list[3] / float(args.n_epochs))
        avg_top20_acc = float(test_topk_precision_list[4] / float(args.n_epochs))
        avg_top50_acc = float(test_topk_precision_list[5] / float(args.n_epochs))
        avg_top100_acc = float(test_topk_precision_list[6] / float(args.n_epochs))

        avg_top1_recall = float(test_topk_recall_list[0] / float(args.n_epochs))
        avg_top2_recall = float(test_topk_recall_list[1] / float(args.n_epochs))
        avg_top5_recall = float(test_topk_recall_list[2] / float(args.n_epochs))
        avg_top10_recall = float(test_topk_recall_list[3] / float(args.n_epochs))
        avg_top20_recall = float(test_topk_recall_list[4] / float(args.n_epochs))
        avg_top50_recall = float(test_topk_recall_list[5] / float(args.n_epochs))
        avg_top100_recall = float(test_topk_recall_list[6] / float(args.n_epochs))

        avg_top1_f1 = float(test_topk_f1_list[0] / float(args.n_epochs))
        avg_top2_f1 = float(test_topk_f1_list[1] / float(args.n_epochs))
        avg_top5_f1 = float(test_topk_f1_list[2] / float(args.n_epochs))
        avg_top10_f1 = float(test_topk_f1_list[3] / float(args.n_epochs))
        avg_top20_f1 = float(test_topk_f1_list[4] / float(args.n_epochs))
        avg_top50_f1 = float(test_topk_f1_list[5] / float(args.n_epochs))
        avg_top100_f1 = float(test_topk_f1_list[6] / float(args.n_epochs))

        print('test average_auc: %.4f   average_acc: %.4f  average_f1: %.4f'
              % (float(np.mean(test_auc_list)), float(np.mean(test_acc_list)), float(np.mean(test_f1_list))))

        print('avg_precision@k top1: %.4f   top2: %.4f   top5: %.4f    top10: %.4f   top20: %.4f  top50: %.4f   top100: %.4f'
              % (avg_top1_acc, avg_top2_acc, avg_top5_acc, avg_top10_acc, avg_top20_acc, avg_top50_acc, avg_top100_acc))

        print('avg_recall@k top1: %.4f   top2: %.4f   top5: %.4f    top10: %.4f   top20: %.4f  top50: %.4f   top100: %.4f'
              % (avg_top1_recall, avg_top2_recall, avg_top5_recall, avg_top10_recall, avg_top20_recall, avg_top50_recall, avg_top100_recall))

        print('avg_f1@k top1: %.4f   top2: %.4f   top5: %.4f    top10: %.4f   top20: %.4f  top50: %.4f   top100: %.4f'
              % (avg_top1_f1, avg_top2_f1, avg_top5_f1, avg_top10_f1, avg_top20_f1, avg_top50_f1, avg_top100_f1))


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)  # ?????????keys?????????ID???values???????????????????????????????????????????????????item?????????label=0???1
        test_record = get_user_record(test_data, False)  # ?????????keys?????????ID???values????????????????????????????????????????????????item??????????????????label=1
        user_list = list(set(train_record.keys()) & set(test_record.keys()))  # ?????????????????????????????????????????????
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)  # ????????????100???????????????????????????????????????????????????
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    acc_list = []
    while start + batch_size <= data.shape[0]:
        auc, acc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)

        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}  # ??????
    recall_list = {k: [] for k in k_list}  # ??????

    for user in user_list:
        test_item_list = list(item_set - train_record[user])  # ??????????????????????????????????????????????????????items
        item_score_map = dict()  # ?????????????????????item?????????
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,  # ????????????????????????????????????????????????????????????
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})  # ?????????????????????
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)  # ???item_score_map????????????
        item_sorted = [i[0] for i in item_score_pair_sorted]  # item_sorted????????????????????????item?????????i???????????????[item???????????????]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])  # ???k????????? ??????????????????????????????item?????? ????????????
            precision_list[k].append(hit_num / k)  # precision_list??????????????????keys???k???values???????????????????????????????????????
            recall_list[k].append(hit_num / len(test_record[user]))  # recall_list??????????????????keys???k???values???????????????????????????????????????

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(len(k_list))]

    return precision, recall, f1


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
