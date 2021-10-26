import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})


def read_item_index_to_entity_id_file():        # 读取原始物品ID对应的原始实体ID文件，然后赋予新的共同的ID
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i                  # 原始物品ID对应的新的ID
        entity_id2index[satori_id] = i                      # 原始实体ID对应的新的ID
        i += 1                                                # 原始实体和物品对应共同的ID


def convert_rating():         # 读取原始评分文件，单个用户中，把大于阈值的评分赋值为1，把没有过评分的物品赋值为0，然后保存到最终的评分文件中，里面保存的都是用户和物品的新ID
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:   # 去除第一行，逐行读取，循环完毕得到用户有过正评分的物品集合，用户有过负评分的物品集合
        array = line.strip().split(SEP[DATASET])                # 其中，用户对应是原始ID，items集合对应新ID

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]                                # array[1]为moiveID
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]         # 把原物品ID换成新的ID

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()        # 用户的原始ID作为字典的key，有过正评分的物品的新ID集合作为values
            user_pos_ratings[user_index_old].add(item_index)    # 用户有过正评分的物品集合
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()        # 用户的原始ID作为字典的key，有过负评分的物品的新ID集合最为values
            user_neg_ratings[user_index_old].add(item_index)    # 用户有过负评分的物品集合

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()             # 用户原ID对新ID字典
    for user_index_old, pos_item_set in user_pos_ratings.items():  # pos_item_set是一个物品集合
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set                 # 总电影减去用户有过正评分的电影，item_set中存储的是新的item的ID
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]   # 减去用户有过负评分的电影，得到用户未观看的电影
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():       # 读取原始知识图谱文件，把实体原始ID转换成新ID，给关系分配ID，然后保存到最终的知识图谱文件中
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    for line in open('../data/' + DATASET + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()            # 实体原始ID对新ID字典
    relation_id2index = dict()          # 关系对新ID字典
    item_index_old2new = dict()         # 物品原始ID对新ID字典

    read_item_index_to_entity_id_file()     # 读取原始物品ID对原始实体ID文件，然后赋予新的共同的ID
    convert_rating()  # 读取原始评分文件，单个用户中，把大于阈值的评分赋值为1，把没有过评分的物品赋值为0，然后保存到最终的评分文件中，里面保存的都是用户和物品的新ID
    convert_kg()    # 读取原始知识图谱文件，把实体原始ID转换成新ID，给关系分配ID，然后保存到最终的知识图谱文件中

    print('done')
