import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)     # adj表示近邻，adj_entity, adj_relation一阶近邻实体及关系矩阵
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]  # 评分的行数

    # eval_indices，test_indices，train_indices都是列表
    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)  # 可能随机到重复的
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))  # 选取实体最多的一端
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)            # kg中存放的是字典型的一阶近邻实体及关系，keys：实体；values:[(实体，关系),(实体，关系)...]
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)    # adj_entity, adj_relation一阶近邻实体及关系矩阵

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):                        # 获得字典型的一阶近邻实体及关系，keys：实体；values:[(实体，关系),(实体，关系)...]
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]  # neighbors是[（实体，关系）,...]列表
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:    # 在entity的n_neighbors个近邻中选取neighbor_sample_size个近邻及其对应关系
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            # sampled_indices是列表
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            # 多次重复被抽到，达到neighbor_sample_size个

        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])   # 每行adj_entity存放entity被抽样的近邻实体
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])  # 每行adj_relation存放entity被抽样的近邻关系
        # neighbors是entity的一阶近邻实体及关系列表，即98行所示
    return adj_entity, adj_relation
