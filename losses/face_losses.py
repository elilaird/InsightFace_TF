import tensorflow as tf
import pandas as pd
import numpy as np
import math


def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


def combine_loss_val(embedding, labels, w_init, out_num, margin_a, margin_m, margin_b, s):
    '''
    This code is contributed by RogerLo. Thanks for you contribution.

    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                              initializer=w_init, dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    ordinal = tf.constant(list(range(0, embedding.get_shape().as_list()[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
        if margin_a == 1.0 and margin_m == 0.0:
            s_m = s * margin_b
            new_zy = sel_cos_t - s_m
        else:
            cos_value = sel_cos_t / s
            t = tf.acos(cos_value)
            if margin_a != 1.0:
                t = t * margin_a
            if margin_m > 0.0:
                t = t + margin_m
            body = tf.cos(t)
            if margin_b > 0.0:
                body = body - margin_b
            new_zy = body * s
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
    predict_cls = tf.argmax(updated_logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    predict_cls_s = tf.argmax(zy, 1)
    accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    return zy, loss, accuracy, accuracy_s, predict_cls_s

def _calculate_clustering_index(S):

    #create pandas dataframe for grouping
    colnames = ['demographic'] + ['V'+str(i) for i in range(1, S.shape[1])]
    df = pd.DataFrame(
        data=S,
        columns=colnames
    )

    df = pd.melt(df, id_vars=['demographic'], var_name='PC', value_name='Coef', value_vars=colnames[1:])

    #calculate overall variance per principal component
    var_overall = df \
        .groupby(['PC']) \
        .agg({'Coef':'var'}) \
        .reset_index() \
        .rename(columns={'Coef': 'var.overall'})

    #calculate within cluster variance
    var_within = df \
        .groupby(['PC','demographic']) \
        .agg({'PC':'count','Coef':'var'}) \
        .rename(columns={'PC':'n', 'Coef':'var.within'})

    var_within['product'] = var_within['n'] * var_within['var.within']

    var_within = var_within[['n','product']] \
        .groupby(['PC']) \
        .agg({'n':'sum', 'product':'sum'}) \
        .rename(columns={'n':'sum_n', 'product':'sum_prod'})

    var_within['var.within.sum'] = var_within['sum_prod'] / var_within['sum_n']

    var_within.reset_index(inplace=True)

    clustering = var_within.join(var_overall.set_index('PC'), on='PC', how='left')
    clustering['clustering_index'] = 1 - (clustering['var.within.sum'] / clustering['var.overall'])
    clustering = clustering[['PC','clustering_index']]

    #print(clustering.sort_values(by=['clustering_index'], ascending=False))

    return clustering['clustering_index'].values.astype(np.float32)





def broad_homogeneity_loss(embedding, labels):

    #in tensorflow
    # #normalize embeddings
    # embedding_norm = tf.norm(embedding, ord=2, axis=1, keep_dims=True)
    # embedding_norm = tf.reshape(embedding_norm, (-1,1))
    # embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
    #
    # #compute svd
    # _, u, _ = tf.svd(embedding, compute_uv=True, name='svd_embedding')
    #
    # #calculate clustering effects
    # S = tf.stack([labels[1], u])
    # clustering = _calculate_clustering_index(S.eval())

    #normalize embeddings
    embedding_norm = np.linalg.norm(embedding, axis=1)
    embedding = np.divide(embedding, embedding_norm[:, np.newaxis])

    #compute svd
    u, _, _ = np.linalg.svd(embedding, full_matrices=True)

    dem = labels[:,1].reshape(-1,1)
    dem = dem.astype('float32')

    #calculate clustering
    S = np.hstack((dem,u))
    clustering = _calculate_clustering_index(S)


    return clustering

