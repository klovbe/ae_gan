import numpy as np
import tensorflow as tf
import tqdm
from Dataset import DataSet
from simple_separate import Simple_separate
import os
import pandas as pd
import shutil
import load
import keras.backend as K


def train():
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    PRETRAIN_EPOCH = 1000
    PRETRAIN_EPOCH_d = 1100
    feature_nums = 15549
    dropout_value = 0.9
    dropout_sign = 1.0
    train_datapath = r"F:/project/simulation_data/drop60_p.train"
    EPOCH = 2500
    # outDir = r"F:/project/simulation_data/drop60/bn_"
    model_name = "AE-GAN_bn_dp_0.9_0_separate"
    load_checkpoint = False
    outDir = os.path.join("F:/project/simulation_data/drop60", model_name)
    model = "separate"

    x = tf.placeholder(tf.float32, [None, feature_nums], name="input_data")
    # completion = tf.placeholder(tf.float32, [BATCH_SIZE, feature_nums])
    is_training = tf.placeholder(tf.bool, [], name="is_training")
    # completed = tf.placeholder(tf.float32,[None, feature_nums], name="generator_out")
    mask = tf.placeholder(tf.float32, [None, feature_nums], name="input_data")


    model = Simple_separate(x, mask, is_training, batch_size=BATCH_SIZE, feature_num=feature_nums, dropout_value=dropout_value,
                    dropout_sign=dropout_sign, is_bn=True)
    sess = tf.Session()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)

    with tf.name_scope("adam_optimizer"):
        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        gv_train_op = opt.minimize(model.gv_loss, global_step=global_step, var_list=model.g_variables)
        g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
        d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()

    load_model_dir = os.path.join('./backup', model_name)
    if load_checkpoint and os.path.exists('./backup/' + model_name):
        ckpt = tf.train.get_checkpoint_state(load_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    elif os.path.exists(load_model_dir):
        shutil.rmtree(load_model_dir)
    else:
        os.makedirs(load_model_dir)

    logs_dir = os.path.join("./logs", model_name)
    if load_checkpoint is False and os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    # if tf.train.get_checkpoint_state('./backup'):
    #     saver = tf.train.Saver()
    #     saver.restore(sess, './backup/latest')
    #
    # logs_dir = os.path.join("./logs", model_name)
    # if os.path.exists(logs_dir) == False:
    #     os.makedirs(logs_dir)
    # writer = tf.summary.FileWriter(logs_dir, sess.graph)


    dataset = DataSet(train_datapath, BATCH_SIZE)
    # each epoch has step_num steps
    step_num = dataset.steps

    while sess.run(epoch) < EPOCH:
        sess.run(tf.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        # Completion
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = dataset.next()
                _, gv_loss, gv_summary_str = sess.run([gv_train_op, model.gv_loss, model.gv_sum],
                                                      feed_dict={x: x_batch, is_training: True, K.learning_phase(): 1})
                if i % 10 == 0:
                    writer.add_summary(gv_summary_str)

            print('Completion loss: {}'.format(gv_loss))
            if sess.run(epoch) % 100 == 0:
                saver.save(sess, load_model_dir + '/pretrained_g', write_meta_graph=False)
            if sess.run(epoch) == PRETRAIN_EPOCH:
                dataset = DataSet(train_datapath, BATCH_SIZE, onepass=True, shuffle=False)
                imitate_datas = []
                complete_datas = []
                embed_datas = []
                for i in tqdm.tqdm(range(step_num + 1)):
                    x_batch = dataset.next()
                    mask = x_batch == 0
                    embed, imitation, completion = sess.run([model.encoderv_out, model.imitation, model.completion],
                                                            feed_dict={x: x_batch, is_training: False,
                                                                       K.learning_phase(): 0})
                    completion = np.array(completion, dtype=np.float)
                    imitation = np.array(imitation, dtype=np.float)
                    embed = np.array(embed, dtype=np.float)
                    mask = mask.astype(float)
                    completion = x_batch * (1 - mask) + completion * mask
                    imitation = x_batch * (1 - mask) + imitation * mask
                    complete_datas.append(completion)
                    imitate_datas.append(imitation)
                    embed_datas.append(embed)

                dataset = DataSet(train_datapath, BATCH_SIZE)
                complete_datas = np.reshape(np.concatenate(complete_datas, axis=0), (-1, feature_nums))
                imitate_datas = np.reshape(np.concatenate(imitate_datas, axis=0), (-1, feature_nums))
                embed_datas = np.reshape(np.concatenate(embed_datas, axis=0), (-1, feature_nums // 32))
                df_c = pd.DataFrame(complete_datas)
                df_i = pd.DataFrame(imitate_datas)
                df_e = pd.DataFrame(embed_datas)
                if os.path.exists(outDir) == False:
                    os.makedirs(outDir)
                # outPath = os.path.join(outDir, "infer.complete")
                df_c.to_csv(outDir + "generator.imitate", index=None)
                df_i.to_csv(outDir + "generator.complete", index=None)
                df_e.to_csv(outDir + "generator.embed", index=None)
                print("save complete data to {}".format(outDir + "infer.complete"))
                saver.save(sess, load_model_dir + '/pretrained_g', write_meta_graph=False)

        # Discrimitation
        elif sess.run(epoch) <= PRETRAIN_EPOCH_d:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = dataset.next()
                _, d_loss, d_summary_str = sess.run(
                    [d_train_op, model.d_loss, model.d_sum],
                    feed_dict={x: x_batch, is_training: True, K.learning_phase(): 1})
                if i % 10 == 0:
                    writer.add_summary(d_summary_str)

            print('Discriminator loss: {}'.format(d_loss))
            if sess.run(epoch) % 100 == 0:
                saver = tf.train.Saver()
                saver.save(sess, load_model_dir + '/pretrained_d', write_meta_graph=False)

        # together
        elif sess.run(epoch) < EPOCH:
            for i in tqdm.tqdm(range(step_num)):
                x_batch = dataset.next()
                _, d_loss, d_summary_str = sess.run(
                    [d_train_op, model.d_loss, model.d_sum],
                    feed_dict={x: x_batch, is_training: True, K.learning_phase(): 1})
                if i % 10 == 0:
                    writer.add_summary(d_summary_str)

                _, g_loss, g_summary_str = sess.run([g_train_op, model.g_loss, model.g_sum],
                                                    feed_dict={x: x_batch, is_training: True, K.learning_phase(): 1})
                if i % 10 == 0:
                    writer.add_summary(g_summary_str)

            print('Completion loss: {}'.format(g_loss))
            print('Discriminator loss: {}'.format(d_loss))
            if sess.run(epoch) % 100 == 0:
                saver = tf.train.Saver()
                saver.save(sess, load_model_dir + '/latest', write_meta_graph=False)


        elif sess.run(epoch) == EPOCH:
            dataset = DataSet(train_datapath, BATCH_SIZE, onepass=True, shuffle=False)
            imitate_datas = []
            complete_datas = []
            embed_datas = []
            for i in tqdm.tqdm(range(step_num + 1)):
                x_batch = dataset.next()
                mask = x_batch == 0
                embed, imitation, completion = sess.run([model.encoderv_out, model.imitation, model.completion],
                                                        feed_dict={x: x_batch, is_training: False,
                                                                   K.learning_phase(): 0})
                completion = np.array(completion, dtype=np.float)
                imitation = np.array(imitation, dtype=np.float)
                embed = np.array(embed, dtype=np.float)
                mask = mask.astype(float)
                completion = x_batch * (1 - mask) + completion * mask
                imitation = x_batch * (1 - mask) + imitation * mask
                complete_datas.append(completion)
                imitate_datas.append(imitation)
                embed_datas.append(embed)
            # saver = tf.train.Saver()
            # saver.save(sess, load_model_dir+'/complete', write_meta_graph=False)

            complete_datas = np.reshape(np.concatenate(complete_datas, axis=0), (-1, feature_nums))
            imitate_datas = np.reshape(np.concatenate(imitate_datas, axis=0), (-1, feature_nums))
            embed_datas = np.reshape(np.concatenate(embed_datas, axis=0), (-1, feature_nums // 32))
            df_c = pd.DataFrame(complete_datas)
            df_i = pd.DataFrame(imitate_datas)
            df_e = pd.DataFrame(embed_datas)
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            # outPath = os.path.join(outDir, "infer.complete")
            df_c.to_csv(outDir + "infer.imitate", index=None)
            df_i.to_csv(outDir + "infer.complete", index=None)
            df_e.to_csv(outDir + "infer.embed", index=None)
            print("save complete data to {}".format(outDir))


if __name__ == '__main__':
    train()

