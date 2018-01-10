from layer import *
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras import constraints
from keras import backend as K

class Simple_model:
    def __init__(self, x, is_training, batch_size, feature_num, dropout_value, dropout_sign, is_bn, is_log, is_mask):
        self.batch_size = batch_size
        self.feature_num = feature_num
        self.dropout_value = dropout_value
        self.dropout_sign = dropout_sign
        self.is_bn = is_bn
        self.is_log = is_log
        self.is_mask = is_mask
        is_binary = True

        if is_bn:
            self.encoderv_out = self.encoder_value_bn(x, is_training)
            self.imitation = self.decoder_value_bn(self.encoderv_out, is_training)
            self.imitation_sign = self.decoder_sign_bn(self.encoder_sign_bn(x, is_training), is_training)
        else:
            self.encoderv_out = self.encoder_value(x, is_training)
            self.imitation = self.decoder_value(self.encoderv_out, is_training)
            self.imitation_sign = self.decoder_sign(self.encoder_sign(x, is_training), is_training)

        with tf.name_scope("cal_mask"):
            mask = tf.equal(x, tf.zeros_like(x))
            mask = tf.cast(mask, dtype=tf.float32)
            self.mask = 1 - mask

        if is_mask:
            with tf.name_scope("generate_data"):
                self.completion = self.imitation * self.imitation_sign
                self.mse_loss = self.calc_gv_loss(x, self.completion)
                self.completion = x * self.mask + self.completion * (1 - self.mask)
        else:
            with tf.name_scope("generate_data"):
                self.completion = self.imitation * self.imitation_sign
                self.mse_loss = self.calc_gv_loss(x, self.completion)

        if is_binary:
            with tf.name_scope("cal_binary_loss"):
                self.loss_binary = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.imitation_sign, labels=self.mask))
            self.gv_loss = self.mse_loss + self.loss_binary
            self.mse_sum = tf.summary.scalar("mse_loss",self.mse_loss)
            self.binary_sum = tf.summary.scalar("binary_loss",self.loss_binary)
        else:
            self.gv_loss = self.mse_loss

        self.real = self.discriminator(x, reuse=None)
        self.fake = self.discriminator(self.completion, reuse=True)
        # self.gv_loss= self.calc_gv_loss(x, self.completion)
        if is_log:
            self.d_loss_real,self.d_loss_fake,self.accuracy = self.calc_d_loss(self.real, self.fake)
            self.g_loss = tf.subtract(self.gv_loss, self.d_loss_fake, name="cal_g_loss")
            self.d_loss = tf.add(self.d_loss_fake, self.d_loss_real, name="cal_d_loss")
        else:
            self.d_loss_real, self.d_loss_fake, self.dg_loss_fake, self.accuracy = self.calc_d_loss1(self.real,self.fake)
            self.g_loss = tf.add(self.gv_loss, self.dg_loss_fake, name="cal_g_loss")
            self.d_loss = tf.add(self.d_loss_fake, self.d_loss_real, name="cal_d_loss")


        scope_list = ["encoder_value", "encoder_sign", "decoder_sign", "decoder_value"]
        self.g_variables = []
        for scope in scope_list:
            self.g_variables.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
        # self.g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.gv_loss_sum = tf.summary.scalar("gv_loss", self.gv_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.acc_sum =tf.summary.scalar("accuracy",self.accuracy)
        self.sign_sum = tf.summary.histogram("sign", self.imitation_sign)

        self.gv_sum = tf.summary.merge(
            [self.gv_loss_sum, self.sign_sum])
        self.g_sum = tf.summary.merge(
            [self.gv_loss_sum, self.d_loss_fake_sum, self.g_loss_sum, self.sign_sum])
        self.d_sum = tf.summary.merge(
            [self.d_loss_real_sum, self.d_loss_sum,self.acc_sum])

        if is_binary:
            self.gv_sum = tf.summary.merge([self.binary_sum,self.mse_sum,self.gv_loss_sum, self.sign_sum])
            self.g_sum = tf.summary.merge([self.binary_sum,self.mse_sum,self.gv_loss_sum, self.d_loss_fake_sum, self.g_loss_sum, self.sign_sum])

    def encoder_value(self, input ,is_training):
        with tf.variable_scope("encoder_value"):
            out = Dense(self.feature_num // 4, activation="relu")(input)
            if self.dropout_value < 1.0:
                out = tf.nn.dropout(out, 1.0-self.dropout_value, name="dropout_ev")
            out = Dense(self.feature_num // 16, activation="relu")(out)
            out = Dense(self.feature_num // 32, activation="linear")(out)
            out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

            # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
            # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
            # out = self.activation(out)

            # (None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
        return out


    def decoder_value(self, input, is_training):
        with tf.variable_scope("decoder_value") :
            # out = Dropout(0.2)(input)
            out = Dense(self.feature_num // 16, activation="relu")(input)
            out = Dense(self.feature_num // 4, activation="relu")(out)
            # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)

            if self.dropout_value < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_value, name="dropout_dv")
            out = Dense(self.feature_num, activation="linear", kernel_regularizer=regularizers.l2(0.01))(out)
            out = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")(out)

            # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
            # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
            # out = self.activation(out)

            # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
        return out

    def encoder_sign(self, input, is_training):
        with tf.variable_scope("encoder_sign"):
            out = Dense(self.feature_num // 4, activation="relu")(input)
            if self.dropout_sign < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_sign, name="dropout_es")
            # if self.dropout < 1.0:
            #     out = keras.layers.Dropout(self.dropout)(out)
            out = Dense(self.feature_num // 16, activation="relu")(out)
            out = Dense(self.feature_num // 32, activation="linear")(out)
            out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

            # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
            # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
            # out = self.activation(out)

            # (None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
        return out


    def decoder_sign(self, input, is_training):
        with tf.variable_scope("decoder_sign") :
            # out = Dropout(0.2)(input)
            out = Dense(self.feature_num // 16, activation="relu")(input)
            out = Dense(self.feature_num // 4, activation="relu")(out)
            # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)
            if self.dropout_sign < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_sign, name="dropout_ds")
            # if self.dropout > 0.0:
            #     out = keras.layers.Dropout(self.dropout)(out)
            out = Dense(self.feature_num, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(out)

           # out = keras.layers.Activation(weights=None, alpha_initializer="zero")(out)

            # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
            # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
            # out = self.activation(out)

            # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
        return out

    def encoder_value_bn(self, input, is_training):
        with tf.variable_scope("encoder_value"):
            out = Dense(self.feature_num // 4, activation="linear")(input)
            out = batch_norm(out, is_training = is_training)
            out = keras.layers.activations.relu(out)
            if self.dropout_value < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_value, name="dropout_ev")
            out = Dense(self.feature_num // 16, activation="linear")(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            out = Dense(self.feature_num // 32,activation="linear")(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

            # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
            # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
            # out = self.activation(out)

            # (None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
        return out

    def decoder_value_bn(self, input, is_training):
        with tf.variable_scope("decoder_value"):
            # out = Dropout(0.2)(input)
            out = Dense(self.feature_num // 16,activation="linear")(input)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            out = Dense(self.feature_num // 4, activation="linear")(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)

            if self.dropout_value < 1.0:
                out = tf.nn.dropout(out, 1 - self.dropout_value, name="dropout_dv")
            out = Dense(self.feature_num,activation="linear", kernel_regularizer=regularizers.l2(0.01))(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")(out)

            # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
            # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
            # out = self.activation(out)

            # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
        return out

    def encoder_sign_bn(self, input, is_training):
        with tf.variable_scope("encoder_sign"):
            out = Dense(self.feature_num // 4)(input)
            out = batch_norm(out, is_training = is_training)
            out = keras.layers.activations.relu(out)
            if self.dropout_sign < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_sign, name="dropout_es")
            out = Dense(self.feature_num // 16)(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            out = Dense(self.feature_num // 32)(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.advanced_activations.PReLU(alpha_initializer="zero", weights=None)(out)

            # out = layers.linear(input, self.hidden_size, scope="enc_first_layer")
            # out = layers.linear(out, self.hidden_size // 3, scope="enc_second_layer")
            # out = self.activation(out)

            # (None, fe) -> (None, fe // 3) -> tanh -> (None, fe // 9) -> tanh
        return out

    def decoder_sign_bn(self, input, is_training):
        with tf.variable_scope("decoder_sign"):
            # out = Dropout(0.2)(input)
            out = Dense(self.feature_num // 16)(input)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            out = Dense(self.feature_num // 4)(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.relu(out)
            # out = Dense(self.feature_num, kernel_constraint=constraints.non_neg, bias_constraint=constraints.non_neg)(out)
            if self.dropout_sign < 1.0:
                out = tf.nn.dropout(out, 1.0 - self.dropout_sign, name="dropout_ds")
            # if self.dropout > 0.0:
            #     out = keras.layers.Dropout(self.dropout)(out)
            out = Dense(self.feature_num, kernel_regularizer=regularizers.l2(0.01))(out)
            out = batch_norm(out, is_training=is_training)
            out = keras.layers.activations.sigmoid(out)

            # out = keras.layers.Activation(weights=None, alpha_initializer="zero")(out)

            # out = layers.linear(input, self.feature_num, scope="dec_first_layer")
            # out = layers.linear(out, self.feature_num, scope="dec_second_layer")
            # out = self.activation(out)

            # (None, fe // 9) -> (None, fe // 3) -> (None, fe)
        return out

    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [5, 5, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)
        return x

    def discriminator(self, input, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('d_layer1'):
                out = full_connection_layer(input, self.feature_num // 5)
                out = tf.nn.relu(out)
            with tf.variable_scope('d_layer2'):
                out = full_connection_layer(out, self.feature_num // 25)
                out = tf.nn.relu(out)
            with tf.variable_scope('d_flatten'):
                init_const = tf.constant_initializer(0.0)
                init_norm = tf.random_normal_initializer()
                w = tf.get_variable('d_w', [self.feature_num // 25, 1], initializer=init_norm)
                b = tf.get_variable('d_b', [1], initializer=init_const)
                h_logits = tf.matmul(out, w) + b
            # h_prob = tf.sigmoid(h_logits)
        return h_logits


    def _discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
                    x = conv_layer(x, [5, 5, 3, 64], 2)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = full_connection_layer(output, 1)

        return output


    def calc_gv_loss(self, x, completion):
        # loss = tf.nn.l2_loss(x - completion)
        with tf.name_scope("cal_gv_loss"):
            loss = tf.pow(x - completion,2)
            loss= tf.reduce_mean(loss)
        return loss


    def calc_d_loss(self, real, fake):
        alpha = 0.1
        #logits: theta *x ,same dimension as categories
        with tf.name_scope("cal_d_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
            d_loss_real = d_loss_real * alpha
            d_loss_fake = d_loss_fake * alpha
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    real = tf.sigmoid(real)
                    real_minus = 1 - real
                    real_two = tf.concat((real, real_minus), -1)
                    label = tf.zeros(shape=tf.shape(real),dtype=tf.int64)
                    # index = tf.argmax(real_two, 1)
                    correct_prediction = tf.equal(tf.argmax(real_two, 1), label)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return d_loss_real,d_loss_fake,accuracy

    def calc_d_loss1(self, real, fake):
        alpha = 0.5
        #logits: theta *x ,same dimension as categories
        with tf.name_scope("cal_d_loss"):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
            dg_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
            d_loss_real = d_loss_real * alpha
            d_loss_fake = d_loss_fake * alpha
            dg_loss_fake = dg_loss_fake * alpha
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    real = tf.sigmoid(real)
                    real_minus = 1 - real
                    real_two = tf.concat((real, real_minus), -1)
                    label = tf.zeros(shape=tf.shape(real),dtype=tf.int64)
                    # index = tf.argmax(real_two, 1)
                    correct_prediction = tf.equal(tf.argmax(real_two, 1), label)
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return d_loss_real,d_loss_fake,dg_loss_fake,accuracy

