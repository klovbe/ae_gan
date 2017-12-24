import tensorflow as tf


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    with tf.name_scope("gumbel_sampling"):
        U = tf.random_uniform(shape, minval=0, maxval=1, name="unif_sampling")
        with tf.name_scope("u_trans_g"):
            out = -tf.log(-tf.log(U + eps) + eps)
    return out

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  with tf.name_scope("gumbel_softmax_sample"):
    y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)



def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
    return tf.assign_add(orig_val, scaled_diff)


def batch_norm(x,
               phase,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):

    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, [0], keep_dims=True)
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

def mlp(input, h_dim):
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w0 = tf.get_variable('w0', [input.get_shape()[1], h_dim], initializer=init_norm)
  b0 = tf.get_variable('b0', [h_dim], initializer=init_const)
  w1 = tf.get_variable('w1', [h_dim, h_dim], initializer=init_norm)
  b1 = tf.get_variable('b1', [h_dim], initializer=init_const)
  h0 = tf.tanh(tf.matmul(input, w0) + b0)
  h1 = tf.tanh(tf.matmul(h0, w1) + b1)
  return h1


def generator(input, h_dim, feature_nums):
  transform, params = mlp(input, h_dim)
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w = tf.get_variable('g_w', [h_dim, feature_nums], initializer=init_norm)
  b = tf.get_variable('g_b', [feature_nums], initializer=init_const)
  h = tf.matmul(transform, w) + b
  # s = tf.sigmoid(h)
  s = tf.tanh(h)
  return s, params + [w, b]


def minibatch(input, num_kernels=5, kernel_dim=3):
  x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
  activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
  diffs = tf.expand_dims(activation, 3) - \
          tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
  abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
  minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
  return tf.concat([input, minibatch_features], 1)


def discriminator1(input, h_dim):
  transform, params = mlp(input, h_dim)
  init_const = tf.constant_initializer(0.0)
  init_norm = tf.random_normal_initializer()
  w = tf.get_variable('d_w', [h_dim, 1], initializer=init_norm)
  b = tf.get_variable('d_b', [1], initializer=init_const)
  h_logits = tf.matmul(transform, w) + b
  h_prob = tf.sigmoid(h_logits)
  return h_prob, h_logits, params + [w, b]


# In[16]:

def optimizer(loss, var_list, num_decay_steps=1000):
  initial_learning_rate = 0.01
  decay = 0.95
  batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    batch,
    num_decay_steps,
    decay,
    staircase=True
  )
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss,
    global_step=batch,
    var_list=var_list
  )
  return optimizer


anim_frames = []


def plot_distributions(GAN, session, loss_d, loss_g):
  num_points = 100000
  num_bins = 100
  xs = np.linspace(-GAN.gen.range, GAN.gen.range, num_points)
  bins = np.linspace(-GAN.gen.range, GAN.gen.range, num_bins)

  # p(data)
  d_sample = GAN.data.sample(num_points)

  # decision boundary
  ds = np.zeros((num_points, 1))  # decision surface
  for i in range(num_points // GAN.batch_size):
    ds[GAN.batch_size * i:GAN.batch_size * (i + 1)] = session.run(GAN.D1, {
      GAN.x: np.reshape(xs[GAN.batch_size * i:GAN.batch_size * (i + 1)], (GAN.batch_size, 1))
    })

  # p(generator)
  zs = np.linspace(-GAN.gen.range, GAN.gen.range, num_points)
  gs = np.zeros((num_points, 1))  # generator function
  for i in range(num_points // GAN.batch_size):
    gs[GAN.batch_size * i:GAN.batch_size * (i + 1)] = session.run(GAN.G, {
      GAN.z: np.reshape(
        zs[GAN.batch_size * i:GAN.batch_size * (i + 1)],
        (GAN.batch_size, 1)
      )
    })

  anim_frames.append((d_sample, ds, gs, loss_d, loss_g))
