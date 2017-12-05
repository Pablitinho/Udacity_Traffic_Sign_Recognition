import tensorflow as tf

#----------------------------------------------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name="weight")
#----------------------------------------------------------------------------------
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name="bias")
#----------------------------------------------------------------------------------
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
#----------------------------------------------------------------------------------
def cnn_layer(input_tensor, layer_name,filter_size,num_dimension,num_filters,dropout_value, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # ----------------------------------------------------------------------------------
  # Adding a name scope ensures logical grouping of the layers in the graph.
  # ----------------------------------------------------------------------------------
  with tf.name_scope(layer_name):
    # ----------------------------------------------------------------------------------
    # This Variable will hold the state of the weights for the layer
    # ----------------------------------------------------------------------------------
    with tf.name_scope('weights'):
      weights = weight_variable([filter_size, filter_size, num_dimension, num_filters])
      variable_summaries(weights)
    # ----------------------------------------------------------------------------------
    # Bias
    # ----------------------------------------------------------------------------------
    with tf.name_scope('biases'):
      biases = bias_variable([num_filters])
      variable_summaries(biases)
    # ----------------------------------------------------------------------------------
    # Convolution
    # ----------------------------------------------------------------------------------
    with tf.name_scope('Convolution'):
        convolution = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True, name="Convolution")
    # ----------------------------------------------------------------------------------
    # Bias Addition to the convolution
    # ----------------------------------------------------------------------------------
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.add(convolution, biases)
      tf.summary.histogram('pre_activations', preactivate)
    # ----------------------------------------------------------------------------------
    # Activation
    # ----------------------------------------------------------------------------------
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    max_pool = tf.nn.max_pool(activations, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
    #dropout_max_pool = tf.nn.dropout(max_pool, keep_prob=dropout_value)
    return max_pool
# ----------------------------------------------------------------------------------
def full_connect_layer(input_tensor, layer_name, size_input, size_FC, num_ouputs, dropout_value, act=tf.nn.relu, act2=tf.nn.relu):

    # ---------------------------------------------------------------------
    # Create the Weight 1
    # ---------------------------------------------------------------------
    with tf.name_scope(layer_name):
        with tf.name_scope('weights_fc_1'):
          weights_fc_1 = weight_variable([size_input, size_FC])
          variable_summaries(weights_fc_1)
        # ---------------------------------------------------------------------
        # Create the Bias 1
        # ---------------------------------------------------------------------
        with tf.name_scope('biases_fc_1'):
          biases_fc_1 = bias_variable([size_FC])
          variable_summaries(biases_fc_1)
        # ---------------------------------------------------------------------
        # Create the Weight 2
        # ---------------------------------------------------------------------
        with tf.name_scope('weights_fc_2'):
          weights_fc_2 = weight_variable([size_FC, size_FC])
          variable_summaries(weights_fc_2)
        # ---------------------------------------------------------------------
        # Create the Bias 2
        # ---------------------------------------------------------------------
        with tf.name_scope('biases_fc_2'):
          biases_fc_2 = bias_variable([size_FC])
          variable_summaries(biases_fc_2)
        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # Create the Weight 3
        # ---------------------------------------------------------------------
        with tf.name_scope('weights_fc_3'):
          weights_fc_3 = weight_variable([size_FC, num_ouputs])
          variable_summaries(weights_fc_3)
        # ---------------------------------------------------------------------
        # Create the Bias 3
        # ---------------------------------------------------------------------
        with tf.name_scope('biases_fc_3'):
          biases_fc_3 = bias_variable([num_ouputs])
          variable_summaries(biases_fc_3)
        # ---------------------------------------------------------------------------------------------
        # Reshape the input
        # ---------------------------------------------------------------------------------------------
        input_tensor_reshaped = tf.reshape(input_tensor, [-1, size_input])
        # ---------------------------------------------------------------------------------------------
        # First FC Layer
        # ---------------------------------------------------------------------------------------------
        with tf.name_scope('Wx_plus_b_fc_1'):
            preactivate_fc_1 = tf.matmul(input_tensor_reshaped, weights_fc_1) + biases_fc_1
            tf.summary.histogram('pre_activations', preactivate_fc_1)

        activations_fc_1 = act(preactivate_fc_1, name='activation_fc_1')
        dropout_act1 = tf.nn.dropout(activations_fc_1, keep_prob=dropout_value)
        tf.summary.histogram('activations_fc_1', activations_fc_1)
        # ---------------------------------------------------------------------------------------------
        # Second FC Layer
        # ---------------------------------------------------------------------------------------------
        with tf.name_scope('Wx_plus_b_fc_2'):
            preactivate_fc_2 = tf.matmul(dropout_act1, weights_fc_2) + biases_fc_2
            tf.summary.histogram('pre_activations_fc_2', preactivate_fc_2)
        # ----------------------------------------------------------------------------------
        # Activation
        # ----------------------------------------------------------------------------------
        activations_fc_2 = act(preactivate_fc_2, name='activation_fc_2')
        dropout_act2 = tf.nn.dropout(activations_fc_2, keep_prob=dropout_value)
        tf.summary.histogram('activations_fc_2', activations_fc_2)
        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # Third FC Layer
        # ---------------------------------------------------------------------------------------------
        with tf.name_scope('Wx_plus_b_fc_3'):
            preactivate_fc_3 = tf.matmul(dropout_act2, weights_fc_3) + biases_fc_3
            tf.summary.histogram('pre_activations_fc_3', preactivate_fc_3)

        return preactivate_fc_3
        # if act2 is None:
        #     return preactivate_fc_2
        # else:
        #     return activations_fc_2

# ----------------------------------------------------------------------------------
def generate_graph_cnn(shape,num_classes, dropout_value):

    filter_size_1 = 5
    filter_size_2 = 5
    filter_size_3 = 5
    num_filters_1 = 6
    num_filters_2 = 16
    num_filters_3 = 16
    graph_1 = tf.Graph()
    with graph_1.as_default():

        # Placeholders (Input and Output)
        ph_train = tf.placeholder(tf.float32, shape=(None, shape[0], shape[1], shape[2]))
        ph_train_labels = tf.placeholder(tf.float32, shape=(None, num_classes))
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        # ----------------------------------------------------------------------------------
        # Layers definition
        # ----------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # Layer 1
        # ---------------------------------------------------------------------------------------------
        layer_1 = cnn_layer(ph_train, 'layer_1', filter_size_1, shape[2], num_filters_1, dropout_value, act=tf.nn.relu)
        layer_2 = cnn_layer(layer_1, 'layer_2', filter_size_2, int(layer_1.shape[3]), num_filters_2, dropout_value, act=tf.nn.relu)
        #layer_3 = cnn_layer(layer_2, 'layer_3', filter_size_3, int(layer_2.shape[3]), num_filters_3, act=tf.nn.relu)

        # ---------------------------------------------------------------------------------------------
        # Full Connected Layer
        # ---------------------------------------------------------------------------------------------
        output_nn = full_connect_layer(layer_2, 'Full_Connected_Layer', int(layer_2.shape[1] * layer_2.shape[2] * layer_2.shape[3]), 1024, num_classes, dropout_value, act=tf.nn.relu, act2=tf.nn.relu)
        # ----------------------------------------------------------------------------------
        # Minimization definition
        # ----------------------------------------------------------------------------------
        with tf.name_scope('accuracy'):
            # L2 regularitation
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'weight' in v.name]) * 0.0001
            #loss = tf.reduce_mean(tf.square(output_nn - ph_train_labels))
            #softmax = tf.nn.softmax(output_nn)
            #loss = tf.reduce_mean(-tf.reduce_sum(ph_train_labels*tf.log(softmax), reduction_indices=1))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_nn, labels=ph_train_labels))+lossL2
        tf.summary.scalar('accuracy', loss)
        # ----------------------------------------------------------------------------------
        # Trainer
        # ----------------------------------------------------------------------------------
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train = optimizer.minimize(loss)
        # ----------------------------------------------------------------------------------

    return ph_train, ph_train_labels, output_nn, graph_1, loss, train, keep_prob
#----------------------------------------------------------------------------------