import tensorflow as tf
class FastRCNN(object):

    def __init__(self, path):

        self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
        self.im_dims = tf.placeholder(tf.float32, shape=[1, 2])
        self.y = tf.placeholder(tf.float32, shape=(None, 4))
        self.y_ = tf.placeholder(tf.int32, shape=None)
        self.learn_rate = tf.placeholder(tf.float32)
        self.reg_param1 = tf.placeholder(tf.float32)
        self.reg_param2 = tf.placeholder(tf.float32)
        self.weights_array = []
        self.bias_array = []
        self.reg_conv = tf.constant(0.)
        self.reg_fc = tf.constant(0.)
        self.pr = tf.placeholder(tf.float32)
        self.roidb = tf.placeholder(tf.float32, shape=(None, 5))
        self.create_graph(path)

    def create_conv_layer(self, graph, prev_layer, layer_scope, pretrained_weight_path_id):
        with tf.variable_scope(layer_scope):
            W = graph.get_tensor_by_name('import/conv' + pretrained_weight_path_id + '/filter:0')
            b = graph.get_tensor_by_name('import/conv' + pretrained_weight_path_id + '/bias:0')
            W = tf.Variable(W, name='weights')
            b = tf.Variable(b, name='bias')
            self.reg_conv = tf.add(self.reg_conv, tf.nn.l2_loss(W))
            conv = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME') + b
            relu = tf.nn.relu(conv)
            self.weights_array.append(W)
            self.bias_array.append(b)
            return relu

    def create_pool_layer(self, prev_layer, layer_scope):
        with tf.variable_scope(layer_scope):
            return tf.nn.max_pool(prev_layer, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    def create_graph(self, path):

        with open(path, mode='rb') as f:
            self.fileContent = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(self.fileContent)
        images = tf.placeholder("float", [None, 224, 224, 3])
        tf.import_graph_def(graph_def, input_map={"images": images})
        graph = tf.get_default_graph()

        layer = self.create_conv_layer(graph, self.x, 'conv1', '1_1')

        layer = self.create_conv_layer(graph, layer, 'conv2', '1_2')

        layer = self.create_pool_layer(layer, 'pool1')

        layer = self.create_conv_layer(graph, layer, 'conv3', '2_1')

        layer = self.create_conv_layer(graph, layer, 'conv4', '2_2')

        layer = self.create_pool_layer(layer, 'pool2')

        layer = self.create_conv_layer(graph, layer, 'conv5', '3_1')

        layer = self.create_conv_layer(graph, layer, 'conv6', '3_2')

        layer = self.create_conv_layer(graph, layer, 'conv7', '3_3')

        layer = self.create_pool_layer(layer, 'pool3')

        layer = self.create_conv_layer(graph, layer, 'conv8', '4_1')

        layer = self.create_conv_layer(graph, layer, 'conv9', '4_2')

        layer = self.create_conv_layer(graph, layer, 'conv10', '4_3')

        layer = self.create_pool_layer(layer, 'pool4')

        layer = self.create_conv_layer(graph, layer, 'conv11', '5_1')

        layer = self.create_conv_layer(graph, layer, 'conv12', '5_2')

        self.relu13 = self.create_conv_layer(graph, layer, 'conv13', '5_3')

    def roi_pooling(self , featureMaps, rois, im_dims):
        '''
        Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
        formatted as:
        (image_id, x1, y1, x2, y2)
        Note: Since mini-batches are sampled from a single image, image_id = 0s
        '''
        with tf.variable_scope('roi_pool'):
            # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
            box_ind = tf.cast(rois[:, 0], dtype=tf.int32)
            # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]

            boxes = rois[:, 1:]
            normalization = tf.cast(tf.stack([im_dims[:, 1], im_dims[:, 0], im_dims[:, 1], im_dims[:, 0]], axis=1),
                                    dtype=tf.float32)
            boxes = tf.div(boxes, normalization)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)  # y1, x1, y2, x2
            # ROI pool output size
            crop_size = tf.constant([14, 14])
            # ROI pool
            pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind,
                                                      crop_size=crop_size)
            # Max pool to (7x7)
            pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME')

        return pooledFeatures, boxes, box_ind
    def add_roi_pooling(self):
        with tf.variable_scope('roi_pooling_layer'):
            self.pool5 , self.box , self.box_ing = self.roi_pooling(self.relu13, self.roidb, self.im_dims)

    def connect_detector(self, fc_size=1024):

        with tf.variable_scope('fc1'):
            self.dense_pool = tf.reshape(self.pool5, [-1, 512*7*7])
            W14 = tf.truncated_normal(shape=(512*7*7, fc_size), stddev=1e-02)
            b14 = tf.constant(0.1, shape=[fc_size])
            W14 = tf.Variable(W14, name='weights')
            b14 = tf.Variable(b14, name='bias')
            self.reg_fc = tf.add(self.reg_fc, tf.nn.l2_loss(W14))
            self.fc1 = tf.matmul(self.dense_pool, W14) + b14

        with tf.variable_scope('classification'):
            W15 = tf.truncated_normal(shape=(fc_size, 2), stddev=1e-02)
            b15 = tf.constant(0.1, shape=[2])
            W15 = tf.Variable(W15, name='weights')
            b15 = tf.Variable(b15, name='bias')
            self.reg_fc = tf.add(self.reg_fc, tf.nn.l2_loss(W14))
            self.logits = tf.matmul(self.fc1, W15) + b15

        with tf.variable_scope('regression'):
            W16 = tf.truncated_normal(shape=(fc_size, 4), stddev=1e-02)
            b16 = tf.constant(0.1, shape=[4])
            W16 = tf.Variable(W16, name='weights')
            b16 = tf.Variable(b16, name='bias')
            self.reg_fc = tf.add(self.reg_fc, tf.nn.l2_loss(W14))
            self.boxes = tf.matmul(self.fc1, W16) + b16

        with tf.variable_scope('optimization'):
            self.reg_loss = self.reg_loss(self.boxes, self.y)
            self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                            labels=self.y_))
            self.loss_ = tf.add(self.reg_loss, self.class_loss)
            self.opt = tf.train.MomentumOptimizer(self.learn_rate, 0.9).minimize(self.loss_)

    def smooth_l1(self, x, y):
        def smooth_abs(u):
            return tf.cond(tf.less(tf.abs(u), 1.), lambda: 0.5*u**2, lambda: tf.abs(u) - 0.5)
        z = x - y
        smooth_vals = tf.map_fn(smooth_abs, z)
        return tf.reduce_mean(smooth_vals)

    def reg_loss(self, x, y):
        loss = (self.smooth_l1(x[:, 0], y[:, 0]) + self.smooth_l1(x[:, 1], y[:, 1])
                + self.smooth_l1(x[:, 2], y[:, 2]) + self.smooth_l1(x[:, 3], y[:, 3]))
        return loss

    def save_model(self):
