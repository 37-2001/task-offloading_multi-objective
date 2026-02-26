import tensorflow.contrib.layers as tl

from state_normalization import normalize_node_inputs, normalize_job_inputs
from tf_op import *
from msg_passing_path import *
from gcn1 import GraphCNN
from gsn1 import GraphSNN
from agent import Agent
from  sklearn.preprocessing import  StandardScaler

class ActorAgent1(Agent):
    def __init__(self,sess,node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, mec_index, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent1'):

        Agent.__init__(self)
        self.sess = sess
        self.node_input_dim = node_input_dim
        self.job_input_dim = job_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.mec_index = mec_index
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # for computing and storing message passing path
        self.postman = Postman()
        # node input dimension: [total_num_nodes, num_features]
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])

        # job input dimension: [total_num_jobs, num_features]
        self.job_inputs = tf.placeholder(tf.float32, [None, self.job_input_dim])

        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)

        # valid mask for node action ([batch_size, total_num_nodes])任务动作的有效掩码
        self.node_valid_mask = tf.placeholder(tf.float32, [None, None])

        # valid mask for executor limit on jobs ([batch_size, num_jobs * num_mec_index])
        self.job_valid_mask = tf.placeholder(tf.float32, [None, None])

        # map back the dag summeraization to each node ([total_num_nodes, num_dags])
        self.dag_summ_backward_map = tf.placeholder(tf.float32, [None, None])

        # map gcn_outputs and raw_inputs to action probabilities
        # node_act_probs: [batch_size, total_num_nodes]
        # job_act_probs: [batch_size, total_num_dags]
        self.node_act_probs1, self.job_act_probs1 = self.actor_network1(
            self.node_inputs, self.gcn.outputs, self.job_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1],
            self.node_valid_mask, self.job_valid_mask,
            self.dag_summ_backward_map, self.act_fn)

        # job_acts [batch_size, num_jobs, 1]
        logits = tf.log(self.job_act_probs1)
        noise = tf.random_uniform(tf.shape(logits))
        self.job_acts = tf.argmax(logits - tf.log(-tf.log(noise)), 2)

        # Selected action for node, 0-1 vector ([batch_size, total_num_nodes])
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        # Selecepsted action for job, 0-1 vector ([batch_size, num_jobs, num_mec_index])
        self.job_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # select node action probability
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs1, self.node_act_vec),
            reduction_indices=1, keep_dims=True)

        # select job action probability
        self.selected_job_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.job_act_probs1, self.job_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_node_prob * self.selected_job_prob + \
                   self.eps), -self.adv))

        # node_entropy
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs1, tf.log(self.node_act_probs1 + self.eps)))

        # prob on each job
        self.prob_each_job = tf.reshape(
            tf.sparse_tensor_dense_matmul(self.gsn.summ_mats[0],
                                          tf.reshape(self.node_act_probs1, [-1, 1])),
            [tf.shape(self.node_act_probs1)[0], -1])

        # job entropy
        self.job_entropy = \
            tf.reduce_sum(tf.multiply(self.prob_each_job,tf.reduce_sum(tf.multiply(self.job_act_probs1,
            tf.log(self.job_act_probs1 + self.eps)),
            reduction_indices=2)))

        # entropy loss
        self.entropy_loss = self.node_entropy + self.job_entropy

        # normalize entropy
        self.entropy_loss /= \
            (tf.log(tf.cast(tf.shape(self.node_act_probs1)[1], tf.float32)) +
             tf.log(float(len(self.mec_index))))
        # normalize over batch size (note: adv_loss is sum)

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate). \
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network1(self, node_inputs, gcn_outputs, job_inputs,
                      gsn_dag_summary, gsn_global_summary,
                      node_valid_mask, job_valid_mask,
                      gsn_summ_backward_map, act_fn):

        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_valid_mask)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim])

        # (2) reshape job inputs to batch format
        job_inputs_reshape = tf.reshape(
            job_inputs, [batch_size, -1, self.job_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim])

        # (5) reshape gsn_dag_summary to batch format
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_summ_backward_map_extend = tf.tile(
            tf.expand_dims(gsn_summ_backward_map, axis=0), [batch_size, 1, 1])
        gsn_dag_summ_extend = tf.matmul(
            gsn_summ_backward_map_extend, gsn_dag_summ_reshape)

        # (6) reshape gsn_global_summary to batch format
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_job = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        gsn_global_summ_extend_node = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_extend)[1], 1])
        # (4) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_node = tf.concat([
                node_inputs_reshape, gcn_outputs_reshape,
                gsn_dag_summ_extend,
                gsn_global_summ_extend_node], axis=2)

            node_hid_0 = tl.fully_connected(merge_node, 64, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 32, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 16, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_num_nodes)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])
            # valid mask on node
            node_valid_mask = (node_valid_mask - 1) * 10000.0
            # apply mask
            node_outputs = node_outputs + node_valid_mask
            # do masked softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)
            # -- part B, the distribution over executor limits --
            merge_job = tf.concat([
                job_inputs_reshape,
                gsn_dag_summ_reshape,
                gsn_global_summ_extend_job], axis=2)

            expanded_state = expand_act_on_state(
                merge_job, [l / 1.0 for l in self.mec_index])

            job_hid_0 = tl.fully_connected(expanded_state, 64, activation_fn=act_fn)
            job_hid_1 = tl.fully_connected(job_hid_0, 32, activation_fn=act_fn)
            job_hid_2 = tl.fully_connected(job_hid_1, 16, activation_fn=act_fn)
            job_outputs = tl.fully_connected(job_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_mecs)
            job_outputs = tf.reshape(job_outputs, [batch_size, -1])

            # valid mask on job
            job_valid_mask = (job_valid_mask - 1) * 10000.0

            # apply mask
            job_outputs = job_outputs + job_valid_mask

            # reshape output dimension for softmaxing the executor limits
            # (batch_size, num_jobs, num_exec_limits)
            job_outputs = tf.reshape(
                job_outputs, [batch_size, -1, len(self.mec_index)])

            # do masked softmax over jobs
            job_outputs = tf.nn.softmax(job_outputs, dim=-1)

            return node_outputs, job_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(
                self.act_gradients + [self.lr_rate],
                gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + self.gsn.summ_mats,
                [node_inputs] + summ_mats)
        })

    def get_params(self):
        return self.sess.run(self.params)

    def load_model1(self, model_path):
        variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor_agent1")
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(self.sess, model_path)

    def save_model1(self, file_path):
        self.saver.save(self.sess, file_path)

    def get_gradients(self, node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map,
            node_act_vec, job_act_vec, adv, entropy_weight):

        return self.sess.run([self.act_gradients,
            [self.adv_loss, self.entropy_loss], self.node_entropy, self.job_entropy, self.selected_job_prob,self.selected_node_prob],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + \
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map] + [self.node_act_vec] + \
                [self.job_act_vec] + [self.adv] + [self.entropy_weight],
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] + \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map] + [node_act_vec] + \
                [job_act_vec] + [adv] + [entropy_weight])
        })


    def predict(self, node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            gcn_mats, gcn_masks, summ_mats,
            running_dags_mat, dag_summ_backward_map):
        return self.sess.run([self.node_act_probs1, self.job_act_probs1],
            feed_dict={i: d for i, d in zip(
                [self.node_inputs] + [self.job_inputs] + \
                [self.node_valid_mask] + [self.job_valid_mask] + \
                self.gcn.adj_mats + self.gcn.masks + self.gsn.summ_mats + \
                [self.dag_summ_backward_map],
                [node_inputs] + [job_inputs] + \
                [node_valid_mask] + [job_valid_mask] +  \
                gcn_mats + gcn_masks + \
                [summ_mats, running_dags_mat] + \
                [dag_summ_backward_map])
        })


    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs):
        """
        Translate the observation to matrix form
        """
        job_dags, frontier_nodes, free_mecs, node_action_map, mec_action_map = obs

        # compute total number of nodes
        total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags), self.job_input_dim])
        # gather job level inputs
        job_idx = 0
        for job_dag in job_dags:
            # 接入的服务器索引
            job_inputs[job_idx, 0] = job_dag.access_mec_id
            # 信道增益
            for i in range(0, len(args.mec_capacity)):
                job_inputs[job_idx, i + 1] = args.mec_capacity[i]
            job_idx += 1
        # gather node level inputs 4个
        job_idx = 0
        node_idx = 0
        for job_dag in job_dags:
            for node in job_dag.nodes:
                # node_inputs[node_idx, :2] = job_inputs[job_idx, :2]
                # node的cpu周期数
                node_inputs[node_idx, 0] = node.cpu_circles
                # node的上传的数据量
                node_inputs[node_idx, 1] = node.data_size
                # node的传输时间
                mapping = {'soft': 0, 'hard': 1}
                node_inputs[node_idx, 2] = mapping[node.job_dag.var]
                # node的平均计算时间
                node_inputs[node_idx, 3] = node.get_avg_computation_time(args.mec_capacity) + node.get_transmission_time()
                # node的分配的ddl
                node_inputs[node_idx, 4] = node.assigned_ddl()
                node_idx += 1
            job_idx += 1
        # transfer = StandardScaler()
        # node_inputs = transfer.fit_transform(node_inputs)
        node_inputs = normalize_node_inputs(node_inputs)
        job_inputs = normalize_job_inputs(job_inputs)

        return node_inputs, job_inputs, job_dags, frontier_nodes,\
               free_mecs, node_action_map, mec_action_map

    def get_valid_masks(self, job_dags, frontier_nodes, free_mecs, node_action_map, mec_action_map):

        job_valid_mask = np.zeros([1, len(job_dags) * len(self.mec_index)])
        base = 0
        for i in range(len(job_dags)):
            for mec in free_mecs:  # free_mecs是空闲的mec集合
                mec_idx = mec_action_map.inverse_map[mec]
                job_valid_mask[0, base + mec_idx] = 1
            base += args.num_mecs + 1

        # 选node动作的有效掩码，必须是可调度的node才是有效的动作
        total_num_nodes = int(np.sum(
            job_dag.num_nodes for job_dag in job_dags))
        node_valid_mask = np.zeros([1, total_num_nodes])
        for node in frontier_nodes:  # frontier_nodes 是可调度的任务的集合
            node_act = node_action_map.inverse_map[node]
            node_valid_mask[0, node_act] = 1
        return node_valid_mask, job_valid_mask

    def invoke_model(self, obs):
        # implement this module here for training 在这里实现这个模块进行培训
        # (to pick up state and action to record,为了拾取状态和动作进行记录)
        node_inputs, job_inputs,job_dags, frontier_nodes, free_mecs, node_action_map, mec_action_map = self.translate_state(obs)

        # get message passing path (with cache)
        gcn_mats, gcn_masks, dag_summ_backward_map, \
        running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags)

        # get node and job valid masks
        node_valid_mask, job_valid_mask = \
            self.get_valid_masks(job_dags, frontier_nodes,free_mecs, node_action_map, mec_action_map)

        # get summarization path that ignores finished nodes
        summ_mats = get_unfinished_nodes_summ_mat(job_dags)

        # invoke learning model
        node_act_probs, job_act_probs = \
            self.predict(node_inputs,job_inputs,
                         node_valid_mask,job_valid_mask,
                         gcn_mats, gcn_masks, summ_mats,
                         running_dags_mat, dag_summ_backward_map)

        return node_act_probs, job_act_probs, \
               node_inputs, job_inputs, \
               node_valid_mask, job_valid_mask, \
               gcn_mats, gcn_masks, summ_mats, \
               running_dags_mat, dag_summ_backward_map, \
               job_dags_changed

