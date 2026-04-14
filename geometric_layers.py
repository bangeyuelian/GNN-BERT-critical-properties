import tensorflow as tf
import numpy as np


class DistanceAwareAttention(tf.keras.layers.Layer):
    """考虑3D距离的注意力机制"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, distance_matrix, mask=None):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 分割多头
        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # 缩放点积注意力
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [batch_size, num_heads, seq_len, seq_len]

        # 3D距离感知：添加距离权重
        distance_weights = tf.exp(-distance_matrix / 5.0)  # 距离衰减
        distance_weights = tf.expand_dims(distance_weights, 1)  # [batch_size, 1, seq_len, seq_len]
        distance_weights = tf.tile(distance_weights,
                                   [1, self.num_heads, 1, 1])  # [batch_size, num_heads, seq_len, seq_len]
        matmul_qk = matmul_qk + tf.math.log(distance_weights + 1e-9)

        # 缩放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 修复掩码形状问题
        if mask is not None:
            # 确保掩码形状正确 [batch_size, 1, 1, seq_len]
            if len(mask.shape) == 4 and mask.shape[-1] != seq_len:
                # 如果掩码长度不匹配，重新调整到正确的序列长度
                mask = tf.slice(mask, [0, 0, 0, 0], [-1, 1, 1, seq_len])

            # 扩展掩码到多头 [batch_size, num_heads, seq_len, seq_len]
            mask = tf.tile(mask, [1, self.num_heads, seq_len, 1])

            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)


class GeometricMessagePassing(tf.keras.layers.Layer):
    """3D几何消息传递层"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 距离相关的变换
        self.distance_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        # 消息网络
        self.message_net = tf.keras.layers.Dense(d_model)

    def call(self, node_features, distance_matrix, mask=None):
        batch_size = tf.shape(node_features)[0]
        seq_len = tf.shape(node_features)[1]

        # 构建邻接矩阵（基于3D距离）
        adjacency = tf.cast(distance_matrix < 6.0, tf.float32)  # 6Å阈值
        adjacency = tf.linalg.set_diag(adjacency, tf.zeros([batch_size, seq_len]))  # 移除自连接

        # 计算距离权重
        distance_weights = tf.exp(-distance_matrix / 5.0)

        # 修复：正确计算距离特征
        distance_features = self.distance_mlp(
            tf.expand_dims(distance_weights, -1))  # [batch_size, seq_len, seq_len, d_model]

        # 修复消息聚合 - 使用更高效的方法
        # 扩展节点特征用于消息传递
        node_features_expanded = tf.expand_dims(node_features, 1)  # [batch_size, 1, seq_len, d_model]
        node_features_expanded = tf.tile(node_features_expanded,
                                         [1, seq_len, 1, 1])  # [batch_size, seq_len, seq_len, d_model]

        # 应用邻接矩阵掩码
        adjacency_mask = tf.expand_dims(adjacency, -1)  # [batch_size, seq_len, seq_len, 1]

        # 组合邻居特征和距离特征
        neighbor_messages = node_features_expanded * adjacency_mask
        distance_messages = distance_features * adjacency_mask

        # 聚合所有邻居的消息
        aggregated_messages = tf.reduce_sum(neighbor_messages + distance_messages,
                                            axis=2)  # [batch_size, seq_len, d_model]

        return self.message_net(aggregated_messages)


class GeometricGNN(tf.keras.layers.Layer):
    """3D几何GNN模块"""

    def __init__(self, d_model, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append({
                'attention': DistanceAwareAttention(d_model, num_heads),
                'message_passing': GeometricMessagePassing(d_model),
                'layer_norm1': tf.keras.layers.LayerNormalization(epsilon=1e-6),
                'layer_norm2': tf.keras.layers.LayerNormalization(epsilon=1e-6),
            })

    def call(self, node_features, distance_matrix, mask=None):
        x = node_features

        for layer in self.layers:
            # 注意力子层
            attn_output = layer['attention'](x, x, x, distance_matrix, mask)
            x = layer['layer_norm1'](x + attn_output)

            # 消息传递子层
            msg_output = layer['message_passing'](x, distance_matrix, mask)
            x = layer['layer_norm2'](x + msg_output)

        return x