import tensorflow as tf
from geometric_layers import GeometricGNN


def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    """
    q, k, v: (B, H, L, d)
    mask:    (B, 1, 1, L)
    adjoin_matrix: (B, H or 1, L, L)  —— 已 padding 到 full seq_len
    """

    # ---- QK^T ----
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, H, L, L)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)

    # ---- padding mask ----
    if mask is not None:
        logits += (mask * -1e9)

    # ---- graph bias (ONLY here) ----
    if adjoin_matrix is not None:
        # 强制 adjoin_matrix 与 logits 同 shape（防炸）
        adjoin_matrix = tf.cast(adjoin_matrix, logits.dtype)

        # 如果 adjoin_matrix 只有 1 个 head，自动 broadcast
        if adjoin_matrix.shape[1] == 1:
            adjoin_matrix = tf.tile(
                adjoin_matrix,
                [1, tf.shape(logits)[1], 1, 1]
            )

        # 安全断言（强烈建议保留）
        tf.debugging.assert_equal(
            tf.shape(logits),
            tf.shape(adjoin_matrix),
            message="Graph bias shape mismatch with attention logits"
        )

        logits += adjoin_matrix

    # ---- softmax ----
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # ---- attention output ----
    output = tf.matmul(attention_weights, v)  # (B, H, L, d)

    return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, adjoin_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 修正：将参数作为关键字参数传递
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            adjoin_matrix=adjoin_matrix
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=gelu),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, adjoin_matrix):
        """
        x: [B, seq_len, d_model]  (seq_len includes <global>)
        adjoin_matrix: [B, 1 or H, seq_len, seq_len]
        """

        # ---- remove <global> token (index 0) ----
        x_atom = x[:, 1:, :]                 # [B, L, d]
        mask_atom = mask[:, :, :, 1:]        # [B, 1, 1, L]

        # ---- align adjoin matrix ----
        if adjoin_matrix is not None:
            adjoin_atom = adjoin_matrix[:, :, 1:, 1:]  # [B, 1, L, L]

            if adjoin_atom.shape[1] == 1:
                adjoin_atom = tf.tile(
                    adjoin_atom,
                    [1, self.num_heads, 1, 1]
                )
        else:
            adjoin_atom = None

        # ---- graph-aware self-attention (atom-only) ----
        attn_output, attention_weights = self.mha(
            q=x_atom,
            k=x_atom,
            v=x_atom,
            mask=mask_atom,
            adjoin_matrix=adjoin_atom
        )

        attn_output = self.dropout1(attn_output, training=training)
        out1_atom = self.layernorm1(x_atom + attn_output)

        # ---- FFN ----
        ffn_output = self.ffn(out1_atom)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2_atom = self.layernorm2(out1_atom + ffn_output)

        # ---- prepend <global> token back ----
        out2 = tf.concat([x[:, :1, :], out2_atom], axis=1)

        return out2, attention_weights



class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, adjoin_matrix):
        adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            # 修正：将参数作为关键字参数传递
            x, attention_weights = self.enc_layers[i](
                x,
                training=training,  # 作为关键字参数
                mask=mask,  # 作为关键字参数
                adjoin_matrix=adjoin_matrix  # 作为关键字参数
            )
        return x


class Hybrid3DBertModel(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17, dropout_rate=0.1):
        super(Hybrid3DBertModel, self).__init__()
        self.d_model = d_model

        # 1. BERT编码器部分
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=vocab_size,
            rate=dropout_rate
        )

        # 2. 3D几何GNN部分
        geometric_dim = d_model // 2
        self.geometric_gnn = GeometricGNN(geometric_dim, num_layers=2, num_heads=4)
        self.coord_projection = tf.keras.layers.Dense(geometric_dim)

        # 3. 融合层
        self.fusion_gate = tf.keras.layers.Dense(d_model, activation='sigmoid')
        self.fusion_projection = tf.keras.layers.Dense(d_model)

        # 4. 输出层
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def compute_distance_matrix(self, coords):
        """计算3D距离矩阵"""
        # coords: [batch_size, seq_len, 3]
        batch_size = tf.shape(coords)[0]
        seq_len = tf.shape(coords)[1]

        # 使用更稳定的距离计算方法
        coords_expanded1 = tf.expand_dims(coords, 2)  # [batch_size, seq_len, 1, 3]
        coords_expanded2 = tf.expand_dims(coords, 1)  # [batch_size, 1, seq_len, 3]

        diff = coords_expanded1 - coords_expanded2  # [batch_size, seq_len, seq_len, 3]
        squared_diff = tf.reduce_sum(tf.square(diff), axis=-1)  # [batch_size, seq_len, seq_len]
        distances = tf.sqrt(squared_diff + 1e-8)  # [batch_size, seq_len, seq_len]

        return distances

    def call(self, x, coords=None, adjoin_matrix=None, mask=None, training=False):
        # ========== A. BERT序列处理 ==========
        bert_output = self.encoder(
            x=x,
            training=training,  # 关键字参数
            mask=mask,  # 关键字参数
            adjoin_matrix=adjoin_matrix  # 关键字参数
        )

        # ========== B. 3D几何GNN处理 ==========
        if coords is not None:
            # 计算3D距离矩阵
            distance_matrix = self.compute_distance_matrix(coords)

            # 从3D坐标初始化节点特征
            geometric_features = self.coord_projection(coords)

            # 3D GNN处理几何信息
            geometric_output = self.geometric_gnn(geometric_features, distance_matrix, mask)

            # 扩展几何特征维度到与BERT输出一致
            geometric_output_expanded = tf.keras.layers.Dense(self.d_model)(geometric_output)

            # ========== C. 特征融合 ==========
            # 关键修复：只融合原子部分，排除<global> token
            # BERT输出: [batch_size, seq_len, d_model]，其中seq_len包含<global> token
            # geometric_output: [batch_size, num_atoms, d_model]，其中num_atoms = seq_len - 1

            batch_size = tf.shape(bert_output)[0]
            bert_seq_len = tf.shape(bert_output)[1]
            geom_seq_len = tf.shape(geometric_output_expanded)[1]

            # 分离<global> token和原子tokens
            global_token = bert_output[:, 0:1, :]  # [batch_size, 1, d_model]
            atom_tokens = bert_output[:, 1:, :]  # [batch_size, seq_len-1, d_model]

            # 确保原子tokens数量与几何输出匹配
            min_len = tf.minimum(tf.shape(atom_tokens)[1], geom_seq_len)
            atom_tokens = atom_tokens[:, :min_len, :]
            geometric_output_expanded = geometric_output_expanded[:, :min_len, :]

            # 融合原子特征
            fusion_gate = self.fusion_gate(tf.concat([atom_tokens, geometric_output_expanded], axis=-1))
            fused_atoms = fusion_gate * atom_tokens + (1 - fusion_gate) * geometric_output_expanded
            fused_atoms = self.fusion_projection(fused_atoms)

            # 重新组合<global> token和融合后的原子tokens
            fused_output = tf.concat([global_token, fused_atoms], axis=1)

        else:
            # 如果没有3D坐标，只使用BERT输出
            fused_output = bert_output

        # ========== D. 输出层 ==========
        x = self.fc1(fused_output)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class PredictModel(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17,
                 dropout_rate=0.1, dense_dropout=0.1, use_3d=True):
        super(PredictModel, self).__init__()
        self.use_3d = use_3d

        # 使用混合模型作为编码器
        self.encoder = Hybrid3DBertModel(num_layers=num_layers, d_model=d_model,
                                         num_heads=num_heads, dff=dff, vocab_size=vocab_size,
                                         dropout_rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, x, coords=None, adjoin_matrix=None, mask=None, training=False):
        # 修复：所有参数都作为关键字参数传递
        if self.use_3d and coords is not None:
            encoded = self.encoder(
                x=x,
                coords=coords,
                adjoin_matrix=adjoin_matrix,
                mask=mask,
                training=training  # 明确使用关键字参数
            )
        else:
            # 如果没有3D坐标，传入None
            encoded = self.encoder(
                x=x,
                coords=None,
                adjoin_matrix=adjoin_matrix,
                mask=mask,
                training=training  # 明确使用关键字参数
            )

        x = encoded[:, 0, :]  # 取[CLS] token
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x