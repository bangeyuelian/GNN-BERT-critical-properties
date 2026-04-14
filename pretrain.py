import tensorflow as tf
from model import Hybrid3DBertModel
from dataset import Graph_Bert_Dataset_3D
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

optimizer = tf.keras.optimizers.Adam(1e-4)
# small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights','addH':True}
# medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights','addH':True}
# medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3','addH':True}
# large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights','addH':True}
# medium_balanced = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_balanced','addH':True}
# medium_without_H = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_without_H','addH':False}
#
# arch = medium3      ## small 3 4 128   medium: 6 8  256     large:  12 12 576
# num_layers = arch['num_layers']
# num_heads =  arch['num_heads']
# d_model =  arch['d_model']
# addH = arch['addH']

medium = {
    'name': 'medium',
    'num_layers': 6,
    'num_heads': 8,
    'd_model': 256,
    'path': 'medium_3d_weights',
    'addH': True,
    'use_3d': True
}

arch = medium
num_layers = arch['num_layers']
num_heads = arch['num_heads']
d_model = arch['d_model']
addH = arch['addH']
use_3d = arch['use_3d']

dff = d_model * 2
vocab_size = 28
dropout_rate = 0.1

# 使用混合模型
model = Hybrid3DBertModel(
    num_layers=num_layers,
    d_model=d_model,
    dff=dff,
    num_heads=num_heads,
    vocab_size=vocab_size
)

# 使用新的数据集类
train_dataset, test_dataset = Graph_Bert_Dataset_3D(
    path='3Dsmilespre-log.csv',
    smiles_field='SMILES',
    structure_paths_field='structure_paths',
    addH=addH,
    use_3d=use_3d
).get_data()


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(x, adjoin_matrix, coords, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    with tf.GradientTape() as tape:
        predictions = model(
            x,
            coords=coords,
            adjoin_matrix=adjoin_matrix,
            mask=mask,
            training=True
        )
        loss = loss_function(y, predictions, sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(y, predictions, sample_weight=char_weight)


def test_step(x, adjoin_matrix, coords, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]

    predictions = model(
        x,
        coords=coords,
        adjoin_matrix=adjoin_matrix,
        mask=mask,
        training=False
    )
    loss = loss_function(y, predictions, sample_weight=char_weight)

    test_loss.update_state(loss)
    test_accuracy.update_state(y, predictions, sample_weight=char_weight)


# 确保保存目录存在
if not os.path.exists(arch['path']):
    os.makedirs(arch['path'])

# 用于记录每个epoch的性能
epoch_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
    'epochs': []
}

# 创建Excel记录器
current_date = datetime.now().strftime("%Y%m%d")
excel_filename = f"pretrain_log_{current_date}.xlsx"
excel_path = os.path.join(arch['path'], excel_filename)

# 初始化Excel数据框
excel_data = {
    'Epoch': [],
    'Train_Loss': [],
    'Train_Accuracy': [],
    'Test_Loss': [],
    'Test_Accuracy': [],
    'Best_Model': []
}

print("开始训练...")
for epoch in range(1000):
    start = time.time()

    # 重置指标
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    # 训练阶段
    print(f"Epoch {epoch + 1}/10 - 训练中...")
    for (batch, (x, adjoin_matrix, coords, y, char_weight)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, coords, y, char_weight)

        if batch % 500 == 0:
            print('Epoch {} Batch {} 训练损失: {:.4f}, 训练准确率: {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    # 测试阶段
    print(f"Epoch {epoch + 1}/10 - 测试中...")
    test_batch_count = 0
    for (batch, (x, adjoin_matrix, coords, y, char_weight)) in enumerate(test_dataset):
        test_step(x, adjoin_matrix, coords, y, char_weight)
        test_batch_count += 1
        if batch % 100 == 0:
            print('Epoch {} 测试批次 {}'.format(epoch + 1, batch))

    # 记录当前epoch的性能
    current_train_loss = train_loss.result().numpy()
    current_train_accuracy = train_accuracy.result().numpy()
    current_test_loss = test_loss.result().numpy()
    current_test_accuracy = test_accuracy.result().numpy()

    epoch_history['train_loss'].append(current_train_loss)
    epoch_history['train_accuracy'].append(current_train_accuracy)
    epoch_history['test_loss'].append(current_test_loss)
    epoch_history['test_accuracy'].append(current_test_accuracy)
    epoch_history['epochs'].append(epoch + 1)

    # 记录到Excel数据框（暂时不标记最佳模型）
    excel_data['Epoch'].append(epoch + 1)
    excel_data['Train_Loss'].append(current_train_loss)
    excel_data['Train_Accuracy'].append(current_train_accuracy)
    excel_data['Test_Loss'].append(current_test_loss)
    excel_data['Test_Accuracy'].append(current_test_accuracy)
    excel_data['Best_Model'].append('')  # 初始为空，训练结束后再标记

    print('=' * 60)
    print('Epoch {} 完成'.format(epoch + 1))
    print('训练损失: {:.4f}, 训练准确率: {:.4f}'.format(current_train_loss, current_train_accuracy))
    print('测试损失: {:.4f}, 测试准确率: {:.4f}'.format(current_test_loss, current_test_accuracy))
    print('耗时: {:.2f} 秒'.format(time.time() - start))
    print('=' * 60)

    # 保存权重
    weights_filename = f'bert_weights_{arch["name"]}_epoch_{epoch + 1}.weights.h5'
    weights_path = os.path.join(arch['path'], weights_filename)
    model.save_weights(weights_path)
    print('模型权重已保存: {}'.format(weights_path))
    print()

# 训练结束后分析最佳epoch
print('\n' + '=' * 80)
print('训练完成！性能分析:')
print('=' * 80)

# 找到最佳测试准确率和最小测试损失的epoch
best_accuracy_epoch = epoch_history['epochs'][np.argmax(epoch_history['test_accuracy'])]
best_accuracy = np.max(epoch_history['test_accuracy'])
best_loss_epoch = epoch_history['epochs'][np.argmin(epoch_history['test_loss'])]
best_loss = np.min(epoch_history['test_loss'])

print('最佳测试准确率:')
print('  Epoch: {}, 准确率: {:.4f}'.format(best_accuracy_epoch, best_accuracy))
print('最小测试损失:')
print('  Epoch: {}, 损失: {:.4f}'.format(best_loss_epoch, best_loss))

# 在Excel数据中标记最佳模型
for i in range(len(excel_data['Epoch'])):
    if excel_data['Epoch'][i] == best_accuracy_epoch:
        excel_data['Best_Model'][i] = '★ Best Model ★'

# 保存Excel文件
df = pd.DataFrame(excel_data)
df.to_excel(excel_path, index=False)
print(f'训练日志已保存到Excel: {excel_path}')

# 打印所有epoch的详细性能
print('\n所有Epoch详细性能:')
print('Epoch\t训练损失\t训练准确率\t测试损失\t测试准确率')
print('-' * 80)
for i in range(len(epoch_history['epochs'])):
    epoch_num = epoch_history['epochs'][i]
    train_l = epoch_history['train_loss'][i]
    train_acc = epoch_history['train_accuracy'][i]
    test_l = epoch_history['test_loss'][i]
    test_acc = epoch_history['test_accuracy'][i]
    print('{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(
        epoch_num, train_l, train_acc, test_l, test_acc))

# 保存训练历史
history_filename = os.path.join(arch['path'], 'training_history.npy')
np.save(history_filename, epoch_history)
print(f'\n训练历史已保存: {history_filename}')

# 保存最佳模型（如果需要）
best_model_path = os.path.join(arch['path'], f'best_model_epoch_{best_accuracy_epoch}.weights.h5')
print(f'建议使用最佳模型: {best_model_path}')