import tensorflow as tf
import pandas as pd
import numpy as np
import os
import h5py
from datetime import datetime

# 直接使用 tf.keras 的所有组件
max_norm = tf.keras.constraints.max_norm
layers = tf.keras.layers
keras = tf.keras

from dataset import Graph_Regression_Dataset_3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from model import PredictModel, Hybrid3DBertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def create_directories(fold_dir):
    """确保目录存在"""
    os.makedirs(fold_dir, exist_ok=True)


def load_pretrained_weights(model, weights_path, num_layers, d_model, dff, num_heads, vocab_size):
    """加载预训练权重到模型的encoder部分"""
    print(f"尝试加载预训练权重: {weights_path}")

    if not os.path.exists(weights_path):
        print(f"⚠️ 预训练权重未找到: {weights_path}")
        print("从头开始训练...")
        return False

    try:
        # 创建预训练模型结构
        pretrained_model = Hybrid3DBertModel(
            num_layers=num_layers,
            d_model=d_model,
            dff=dff,
            num_heads=num_heads,
            vocab_size=vocab_size
        )

        # 构建预训练模型
        dummy_x = tf.ones((1, 20), dtype=tf.int64)
        dummy_coords = tf.ones((1, 20, 3), dtype=tf.float32)
        dummy_adj = tf.ones((1, 20, 20), dtype=tf.float32)
        dummy_mask = tf.ones((1, 1, 1, 20), dtype=tf.float32)
        _ = pretrained_model(dummy_x, coords=dummy_coords, adjoin_matrix=dummy_adj, mask=dummy_mask, training=False)

        # 加载预训练权重
        pretrained_model.load_weights(weights_path)
        print("✅ 预训练模型权重加载成功")

        # 将预训练模型的权重复制到当前模型的encoder
        encoder = model.encoder
        copied_layers = 0

        for pretrained_layer, current_layer in zip(pretrained_model.layers, encoder.layers):
            if hasattr(pretrained_layer, 'get_weights') and hasattr(current_layer, 'set_weights'):
                try:
                    weights = pretrained_layer.get_weights()
                    if len(weights) > 0:
                        current_layer.set_weights(weights)
                        print(f"  ✅ 复制权重: {pretrained_layer.name}")
                        copied_layers += 1
                except Exception as e:
                    print(f"  ⚠️ 复制权重失败 {pretrained_layer.name}: {e}")

        print(f"✅ 成功复制 {copied_layers} 个层的预训练权重!")
        return True

    except Exception as e:
        print(f"❌ 加载预训练权重错误: {e}")
        print("从头开始训练...")
        return False


def save_epoch_results(fold_dir, epoch_results, fold, epoch):
    """保存每个epoch的结果到CSV文件"""
    # 为当前fold创建结果DataFrame
    epoch_df = pd.DataFrame(epoch_results)

    # 保存到CSV文件
    csv_path = f'{fold_dir}/epoch_results_fold_{fold + 1}.csv'
    epoch_df.to_csv(csv_path, index=False)

    # 同时保存为Excel格式
    excel_path = f'{fold_dir}/epoch_results_fold_{fold + 1}.xlsx'
    epoch_df.to_excel(excel_path, index=False)

    print(f"✅ Fold {fold + 1} - Epoch {epoch} 结果已保存到: {csv_path}")


def save_predictions_to_excel(fold_dir, y_true, y_pred, dataset_type, fold, additional_info=None):
    """保存预测结果到Excel文件"""
    # 创建包含真实值和预测值的DataFrame
    predictions_df = pd.DataFrame({
        'True_Value': y_true,
        'Predicted_Value': y_pred,
        'Absolute_Error': np.abs(y_true - y_pred),
        'Relative_Error': np.abs((y_true - y_pred) / y_true) * 100  # 百分比相对误差
    })

    # 计算统计指标
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 创建统计信息DataFrame
    stats_df = pd.DataFrame({
        'Metric': ['R²', 'MSE', 'MAE', 'RMSE', '样本数量'],
        'Value': [r2, mse, mae, np.sqrt(mse), len(y_true)]
    })

    # 如果有额外信息，添加到统计表中
    if additional_info:
        for key, value in additional_info.items():
            stats_df = pd.concat([stats_df, pd.DataFrame({'Metric': [key], 'Value': [value]})], ignore_index=True)

    # 保存到Excel文件（多个sheet）
    excel_path = f'{fold_dir}/{dataset_type}_predictions_fold_{fold + 1}.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    print(f"✅ {dataset_type}预测结果已保存到: {excel_path}")
    print(f"   - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    return predictions_df, stats_df


def kfold_main(k_folds=5, seed=42):
    """
    K折交叉验证主函数
    """
    small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights', 'addH': True}
    medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
    medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3',
               'addH': True}
    large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights', 'addH': True}
    medium_balanced = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_balanced',
                       'addH': True}
    medium_without_H = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H',
                        'addH': False}

    arch = large  ## small 3 4 128   medium: 6 8  256     large:  12 12 576
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']
    # 模型配置
    medium  = {
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
    vocab_size = 100

    # 存储每个fold的结果
    fold_results = []

    for fold in range(k_folds):
        print(f"\n{'=' * 60}")
        print(f"开始训练 Fold {fold + 1}/{k_folds} (种子: {seed})")
        print(f"{'=' * 60}")

        # 清除会话
        keras.backend.clear_session()
        tf.random.set_seed(seed + fold)
        np.random.seed(seed + fold)

        # 创建fold目录（按种子区分）
        seed_dir = f'kfold_results/seed_{seed}'
        fold_dir = f'{seed_dir}/fold_{fold + 1}'
        create_directories(fold_dir)

        # 存储每个epoch的结果
        epoch_results = []

        # 加载3D数据集 - 使用K折划分
        graph_dataset = Graph_Regression_Dataset_3D(
            path='3Dsmilespre-log.csv',

            smiles_field='SMILES',
            label_field='logp',
            structure_paths_field='structure_paths',
            normalize=True,
            max_len=120,
            addH=addH
        )

        # 获取当前fold的数据
        train_dataset, test_dataset, val_dataset = graph_dataset.get_k_fold_data(
            k=k_folds, fold=fold, seed=seed
        )
        value_range = graph_dataset.value_range

        # 调试：检查数据集是否为空
        print("检查数据集状态...")
        print(f"训练集类型: {type(train_dataset)}")

        # 尝试获取一个批次的数据来测试
        try:
            print("尝试获取训练集的一个批次...")
            for i, sample in enumerate(train_dataset.take(1)):
                x, adjoin_matrix, coords, y = sample
                print(f"成功获取批次 {i}:")
                print(f"  x shape: {x.shape}")
                print(f"  adjoin_matrix shape: {adjoin_matrix.shape}")
                print(f"  coords shape: {coords.shape}")
                print(f"  y shape: {y.shape}")
                print(f"  y values: {y.numpy()[:5]}")  # 显示前5个值
                break
        except Exception as e:
            print(f"❌ 获取训练数据失败: {e}")
            continue  # 跳过这个fold

        # 获取数据形状示例
        try:
            for sample in train_dataset.take(1):
                x, adjoin_matrix, coords, y = sample
                break
        except Exception as e:
            print(f"❌ 无法获取数据形状: {e}")
            continue

        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]

        # 创建回归模型 - 使用 PredictModel
        print("创建模型...")
        model = PredictModel(
            num_layers=num_layers,
            d_model=d_model,
            dff=dff,
            num_heads=num_heads,
            vocab_size=vocab_size,
            dropout_rate=0.1,
            dense_dropout=0.15,
            use_3d=use_3d
        )

        # 构建模型 - 添加调试
        print("构建模型...")
        dummy_x = tf.ones((1, 20), dtype=tf.int64)
        dummy_coords = tf.ones((1, 20, 3), dtype=tf.float32)
        dummy_adj = tf.ones((1, 20, 20), dtype=tf.float32)
        dummy_mask = tf.ones((1, 1, 1, 20), dtype=tf.float32)

        try:
            _ = model(dummy_x, coords=dummy_coords, adjoin_matrix=dummy_adj, mask=dummy_mask, training=False)
            print("✅ 模型构建成功")
        except Exception as e:
            print(f"❌ 模型构建失败: {e}")
            continue

        print("下游任务模型构建完成!")

        # 加载预训练权重
        weights_path = f"{arch['path']}/bert_weights_{arch['name']}_epoch_{140}.weights.h5"
        load_pretrained_weights(model, weights_path, num_layers, d_model, dff, num_heads, vocab_size)

        # 优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # 训练循环 - 添加更多调试信息
        best_r2 = -10
        best_mse = float('inf')
        best_epoch = 0
        stopping_monitor = 0

        print(f"Fold {fold + 1} - 开始3D临界温度预测训练...")

        for epoch in range(300):
            print(f"Epoch {epoch} 开始...")

            # 训练阶段
            total_loss = 0
            batch_count = 0
            mse_object = tf.keras.metrics.MeanSquaredError()
            mae_object = tf.keras.metrics.MeanAbsoluteError()

            try:
                for batch_num, batch in enumerate(train_dataset):
                    try:
                        x, adjoin_matrix, coords, y = batch

                        with tf.GradientTape() as tape:
                            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                            mask = seq[:, tf.newaxis, tf.newaxis, :]

                            preds = model(
                                x=x,
                                coords=coords,
                                adjoin_matrix=adjoin_matrix,
                                mask=mask,
                                training=True
                            )
                            loss = tf.reduce_mean(tf.square(y - preds))
                            grads = tape.gradient(loss, model.trainable_variables)
                            optimizer.apply_gradients(zip(grads, model.trainable_variables))

                            mse_object.update_state(y, preds)
                            mae_object.update_state(y, preds)
                            total_loss += loss.numpy()
                            batch_count += 1

                        # 每10个批次显示一次进度
                        if batch_count % 10 == 0:
                            print(f'Epoch {epoch} - 批次 {batch_count}, 当前损失: {loss.numpy():.4f}')

                    except tf.errors.OutOfRangeError:
                        print(f"数据集在epoch {epoch}结束，共处理 {batch_count} 个批次")
                        break
                    except Exception as e:
                        print(f"❌ 批次处理失败: {e}")
                        break

            except Exception as e:
                print(f"❌ 训练循环失败: {e}")
                break

            if batch_count == 0:
                print(f"Epoch {epoch}: 没有训练数据，跳过")
                continue

            avg_loss = total_loss / batch_count
            train_mse = mse_object.result().numpy()
            train_mae = mae_object.result().numpy()

            # 每个epoch都输出
            print(f'Fold {fold + 1} - Epoch: {epoch:3d}, Loss: {avg_loss:.4f}, '
                  f'Train MSE: {train_mse * (value_range ** 2):.4f}, '
                  f'Train MAE: {train_mae * value_range:.4f}, Batches: {batch_count}')

            # 验证阶段
            y_true_val = []
            y_preds_val = []
            val_batch_count = 0

            for batch in val_dataset:
                try:
                    x, adjoin_matrix, coords, y = batch

                    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                    mask = seq[:, tf.newaxis, tf.newaxis, :]

                    preds = model(
                        x=x,
                        coords=coords,
                        adjoin_matrix=adjoin_matrix,
                        mask=mask,
                        training=False
                    )
                    y_true_val.append(y.numpy())
                    y_preds_val.append(preds.numpy())
                    val_batch_count += 1

                except tf.errors.OutOfRangeError:
                    break

            if len(y_true_val) == 0:
                print(f"Epoch {epoch}: 验证集为空，跳过验证")
                # 仍然记录训练指标
                epoch_result = {
                    'epoch': epoch,
                    'fold': fold + 1,
                    'seed': seed,
                    'train_loss': avg_loss,
                    'train_mse': train_mse * (value_range ** 2),
                    'train_mae': train_mae * value_range,
                    'val_r2': None,
                    'val_mse': None,
                    'val_mae': None,
                    'is_best': False
                }
                epoch_results.append(epoch_result)
                continue

            y_true_val = np.concatenate(y_true_val, axis=0).reshape(-1)
            y_preds_val = np.concatenate(y_preds_val, axis=0).reshape(-1)

            # 反标准化
            y_true_val_orig = (y_true_val + 0.5) * value_range + graph_dataset.min
            y_preds_val_orig = (y_preds_val + 0.5) * value_range + graph_dataset.min

            val_r2 = r2_score(y_true_val_orig, y_preds_val_orig)
            val_mse = mean_squared_error(y_true_val_orig, y_preds_val_orig)
            val_mae = mean_absolute_error(y_true_val_orig, y_preds_val_orig)

            if epoch % 5 == 0:
                print(f'Fold {fold + 1} - Val R²: {val_r2:.4f}, Val MSE: {val_mse:.4f}, '
                      f'Val MAE: {val_mae:.4f}, Val Batches: {val_batch_count}')

            # 记录当前epoch的结果
            is_best = False
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_mse = val_mse
                best_epoch = epoch
                stopping_monitor = 0
                is_best = True

                # 保存fold最佳模型
                model.save_weights(f'{fold_dir}/best_model.weights.h5')

                # 保存验证集预测结果（npy格式）
                np.save(f'{fold_dir}/val_predictions.npy',
                        [y_true_val_orig, y_preds_val_orig])

                # 保存验证集预测结果到Excel（新增）
                additional_info = {
                    'Fold': fold + 1,
                    'Seed': seed,
                    'Best_Epoch': epoch,
                    'Value_Range': value_range,
                    'Min_Value': graph_dataset.min
                }
                save_predictions_to_excel(
                    fold_dir, y_true_val_orig, y_preds_val_orig,
                    'validation', fold, additional_info
                )

                print(f'🎯 Fold {fold + 1} - 新最佳模型保存在epoch {epoch}! R²: {val_r2:.4f}')
            else:
                stopping_monitor += 1

            # 保存epoch结果
            epoch_result = {
                'epoch': epoch,
                'fold': fold + 1,
                'seed': seed,
                'train_loss': avg_loss,
                'train_mse': train_mse * (value_range ** 2),
                'train_mae': train_mae * value_range,
                'val_r2': val_r2,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'is_best': is_best,
                'batches_processed': batch_count,
                'val_batches_processed': val_batch_count
            }
            epoch_results.append(epoch_result)

            # 每5个epoch保存一次结果
            if epoch % 5 == 0:
                save_epoch_results(fold_dir, epoch_results, fold, epoch)

            if stopping_monitor > 30:  # 增加早停耐心
                print(f'🚨 Fold {fold + 1} - 早停于epoch {epoch}')
                break

        # 训练结束后保存完整的epoch结果
        save_epoch_results(fold_dir, epoch_results, fold, epoch)

        # 测试阶段 - 使用最佳模型
        print(f"\nFold {fold + 1} - 在测试集上评估...")

        # 加载最佳模型
        model_path = f'{fold_dir}/best_model.weights.h5'
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print(f"✅ 加载Fold {fold + 1}的最佳模型 (R²: {best_r2:.4f})")

        y_true_test = []
        y_preds_test = []

        for batch in test_dataset:
            try:
                x, adjoin_matrix, coords, y = batch

                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]

                preds = model(
                    x=x,
                    coords=coords,
                    adjoin_matrix=adjoin_matrix,
                    mask=mask,
                    training=False
                )
                y_true_test.append(y.numpy())
                y_preds_test.append(preds.numpy())
            except tf.errors.OutOfRangeError:
                break

        if len(y_true_test) == 0:
            print("测试集为空!")
            test_r2, test_mse, test_mae = 0, 0, 0
            y_true_test_orig = np.array([])
            y_preds_test_orig = np.array([])
        else:
            y_true_test = np.concatenate(y_true_test, axis=0).reshape(-1)
            y_preds_test = np.concatenate(y_preds_test, axis=0).reshape(-1)

            # 反标准化
            y_true_test_orig = (y_true_test + 0.5) * value_range + graph_dataset.min
            y_preds_test_orig = (y_preds_test + 0.5) * value_range + graph_dataset.min

            test_r2 = r2_score(y_true_test_orig, y_preds_test_orig)
            test_mse = mean_squared_error(y_true_test_orig, y_preds_test_orig)
            test_mae = mean_absolute_error(y_true_test_orig, y_preds_test_orig)

        print(f'Fold {fold + 1} - 测试结果: R²: {test_r2:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')

        # 保存测试结果（npy格式）
        np.save(f'{fold_dir}/test_predictions.npy',
                [y_true_test_orig, y_preds_test_orig])

        # 保存测试集预测结果到Excel（新增）
        if len(y_true_test_orig) > 0:
            additional_info = {
                'Fold': fold + 1,
                'Seed': seed,
                'Best_Epoch': best_epoch,
                'Value_Range': value_range,
                'Min_Value': graph_dataset.min,
                'Test_R2': test_r2,
                'Test_MSE': test_mse,
                'Test_MAE': test_mae
            }
            save_predictions_to_excel(
                fold_dir, y_true_test_orig, y_preds_test_orig,
                'test', fold, additional_info
            )

        # 保存fold结果
        fold_result = {
            'fold': fold + 1,
            'seed': seed,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'best_val_r2': best_r2,
            'best_epoch': best_epoch,
            'total_epochs': len(epoch_results)
        }
        fold_results.append(fold_result)

        # 保存当前fold的详细结果
        np.save(f'{fold_dir}/fold_result.npy', fold_result)

    return fold_results


def analyze_kfold_results(fold_results, seed):
    """分析K折交叉验证结果"""
    if not fold_results:
        print("没有结果可分析")
        return

    print(f"\n{'=' * 60}")
    print(f"K折交叉验证结果分析 (种子: {seed})")
    print(f"{'=' * 60}")

    # 提取各项指标
    test_r2_scores = [result['test_r2'] for result in fold_results]
    test_mse_scores = [result['test_mse'] for result in fold_results]
    test_mae_scores = [result['test_mae'] for result in fold_results]
    best_val_r2_scores = [result['best_val_r2'] for result in fold_results]

    # 打印每折结果
    for i, result in enumerate(fold_results):
        print(f"Fold {i + 1}: "
              f"Test R²={result['test_r2']:.4f}, "
              f"Test MSE={result['test_mse']:.4f}, "
              f"Test MAE={result['test_mae']:.4f}, "
              f"Best Val R²={result['best_val_r2']:.4f}")

    # 打印统计结果
    print(f"\n总体统计 (种子 {seed}):")
    print(f"Test R²: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
    print(f"Test MSE: {np.mean(test_mse_scores):.4f} ± {np.std(test_mse_scores):.4f}")
    print(f"Test MAE: {np.mean(test_mae_scores):.4f} ± {np.std(test_mae_scores):.4f}")
    print(f"Best Val R²: {np.mean(best_val_r2_scores):.4f} ± {np.std(best_val_r2_scores):.4f}")

    return {
        'seed': seed,
        'fold_results': fold_results,
        'mean_test_r2': np.mean(test_r2_scores),
        'std_test_r2': np.std(test_r2_scores),
        'mean_test_mse': np.mean(test_mse_scores),
        'std_test_mse': np.std(test_mse_scores),
        'mean_test_mae': np.mean(test_mae_scores),
        'std_test_mae': np.std(test_mae_scores),
        'mean_val_r2': np.mean(best_val_r2_scores),
        'std_val_r2': np.std(best_val_r2_scores)
    }


def create_summary_table(seeds, k_folds=5):
    """创建所有种子和fold的汇总表格"""
    all_epoch_results = []
    all_fold_results = []
    all_predictions = []

    for seed in seeds:
        for fold in range(k_folds):
            fold_dir = f'kfold_results/seed_{seed}/fold_{fold + 1}'
            csv_path = f'{fold_dir}/epoch_results_fold_{fold + 1}.csv'

            if os.path.exists(csv_path):
                fold_df = pd.read_csv(csv_path)
                all_epoch_results.append(fold_df)
                print(f"✅ 加载种子 {seed} - Fold {fold + 1} 的epoch结果: {len(fold_df)} 行")
            else:
                print(f"⚠️ 种子 {seed} - Fold {fold + 1} 的epoch结果文件未找到: {csv_path}")

            # 加载fold结果
            fold_result_path = f'{fold_dir}/fold_result.npy'
            if os.path.exists(fold_result_path):
                fold_result = np.load(fold_result_path, allow_pickle=True).item()
                all_fold_results.append(fold_result)
                print(f"✅ 加载种子 {seed} - Fold {fold + 1} 的fold结果")

            # 加载预测结果（新增）
            test_pred_path = f'{fold_dir}/test_predictions.xlsx'
            val_pred_path = f'{fold_dir}/validation_predictions.xlsx'

            if os.path.exists(test_pred_path):
                test_pred_df = pd.read_excel(test_pred_path, sheet_name='Predictions')
                test_pred_df['Seed'] = seed
                test_pred_df['Fold'] = fold + 1
                test_pred_df['Dataset'] = 'Test'
                all_predictions.append(test_pred_df)
                print(f"✅ 加载种子 {seed} - Fold {fold + 1} 的测试集预测结果")

            if os.path.exists(val_pred_path):
                val_pred_df = pd.read_excel(val_pred_path, sheet_name='Predictions')
                val_pred_df['Seed'] = seed
                val_pred_df['Fold'] = fold + 1
                val_pred_df['Dataset'] = 'Validation'
                all_predictions.append(val_pred_df)
                print(f"✅ 加载种子 {seed} - Fold {fold + 1} 的验证集预测结果")

    if all_epoch_results:
        # 合并所有epoch的结果
        summary_df = pd.concat(all_epoch_results, ignore_index=True)

        # 保存汇总表格
        current_date = datetime.now().strftime("%Y%m%d")
        summary_csv_path = f'kfold_results/{current_date}_all_seeds_epochs_summary.csv'
        summary_excel_path = f'kfold_results/{current_date}_all_seeds_epochs_summary.xlsx'

        summary_df.to_csv(summary_csv_path, index=False)
        summary_df.to_excel(summary_excel_path, index=False)

        print(f"\n✅ 所有种子的epoch汇总表格已保存:")
        print(f"   CSV格式: {summary_csv_path}")
        print(f"   Excel格式: {summary_excel_path}")
        print(f"   总记录数: {len(summary_df)}")

    if all_fold_results:
        # 创建fold结果汇总
        fold_summary_df = pd.DataFrame(all_fold_results)
        fold_csv_path = f'kfold_results/{current_date}_all_seeds_folds_summary.csv'
        fold_excel_path = f'kfold_results/{current_date}_all_seeds_folds_summary.xlsx'

        fold_summary_df.to_csv(fold_csv_path, index=False)
        fold_summary_df.to_excel(fold_excel_path, index=False)

        print(f"✅ 所有种子的fold汇总表格已保存:")
        print(f"   CSV格式: {fold_csv_path}")
        print(f"   Excel格式: {fold_excel_path}")

    if all_predictions:
        # 创建预测结果汇总（新增）
        predictions_summary_df = pd.concat(all_predictions, ignore_index=True)
        pred_csv_path = f'kfold_results/{current_date}_all_predictions_summary.csv'
        pred_excel_path = f'kfold_results/{current_date}_all_predictions_summary.xlsx'

        predictions_summary_df.to_csv(pred_csv_path, index=False)
        predictions_summary_df.to_excel(pred_excel_path, index=False)

        print(f"✅ 所有预测结果汇总表格已保存:")
        print(f"   CSV格式: {pred_csv_path}")
        print(f"   Excel格式: {pred_excel_path}")
        print(f"   总预测样本数: {len(predictions_summary_df)}")

    return summary_df if all_epoch_results else None


def analyze_all_seeds_results(all_seeds_results):
    """分析所有种子的总体结果"""
    print(f"\n{'=' * 80}")
    print("所有随机种子的总体结果分析")
    print(f"{'=' * 80}")

    # 提取各种子结果
    seeds = [result['seed'] for result in all_seeds_results]
    mean_test_r2 = [result['mean_test_r2'] for result in all_seeds_results]
    mean_test_mae = [result['mean_test_mae'] for result in all_seeds_results]
    mean_val_r2 = [result['mean_val_r2'] for result in all_seeds_results]

    # 打印每个种子的结果
    for result in all_seeds_results:
        print(f"种子 {result['seed']}:")
        print(f"  测试集 - R²: {result['mean_test_r2']:.4f} ± {result['std_test_r2']:.4f}")
        print(f"  测试集 - MAE: {result['mean_test_mae']:.4f} ± {result['std_test_mae']:.4f}")
        print(f"  验证集 - R²: {result['mean_val_r2']:.4f} ± {result['std_val_r2']:.4f}")

    # 计算总体统计
    overall_test_r2 = np.mean(mean_test_r2)
    overall_test_r2_std = np.std(mean_test_r2)
    overall_test_mae = np.mean(mean_test_mae)
    overall_test_mae_std = np.std(mean_test_mae)
    overall_val_r2 = np.mean(mean_val_r2)
    overall_val_r2_std = np.std(mean_val_r2)

    print(f"\n总体统计 ({len(seeds)} 个种子):")
    print(f"测试集 R²: {overall_test_r2:.4f} ± {overall_test_r2_std:.4f}")
    print(f"测试集 MAE: {overall_test_mae:.4f} ± {overall_test_mae_std:.4f}")
    print(f"验证集 R²: {overall_val_r2:.4f} ± {overall_val_r2_std:.4f}")

    return {
        'seeds': seeds,
        'overall_test_r2': overall_test_r2,
        'overall_test_r2_std': overall_test_r2_std,
        'overall_test_mae': overall_test_mae,
        'overall_test_mae_std': overall_test_mae_std,
        'overall_val_r2': overall_val_r2,
        'overall_val_r2_std': overall_val_r2_std,
        'detailed_results': all_seeds_results
    }


if __name__ == "__main__":
    # 定义多个随机种子
    seeds = [7, 17, 27]  # 添加随机种子7和17
    k_folds = 5

    print("开始3D临界温度预测的K折交叉验证...")
    print(f"配置: {k_folds}折, 随机种子: {seeds}")

    # 创建结果目录
    create_directories('kfold_results')

    # 存储所有种子的结果
    all_seeds_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"开始使用随机种子: {seed}")
        print(f"{'=' * 60}")

        # 运行K折交叉验证
        fold_results = kfold_main(k_folds=k_folds, seed=seed)

        # 分析当前种子的结果
        seed_results = analyze_kfold_results(fold_results, seed)
        all_seeds_results.append(seed_results)

        # 保存当前种子的结果
        np.save(f'kfold_results/seed_{seed}/seed_{seed}_results.npy', seed_results)

        print(f"\n✅ 随机种子 {seed} 完成!")

    # 创建所有种子的汇总表格
    summary_table = create_summary_table(seeds, k_folds)

    # 分析所有种子的总体结果
    final_results = analyze_all_seeds_results(all_seeds_results)

    # 保存最终结果
    current_date = datetime.now().strftime("%Y%m%d")
    final_results_path = f'kfold_results/{current_date}_final_kfold_results.npy'
    np.save(final_results_path, final_results)

    # 保存为CSV
    final_summary_df = pd.DataFrame([{
        'seeds': str(seeds),
        'overall_test_r2': final_results['overall_test_r2'],
        'overall_test_r2_std': final_results['overall_test_r2_std'],
        'overall_test_mae': final_results['overall_test_mae'],
        'overall_test_mae_std': final_results['overall_test_mae_std'],
        'overall_val_r2': final_results['overall_val_r2'],
        'overall_val_r2_std': final_results['overall_val_r2_std']
    }])

    final_csv_path = f'kfold_results/{current_date}_final_summary.csv'
    final_summary_df.to_csv(final_csv_path, index=False)

    print(f"\n✅ 最终结果已保存到:")
    print(f"   {final_results_path}")
    print(f"   {final_csv_path}")
