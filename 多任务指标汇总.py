
#config_multi_task.yaml

tasks:
  - name: Blood_stasis
    label_file: F:\graduate\tongue\xinxueguan1397\xueyu\xueyu_labelxuerad.csv
    task_column: label
    models: [Blood_stasis]

  - name: CVD
    label_file: F:\graduate\tongue\xinxueguan1397\xinxueguan\labelxuerad.csv
    task_column: label
    models: [CVD]

  - name: Dampness_syndrome
    label_file: F:\graduate\tongue\xinxueguan1397\shizheng\shizheng_labelxuerad.csv
    task_column: label
    models: [Dampness_syndrome]

  - name: Qi_deficiency
    label_file: F:\graduate\tongue\xinxueguan1397\qixu\qixu_labelxuerad.csv
    task_column: label
    models: [Qi_deficiency]

results_dir: D:\onekey\onekey_comp\comp2-结构化数据\final1\result
output_dir: ./multi_task_output

# 筛选的模型
sel_model: SVM


# 汇总所有模型交叉任务的结果

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from onekey_algo.custom.components import comp1 as okcomp
from onekey_algo.custom.components.metrics import analysis_pred_binary
from onekey_algo.custom.components.delong import delong_roc_test
from onekey_algo.custom.components.metrics import NRI, IDI
from onekey_algo.custom.components import stats
import matplotlib.gridspec as gridspec

def read_pred(path, model_name):
    """读取模型输出，返回 ID + 正类概率"""
    df = pd.read_csv(path)
    if 'label-1' in df.columns:
        return df[['ID', 'label-1']].rename(columns={'label-1': f'{model_name}_pred'})
    raise ValueError(f"{path} 缺少 label-1 列")

def main():
    # 读取配置
    with open('config_multi_task.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 8
    
    # 存储汇总指标和多任务ROC数据
    summary_metrics = []
    task_val_data = {}
    
    # 1. 处理每个任务
    for task in cfg['tasks']:
        task_name = task['name']
        label_file = task['label_file']
        task_col = task['task_column']
        print(f'🔍 Processing task: {task_name}')

        label_df = pd.read_csv(label_file)[['ID', task_col]]
        task_val_data[task_name] = {'y_true': None, 'model_scores': {}}

        for subset in ['train', 'val']:
            print(f'📊 Subset: {subset}')
            all_df = None
            
            # 合并模型预测结果
            for mn in cfg['compare_model']:
                path = os.path.join(cfg['results_dir'], f'{mn}_{subset}.csv')
                pred = read_pred(path, mn)
                if all_df is None:
                    all_df = pred
                else:
                    all_df = pd.merge(all_df, pred, on='ID', how='inner')
            
            all_df = pd.merge(all_df, label_df, on='ID', how='inner').dropna()
            y_true = np.array(all_df[task_col])
            y_scores = [np.array(all_df[f'{mn}_pred']) for mn in cfg['compare_model']]
            
            out_dir = os.path.join(cfg['output_dir'], task_name, subset)
            os.makedirs(out_dir, exist_ok=True)
            
            # 保存验证集数据用于汇总分析
            if subset == 'val':
                task_val_data[task_name]['y_true'] = y_true
                for i, mn in enumerate(cfg['compare_model']):
                    task_val_data[task_name]['model_scores'][mn] = y_scores[i]
            
            # ROC曲线
            fig = plt.figure(figsize=(8, 6))
            okcomp.draw_roc([y_true] * len(cfg['compare_model']), y_scores,
                            labels=cfg['compare_model'], title=f'{task_name} - {subset} AUC')
            plt.savefig(os.path.join(out_dir, 'auc.svg'), bbox_inches='tight')
            plt.close(fig)
            
            # 性能指标
            metrics = []
            youden = {}
            for mn, score in zip(cfg['compare_model'], y_scores):
                acc, auc, ci, tpr, tnr, ppv, npv, prec, rec, f1, th = analysis_pred_binary(y_true, score)
                ci_str = f'{ci[0]:.4f} - {ci[1]:.4f}'
                metrics.append((mn, acc, auc, ci_str, tpr, tnr, ppv, npv, prec, rec, f1, th, subset))
                youden[mn] = th
                
                # 收集验证集指标用于汇总
                if subset == 'val':
                    summary_metrics.append({
                        'Model': mn,
                        'Task': task_name,
                        'AUC': auc,
                        'Accuracy': acc,
                        'Sensitivity': tpr,
                        'Specificity': tnr,
                        'F1': f1
                    })
            
            metrics_df = pd.DataFrame(metrics, columns=[
                'Signature', 'Accuracy', 'AUC', '95% CI', 'Sensitivity', 'Specificity',
                'PPV', 'NPV', 'Precision', 'Recall', 'F1', 'Threshold', 'Cohort'
            ])
            metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            
            # Delong检验
            cm = np.full((len(cfg['compare_model']), len(cfg['compare_model'])), np.nan)
            for i, mni in enumerate(cfg['compare_model']):
                for j, mnj in enumerate(cfg['compare_model']):
                    if i > j:
                        cm[i, j] = delong_roc_test(y_true, all_df[f'{mni}_pred'], all_df[f'{mnj}_pred'])[0][0]
            cm_df = pd.DataFrame(cm[1:, :-1], index=cfg['compare_model'][1:], columns=cfg['compare_model'][:-1])
            
            fig = plt.figure(figsize=(8, 6))
            okcomp.draw_matrix(cm_df, annot=True, cmap='jet_r', cbar=True)
            plt.title(f'{task_name} - {subset} Delong')
            plt.savefig(os.path.join(out_dir, 'delong.svg'), bbox_inches='tight')
            plt.close(fig)
            
            # NRI
            cm = np.zeros((len(cfg['compare_model']), len(cfg['compare_model'])))
            for i, mni in enumerate(cfg['compare_model']):
                for j, mnj in enumerate(cfg['compare_model']):
                    cm[i, j] = NRI(all_df[f'{mni}_pred'] > youden[mni],
                                   all_df[f'{mnj}_pred'] > youden[mnj], y_true)
            cm_df = pd.DataFrame(cm, index=cfg['compare_model'], columns=cfg['compare_model'])
            
            fig = plt.figure(figsize=(8, 6))
            okcomp.draw_matrix(cm_df, annot=True, cmap='jet_r', cbar=True)
            plt.title(f'{task_name} - {subset} NRI')
            plt.savefig(os.path.join(out_dir, 'nri.svg'), bbox_inches='tight')
            plt.close(fig)
            
            # IDI
            idi = np.zeros((len(cfg['compare_model']), len(cfg['compare_model'])))
            idi_p = np.zeros_like(idi)
            for i, mni in enumerate(cfg['compare_model']):
                for j, mnj in enumerate(cfg['compare_model']):
                    idi[i, j], idi_p[i, j] = IDI(all_df[f'{mni}_pred'], all_df[f'{mnj}_pred'], y_true, with_p=True)
            
            for arr, name in zip([idi, idi_p], ['IDI', 'IDI_pvalue']):
                df = pd.DataFrame(arr, index=cfg['compare_model'], columns=cfg['compare_model'])
                
                fig = plt.figure(figsize=(8, 6))
                okcomp.draw_matrix(df, annot=True, cmap='jet_r', cbar=True)
                plt.title(f'{task_name} - {subset} {name}')
                plt.savefig(os.path.join(out_dir, f'{name.lower()}.svg'), bbox_inches='tight')
                plt.close(fig)
            
            # DCA
            fig = plt.figure(figsize=(8, 6))
            okcomp.plot_DCA([all_df[f'{mn}_pred'] for mn in cfg['compare_model']], y_true,
                            title=f'{task_name} - {subset} DCA', labels=cfg['compare_model'], y_min=-0.15)
            plt.savefig(os.path.join(out_dir, 'dca.svg'), bbox_inches='tight')
            plt.close(fig)
            
            # Calibration
            fig = plt.figure(figsize=(8, 6))
            okcomp.draw_calibration(pred_scores=y_scores, n_bins=5, y_test=y_true, model_names=cfg['compare_model'])
            plt.savefig(os.path.join(out_dir, 'calibration.svg'), bbox_inches='tight')
            plt.close(fig)
            
            # Hosmer-Lemeshow
            hosmer = []
            for mn in cfg['compare_model']:
                result = stats.hosmer_lemeshow_test(y_true, all_df[f'{mn}_pred'], bins=15)
                if isinstance(result, tuple) and len(result) == 2:
                    stat, p = result
                else:
                    stat, p = np.nan, result
                hosmer.append((mn, stat, p))
            
            hosmer_df = pd.DataFrame(hosmer, columns=['Model', 'H-L Stat', 'p-value'])
            hosmer_df.to_csv(os.path.join(out_dir, 'hosmer_lemeshow.csv'), index=False)
    
    # 2. 多任务ROC可视化
    if task_val_data:
        print("📈 Generating multi-task visualizations...")
        summary_dir = os.path.join(cfg['output_dir'], 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 多任务ROC曲线
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), 
               plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]
        
        for idx, task_name in enumerate(task_val_data.keys()):
            data = task_val_data[task_name]
            y_true = data['y_true']
            
            for mn in cfg['compare_model']:
                if mn in data['model_scores']:
                    score = data['model_scores'][mn]
                    fpr, tpr, _ = okcomp.roc_curve(y_true, score)
                    roc_auc = okcomp.auc(fpr, tpr)
                    axs[idx].plot(fpr, tpr, label=f'{mn} (AUC={roc_auc:.3f})')
            
            axs[idx].plot([0, 1], [0, 1], 'k--')
            axs[idx].set_xlim([0.0, 1.0])
            axs[idx].set_ylim([0.0, 1.05])
            axs[idx].set_xlabel('False Positive Rate')
            axs[idx].set_ylabel('True Positive Rate')
            axs[idx].set_title(f'Task: {task_name}')
            axs[idx].legend(loc="lower right", fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(summary_dir, 'multi_task_roc.svg'))
        plt.close(fig)
        
        # 保存汇总指标
        if summary_metrics:
            summary_df = pd.DataFrame(summary_metrics)
            summary_df.to_csv(os.path.join(summary_dir, 'multi_task_metrics.csv'), index=False)
            print(f"✅ Saved summary metrics to {os.path.join(summary_dir, 'multi_task_metrics.csv')}")
    
    print('✅ 全部完成！')

if __name__ == "__main__":
    main()

# 汇总单任务模型的结果

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from onekey_algo.custom.components import comp1 as okcomp
from onekey_algo.custom.components.metrics import analysis_pred_binary
from onekey_algo.custom.components import stats
import sys

def read_pred(path, model_name):
    """读取模型输出，返回 ID + 正类概率"""
    df = pd.read_csv(path)
    if 'label-1' in df.columns:
        return df[['ID', 'label-1']].rename(columns={'label-1': f'{model_name}_pred'})
    raise ValueError(f"{path} 缺少 label-1 列")

def main():
    # 读取配置
    with open('config_multi_task.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 9
    
    # 验证模型数量和任务数量是否匹配
    if len(cfg['tasks']) != len(cfg['compare_model']):
        print(f"⚠️ 警告: 任务数量({len(cfg['tasks'])})和模型数量({len(cfg['compare_model'])})不匹配！")
        print("将使用最小数量进行匹配")
        num_pairs = min(len(cfg['tasks']), len(cfg['compare_model']))
    else:
        num_pairs = len(cfg['tasks'])
    
    # 存储汇总指标
    summary_metrics = []
    roc_data = []  # 存储ROC数据用于多任务图
    
    # 1. 处理每个任务及其对应的模型
    for i in range(num_pairs):
        task = cfg['tasks'][i]
        model_name = cfg['compare_model'][i]
        
        task_name = task['name']
        label_file = task['label_file']
        task_col = task['task_column']
        print(f'\n🔍 处理任务: {task_name} - 模型: {model_name}')
        
        # 读取任务标签
        label_df = pd.read_csv(label_file)[['ID', task_col]]
        
        task_metrics = []  # 存储该任务的所有指标
        
        for subset in ['train', 'val']:
            print(f'  📊 子集: {subset}')
            
            # 读取模型预测
            path = os.path.join(cfg['results_dir'], f'{model_name}_{subset}.csv')
            try:
                pred = read_pred(path, model_name)
            except Exception as e:
                print(f"❌ 读取预测文件出错: {e}")
                continue
            
            # 合并标签
            all_df = pd.merge(pred, label_df, on='ID', how='inner').dropna()
            
            if all_df.empty:
                print(f"⚠️ 警告: {model_name}在{task_name}任务{subset}子集上没有匹配的数据")
                continue
            
            y_true = np.array(all_df[task_col])
            y_score = np.array(all_df[f'{model_name}_pred'])
            
            # 创建输出目录
            out_dir = os.path.join(cfg['output_dir'], task_name, model_name, subset)
            os.makedirs(out_dir, exist_ok=True)
            
            # 计算性能指标
            print(f"     ⏳ 计算性能指标...")
            try:
                acc, auc, ci, tpr, tnr, ppv, npv, prec, rec, f1, th = analysis_pred_binary(y_true, y_score)
                ci_str = f'{ci[0]:.4f} - {ci[1]:.4f}'
                
                # 存储指标
                metrics_row = {
                    'Model': model_name,
                    'Task': task_name,
                    'Subset': subset,
                    'Accuracy': f'{acc:.4f}',
                    'AUC': f'{auc:.4f}',
                    '95% CI': ci_str,
                    'Sensitivity': f'{tpr:.4f}',
                    'Specificity': f'{tnr:.4f}',
                    'PPV': f'{ppv:.4f}',
                    'NPV': f'{npv:.4f}',
                    'Precision': f'{prec:.4f}',
                    'Recall': f'{rec:.4f}',
                    'F1': f'{f1:.4f}',
                    'Threshold': f'{th:.4f}',
                    'Samples': len(y_true)
                }
                
                task_metrics.append(metrics_row)
                summary_metrics.append(metrics_row)
                
                # 保存指标到CSV
                pd.DataFrame([metrics_row]).to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
            except Exception as e:
                print(f"❌ 计算指标出错: {e}")
                continue
            
            # 绘制ROC曲线
            print(f"     ⏳ 绘制ROC曲线...")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                okcomp.draw_roc([y_true], [y_score], labels=[model_name], 
                               title=f'{task_name} - {model_name} - {subset} AUC', ax=ax)
                plt.savefig(os.path.join(out_dir, 'auc.svg'), bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"❌ 绘制ROC出错: {e}")
            
            # 绘制DCA曲线
            print(f"     绘制DCA曲线...")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                okcomp.plot_DCA([y_score], y_true, title=f'{task_name} - {model_name} - {subset} DCA', 
                               labels=[model_name], y_min=-0.15, ax=ax)
                plt.savefig(os.path.join(out_dir, 'dca.svg'), bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"❌ 绘制DCA出错: {e}")
            
            # 绘制校准曲线
            print(f"     ⏳ 绘制校准曲线...")
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                okcomp.draw_calibration(pred_scores=[y_score], n_bins=5, y_test=y_true, 
                                       model_names=[model_name], ax=ax)
                plt.savefig(os.path.join(out_dir, 'calibration.svg'), bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"❌ 绘制校准曲线出错: {e}")
            
            # Hosmer-Lemeshow检验
            print(f"     计算Hosmer-Lemeshow检验...")
            try:
                result = stats.hosmer_lemeshow_test(y_true, y_score, bins=15)
                if isinstance(result, tuple) and len(result) == 2:
                    stat, p = result
                else:
                    stat, p = np.nan, result
                
                hl_row = {
                    'Model': model_name,
                    'Task': task_name,
                    'Subset': subset,
                    'H-L Stat': f'{stat:.4f}' if not np.isnan(stat) else 'N/A',
                    'p-value': f'{p:.4f}'
                }
                pd.DataFrame([hl_row]).to_csv(os.path.join(out_dir, 'hosmer_lemeshow.csv'), index=False)
            except Exception as e:
                print(f"❌ 计算Hosmer-Lemeshow检验出错: {e}")
            
            # 存储ROC数据用于多任务图
            try:
                fpr, tpr, _ = okcomp.roc_curve(y_true, y_score)
                roc_auc = okcomp.auc(fpr, tpr)
                roc_data.append({
                    'task': task_name,
                    'model': model_name,
                    'subset': subset,
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                })
            except:
                pass
            
            print(f"    ✅ {subset}子集处理完成")
        
        # 保存该任务的所有指标
        if task_metrics:
            task_metrics_df = pd.DataFrame(task_metrics)
            task_dir = os.path.join(cfg['output_dir'], task_name, model_name)
            task_metrics_df.to_csv(os.path.join(task_dir, 'all_metrics.csv'), index=False)
    
    # 2. 生成多任务ROC对比图
    if roc_data:
        print("\n📈 生成多任务ROC图...")
        summary_dir = os.path.join(cfg['output_dir'], 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        # 创建多任务ROC图
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2)
        axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
        
        # 收集任务数据
        task_plots = {}
        for data in roc_data:
            task = data['task']
            if task not in task_plots:
                task_plots[task] = []
            task_plots[task].append(data)
        
        # 绘制每个任务的ROC曲线
        for i, (task_name, task_data) in enumerate(task_plots.items()):
            if i >= 4:  # 最多绘制4个任务
                break
                
            ax = axes[i]
            for data in task_data:
                label = f"{data['model']} ({data['subset']}, AUC={data['auc']:.3f})"
                ax.plot(data['fpr'], data['tpr'], label=label)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Task: {task_name}', fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle('Multi-Task ROC Curves', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(summary_dir, 'multi_task_roc.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 多任务ROC图已保存至: {os.path.join(summary_dir, 'multi_task_roc.svg')}")
    
    # 3. 保存汇总指标
    if summary_metrics:
        summary_dir = os.path.join(cfg['output_dir'], 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_df = pd.DataFrame(summary_metrics)
        summary_file = os.path.join(summary_dir, 'multi_task_metrics.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ 汇总指标已保存至: {summary_file}")
    
    print('\n🎉 全部完成！')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)
