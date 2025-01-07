
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_model_comparison(results, save_path=None):
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prepare data
            models = list(results.keys())
            train_scores = [results[m]['train_score'] for m in models]
            test_scores = [results[m]['test_score'] for m in models]
            cv_scores = [results[m].get('cv_mean_score', 0) for m in models]
            
            # Plot bar chart
            x = np.arange(len(models))
            width = 0.25
            
            ax1.bar(x - width, train_scores, width, label='Train Score')
            ax1.bar(x, test_scores, width, label='Test Score')
            ax1.bar(x + width, cv_scores, width, label='CV Score')
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            
            # Plot learning curves if available
            if 'learning_curve' in results:
                train_sizes = results['learning_curve']['train_sizes']
                train_means = results['learning_curve']['train_means']
                test_means = results['learning_curve']['test_means']
                train_stds = results['learning_curve']['train_stds']
                test_stds = results['learning_curve']['test_stds']
                
                ax2.plot(train_sizes, train_means, label='Training score')
                ax2.plot(train_sizes, test_means, label='Cross-validation score')
                ax2.fill_between(train_sizes, train_means - train_stds,
                             train_means + train_stds, alpha=0.1)
                ax2.fill_between(train_sizes, test_means - test_stds,
                             test_means + test_stds, alpha=0.1)
                ax2.set_xlabel('Training Examples')
                ax2.set_ylabel('Score')
                ax2.set_title('Learning Curves')
                ax2.legend(loc='best')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            return plt
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            raise
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            return plt
            
        except Exception as e:
            logging.error(f"Error in confusion matrix visualization: {str(e)}")
            raise
