import matplotlib.pyplot as plt
import numpy as np
import logging

class Visualizer:
    @staticmethod
    def plot_model_comparison(results, save_path=None):
        """Create and save model comparison visualization"""
        try:
            plt.figure(figsize=(10, 6))
            
            models = list(results.keys())
            train_scores = [results[m]['train_score'] for m in models]
            test_scores = [results[m]['test_score'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, train_scores, width, label='Train Score')
            plt.bar(x + width/2, test_scores, width, label='Test Score')
            
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models)
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            return plt
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            raise
