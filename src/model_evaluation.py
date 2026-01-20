"""
Comprehensive model evaluation script with performance metrics.
Provides detailed analysis of classifier performance including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- ROC-AUC Curves
- Detailed per-class performance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from typing import Dict, Tuple, List
from datetime import datetime
import json
import os


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
    REVERSE_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """Initialize evaluator with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def evaluate(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_scores: np.ndarray = None
    ) -> Dict:
        """
        Perform comprehensive evaluation.
        
        Args:
            y_true: True labels (list of sentiment strings)
            y_pred: Predicted labels (list of sentiment strings)
            y_scores: Prediction scores/probabilities (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Convert to numeric labels
        y_true_numeric = np.array([self.LABEL_MAPPING[label] for label in y_true])
        y_pred_numeric = np.array([self.LABEL_MAPPING[label] for label in y_pred])
        
        metrics = {}
        
        # Overall metrics
        metrics["overall"] = self._calculate_overall_metrics(y_true_numeric, y_pred_numeric)
        
        # Per-class metrics
        metrics["per_class"] = self._calculate_per_class_metrics(
            y_true_numeric, y_pred_numeric
        )
        
        # Confusion matrix
        metrics["confusion_matrix"] = self._calculate_confusion_matrix(
            y_true_numeric, y_pred_numeric
        )
        
        # Classification report
        metrics["classification_report"] = classification_report(
            y_true_numeric, y_pred_numeric,
            target_names=list(self.REVERSE_MAPPING.values()),
            output_dict=True
        )
        
        # If scores provided, calculate AUC
        if y_scores is not None:
            metrics["roc_auc"] = self._calculate_roc_auc(y_true_numeric, y_scores)
        
        return metrics
    
    def _calculate_overall_metrics(self, y_true, y_pred) -> Dict:
        """Calculate overall performance metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    
    def _calculate_per_class_metrics(self, y_true, y_pred) -> Dict:
        """Calculate per-class metrics."""
        per_class = {}
        
        for class_id, class_name in self.REVERSE_MAPPING.items():
            # Create binary labels for this class
            y_true_binary = (y_true == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)
            
            per_class[class_name] = {
                "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                "f1_score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                "support": int(np.sum(y_true == class_id)),
            }
        
        return per_class
    
    def _calculate_confusion_matrix(self, y_true, y_pred) -> List:
        """Calculate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        return cm.tolist()
    
    def _calculate_roc_auc(self, y_true, y_scores) -> Dict:
        """Calculate ROC-AUC metrics."""
        roc_auc_dict = {}
        
        # Multi-class ROC-AUC
        try:
            roc_auc_dict["macro"] = float(
                roc_auc_score(
                    y_true, y_scores,
                    multi_class="ovr",
                    average="macro"
                )
            )
        except Exception as e:
            roc_auc_dict["macro"] = None
        
        return roc_auc_dict
    
    def generate_confusion_matrix_plot(
        self,
        y_true: List[str],
        y_pred: List[str],
        save_path: str = None
    ) -> str:
        """Generate and save confusion matrix plot."""
        y_true_numeric = np.array([self.LABEL_MAPPING[label] for label in y_true])
        y_pred_numeric = np.array([self.LABEL_MAPPING[label] for label in y_pred])
        
        cm = confusion_matrix(y_true_numeric, y_pred_numeric, labels=[0, 1, 2])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.REVERSE_MAPPING.values()),
            yticklabels=list(self.REVERSE_MAPPING.values()),
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Sentiment Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"confusion_matrix_{self.timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_metrics_plot(
        self,
        metrics: Dict,
        save_path: str = None
    ) -> str:
        """Generate and save overall metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
        
        overall = metrics["overall"]
        per_class = metrics["per_class"]
        
        # Plot 1: Overall Metrics
        ax = axes[0, 0]
        metrics_to_plot = {
            'Accuracy': overall['accuracy'],
            'Macro F1': overall['macro_f1'],
            'Weighted F1': overall['weighted_f1'],
        }
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=['#2ecc71', '#3498db', '#e74c3c'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Per-Class F1 Scores
        ax = axes[0, 1]
        class_names = list(per_class.keys())
        f1_scores = [per_class[name]['f1_score'] for name in class_names]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = ax.bar(class_names, f1_scores, color=colors)
        ax.set_ylim([0, 1])
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title('Per-Class F1 Scores', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Per-Class Support (Data Distribution)
        ax = axes[1, 0]
        supports = [per_class[name]['support'] for name in class_names]
        bars = ax.bar(class_names, supports, color=colors)
        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title('Class Distribution in Test Set', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Precision vs Recall per Class
        ax = axes[1, 1]
        x = np.arange(len(class_names))
        width = 0.35
        
        precisions = [per_class[name]['precision'] for name in class_names]
        recalls = [per_class[name]['recall'] for name in class_names]
        
        bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
        
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Precision vs Recall per Class', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"metrics_{self.timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_detailed_report(
        self,
        y_true: List[str],
        y_pred: List[str],
        model_name: str = "Sentiment Classifier",
        save_path: str = None
    ) -> str:
        """Generate detailed HTML evaluation report."""
        metrics = self.evaluate(y_true, y_pred)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .neutral {{ color: #f39c12; font-weight: bold; }}
                .timestamp {{ color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Sentiment Analysis Model - Evaluation Report</h1>
                <p>{model_name}</p>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Performance Metrics</h2>
                <div>
                    <div class="metric">
                        <div class="metric-value">{metrics['overall']['accuracy']:.4f}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{metrics['overall']['macro_f1']:.4f}</div>
                        <div class="metric-label">Macro F1-Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{metrics['overall']['weighted_f1']:.4f}</div>
                        <div class="metric-label">Weighted F1-Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{len(y_true)}</div>
                        <div class="metric-label">Total Samples</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Metrics Breakdown</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Macro Precision</td>
                        <td>{metrics['overall']['macro_precision']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Macro Recall</td>
                        <td>{metrics['overall']['macro_recall']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Weighted Precision</td>
                        <td>{metrics['overall']['weighted_precision']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Weighted Recall</td>
                        <td>{metrics['overall']['weighted_recall']:.4f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Per-Class Performance</h2>
                <table>
                    <tr>
                        <th>Sentiment Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support (Samples)</th>
                    </tr>
        """
        
        for class_name, class_metrics in metrics['per_class'].items():
            class_html = f"<span class='{class_name}'>{class_name.upper()}</span>"
            html_content += f"""
                    <tr>
                        <td>{class_html}</td>
                        <td>{class_metrics['precision']:.4f}</td>
                        <td>{class_metrics['recall']:.4f}</td>
                        <td>{class_metrics['f1_score']:.4f}</td>
                        <td>{class_metrics['support']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
                <ul>
        """
        
        # Generate insights
        overall_acc = metrics['overall']['accuracy']
        if overall_acc >= 0.85:
            html_content += "<li>✓ Excellent overall accuracy - model performs well</li>"
        elif overall_acc >= 0.75:
            html_content += "<li>✓ Good overall accuracy - model performs satisfactorily</li>"
        else:
            html_content += "<li>⚠ Moderate accuracy - model may need improvements</li>"
        
        # Check per-class performance
        for class_name, class_metrics in metrics['per_class'].items():
            if class_metrics['f1_score'] < 0.7:
                html_content += f"<li>⚠ {class_name.capitalize()} class F1-score is relatively low ({class_metrics['f1_score']:.3f})</li>"
        
        html_content += """
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"evaluation_report_{self.timestamp}.html")
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        return save_path
    
    def save_metrics_json(
        self,
        metrics: Dict,
        save_path: str = None
    ) -> str:
        """Save metrics as JSON for programmatic access."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"metrics_{self.timestamp}.json")
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return save_path


def run_full_evaluation(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "Sentiment Classifier"
) -> Dict:
    """
    Run complete evaluation and generate all reports.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary with paths to all generated reports
    """
    evaluator = ModelEvaluator()
    
    print(f"\n{'='*60}")
    print(f"Model Evaluation Report: {model_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"{'='*60}\n")
    
    # Calculate metrics
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Print overall metrics
    print("Overall Performance:")
    print(f"  Accuracy:      {metrics['overall']['accuracy']:.4f}")
    print(f"  Macro F1:      {metrics['overall']['macro_f1']:.4f}")
    print(f"  Weighted F1:   {metrics['overall']['weighted_f1']:.4f}\n")
    
    # Print per-class metrics
    print("Per-Class Performance:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name.upper():<10} - "
              f"Precision: {class_metrics['precision']:.3f}, "
              f"Recall: {class_metrics['recall']:.3f}, "
              f"F1: {class_metrics['f1_score']:.3f} "
              f"(support: {class_metrics['support']})")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    confusion_matrix_path = evaluator.generate_confusion_matrix_plot(y_true, y_pred)
    metrics_plot_path = evaluator.generate_metrics_plot(metrics)
    
    # Generate reports
    print("Generating reports...")
    html_report_path = evaluator.generate_detailed_report(y_true, y_pred, model_name)
    json_metrics_path = evaluator.save_metrics_json(metrics)
    
    results = {
        "metrics": metrics,
        "confusion_matrix_plot": confusion_matrix_path,
        "metrics_plot": metrics_plot_path,
        "html_report": html_report_path,
        "json_metrics": json_metrics_path,
    }
    
    print(f"\nResults saved to: {evaluator.output_dir}")
    print(f"  - Confusion Matrix: {confusion_matrix_path}")
    print(f"  - Metrics Plot: {metrics_plot_path}")
    print(f"  - HTML Report: {html_report_path}")
    print(f"  - JSON Metrics: {json_metrics_path}\n")
    
    return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic predictions
    sentiments = ["negative", "neutral", "positive"]
    y_true = np.random.choice(sentiments, size=500, p=[0.3, 0.3, 0.4])
    
    # Generate predictions with some error
    y_pred = y_true.copy()
    error_indices = np.random.choice(len(y_pred), size=int(0.15 * len(y_pred)), replace=False)
    for idx in error_indices:
        y_pred[idx] = np.random.choice(sentiments)
    
    # Run evaluation
    results = run_full_evaluation(
        y_true.tolist(),
        y_pred.tolist(),
        model_name="cardiffnlp/twitter-roberta-base-sentiment"
    )
