#!/usr/bin/env python3
"""Performance dashboard for RAG system metrics"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_metrics_data():
    """Load metrics data from exported files"""
    metrics_files = list(Path(".").glob("*metrics*.json"))
    
    all_metrics = []
    for file_path in metrics_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                if 'metrics' in data:
                    all_metrics.extend(data['metrics'])
        except:
            continue
    
    return pd.DataFrame(all_metrics)

def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    print("📈 Generating RAG Performance Dashboard")
    print("=" * 50)
    
    # Load data
    df = load_metrics_data()
    
    if df.empty:
        print("No metrics data found. Run some tests first!")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RAG System Performance Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Composite score over time
    ax1.plot(df['timestamp'], df['composite_score'], 'b-', marker='o', linewidth=2)
    ax1.set_title('Composite Score Trend')
    ax1.set_ylabel('Score (0-1)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Individual metric scores
    metrics = ['answer_relevance', 'faithfulness', 'context_relevance']
    avg_scores = [df[metric].mean() for metric in metrics]
    
    bars = ax2.bar(metrics, avg_scores, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax2.set_title('Average Metric Scores')
    ax2.set_ylabel('Score (0-1)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 3: Latency distribution
    ax3.hist(df['latency_seconds'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Response Latency Distribution')
    ax3.set_xlabel('Latency (seconds)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Answer length vs relevance
    ax4.scatter(df['answer_length'], df['answer_relevance'], alpha=0.6, c=df['composite_score'], cmap='viridis')
    ax4.set_title('Answer Length vs Relevance')
    ax4.set_xlabel('Answer Length (characters)')
    ax4.set_ylabel('Answer Relevance')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Composite Score')
    
    plt.tight_layout()
    plt.savefig('rag_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("📊 SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total queries evaluated: {len(df)}")
    print(f"Time period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    print()
    
    print("Average Scores:")
    print(f"  Composite: {df['composite_score'].mean():.3f}")
    print(f"  Answer Relevance: {df['answer_relevance'].mean():.3f}")
    print(f"  Faithfulness: {df['faithfulness'].mean():.3f}")
    print(f"  Context Relevance: {df['context_relevance'].mean():.3f}")
    print()
    
    print("Performance Metrics:")
    print(f"  Average Latency: {df['latency_seconds'].mean():.2f}s")
    print(f"  95th Percentile Latency: {df['latency_seconds'].quantile(0.95):.2f}s")
    print(f"  Average Answer Length: {df['answer_length'].mean():.0f} chars")
    print()
    
    # Quality assessment
    good_quality = len(df[df['composite_score'] > 0.7]) / len(df)
    print(f"Quality Assessment: {good_quality:.1%} of queries have good quality (score > 0.7)")
    
    if good_quality > 0.8:
        print("✅ Excellent system performance!")
    elif good_quality > 0.6:
        print("⚠️  Good performance, but room for improvement")
    else:
        print("❌ Performance needs significant improvement")
    
    print(f"\n📊 Dashboard saved as 'rag_performance_dashboard.png'")

def export_daily_report():
    """Export daily performance report"""
    df = load_metrics_data()
    
    if df.empty:
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today = datetime.now().date()
    today_data = df[df['timestamp'].dt.date == today]
    
    if not today_data.empty:
        report = {
            "date": today.isoformat(),
            "total_queries": len(today_data),
            "avg_composite_score": today_data['composite_score'].mean(),
            "avg_answer_relevance": today_data['answer_relevance'].mean(),
            "avg_faithfulness": today_data['faithfulness'].mean(),
            "avg_context_relevance": today_data['context_relevance'].mean(),
            "avg_latency": today_data['latency_seconds'].mean(),
            "p95_latency": today_data['latency_seconds'].quantile(0.95),
            "quality_queries": len(today_data[today_data['composite_score'] > 0.7])
        }
        
        with open(f"daily_report_{today}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📅 Daily report exported for {today}")

if __name__ == "__main__":
    create_performance_dashboard()
    export_daily_report()