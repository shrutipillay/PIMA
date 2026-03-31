import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
except ImportError as e:
    st.error(f"Error importing sklearn: {str(e)}")
    st.stop()

st.set_page_config(page_title="Pima Diabetes Classifier", layout="wide")

st.title("🏥 Pima Diabetes Classification Model")

# Load the results
try:
    with open('model_results.json', 'r') as f:
        results = json.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Display best combination info
best_combo = results['best_combination']
st.info(f"**Best Preprocessing:** {best_combo['preprocessing']} | **Best Resampling:** {best_combo['resampling']} | **Best Model:** {best_combo['model']} | **Accuracy:** {best_combo['accuracy']:.4f} ({best_combo['accuracy']*100:.2f}%)")

st.divider()

# Display comprehensive comparison table
st.subheader("📊 Complete Model, Preprocessing and Resampling Comparison (Sorted by Accuracy)")

# Flatten results for dataframe
comparison_data = []
for preprocess_name, resample_dict in results['all_results'].items():
    for resample_name, models_dict in resample_dict.items():
        for model_name, accuracy in models_dict.items():
            comparison_data.append({
                'Preprocessing': preprocess_name,
                'Resampling': resample_name,
                'Model': model_name,
                'Accuracy': round(accuracy, 4),
                'Accuracy %': f"{accuracy*100:.2f}%"
            })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
comparison_df['Rank'] = range(1, len(comparison_df) + 1)
comparison_df = comparison_df[['Rank', 'Preprocessing', 'Resampling', 'Model', 'Accuracy', 'Accuracy %']]

st.dataframe(comparison_df, width='stretch', hide_index=True)

# Display summary by preprocessing and resampling combination
st.divider()
st.subheader("📈 Summary by Preprocessing and Resampling Combination")

summary_data = []
for preprocess_name, resample_dict in results['all_results'].items():
    for resample_name, models_dict in resample_dict.items():
        accuracies = list(models_dict.values())
        summary_data.append({
            'Preprocessing': preprocess_name,
            'Resampling': resample_name,
            'Best Accuracy': f"{max(accuracies):.4f}",
            'Worst Accuracy': f"{min(accuracies):.4f}",
            'Average Accuracy': f"{np.mean(accuracies):.4f}",
            'Best Model': max(models_dict, key=models_dict.get)
        })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Best Accuracy', key=lambda x: x.str.replace('%', '').astype(float), ascending=False)
st.dataframe(summary_df, width='stretch', hide_index=True)

# Display summary by preprocessing technique
st.divider()
st.subheader("🔧 Summary by Preprocessing Technique")

preprocess_summary_data = []
for preprocess_name, resample_dict in results['all_results'].items():
    all_accuracies = []
    for resample_name, models_dict in resample_dict.items():
        all_accuracies.extend(models_dict.values())
    preprocess_summary_data.append({
        'Preprocessing Technique': preprocess_name,
        'Best Accuracy': f"{max(all_accuracies):.4f}",
        'Worst Accuracy': f"{min(all_accuracies):.4f}",
        'Average Accuracy': f"{np.mean(all_accuracies):.4f}"
    })

preprocess_summary_df = pd.DataFrame(preprocess_summary_data)
st.dataframe(preprocess_summary_df, width='stretch', hide_index=True)

# Display summary by resampling technique
st.divider()
st.subheader("⚖️ Summary by Resampling Technique")

resample_summary_data = {}
for preprocess_name, resample_dict in results['all_results'].items():
    for resample_name, models_dict in resample_dict.items():
        if resample_name not in resample_summary_data:
            resample_summary_data[resample_name] = []
        resample_summary_data[resample_name].extend(models_dict.values())

resample_summary = []
for resample_name in sorted(resample_summary_data.keys()):
    accuracies = resample_summary_data[resample_name]
    resample_summary.append({
        'Resampling Technique': resample_name,
        'Best Accuracy': f"{max(accuracies):.4f}",
        'Worst Accuracy': f"{min(accuracies):.4f}",
        'Average Accuracy': f"{np.mean(accuracies):.4f}"
    })

resample_summary_df = pd.DataFrame(resample_summary)
st.dataframe(resample_summary_df, width='stretch', hide_index=True)

# Display summary by model
st.divider()
st.subheader("🤖 Summary by Model")

model_summary = {}
for preprocess_name, resample_dict in results['all_results'].items():
    for resample_name, models_dict in resample_dict.items():
        for model_name, accuracy in models_dict.items():
            if model_name not in model_summary:
                model_summary[model_name] = []
            model_summary[model_name].append(accuracy)

model_data = []
for model_name in sorted(model_summary.keys()):
    accuracies = model_summary[model_name]
    model_data.append({
        'Model': model_name,
        'Best Accuracy': f"{max(accuracies):.4f}",
        'Worst Accuracy': f"{min(accuracies):.4f}",
        'Average Accuracy': f"{np.mean(accuracies):.4f}"
    })

model_df = pd.DataFrame(model_data)
st.dataframe(model_df, width='stretch', hide_index=True)

st.divider()
st.info("💡 This comprehensive analysis shows how different preprocessing techniques, resampling methods, and models affect performance on the Pima Indian Diabetes dataset. Total: 150 combinations (10 models × 5 preprocessing × 3 resampling techniques).")
