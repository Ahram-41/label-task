"""
Correlation test script to compare classification results between labeled_sample.csv and full_sample_label_.csv
Based on the basedata.py model classes and their corresponding columns.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def load_and_prepare_data(basefile: str, newfile: str):
    """Load both CSV files and prepare them for comparison"""
    print("Loading data files...")

    # Load the CSV files
    df_base = pd.read_csv(basefile)
    df_new = pd.read_csv(newfile)

    print(f"Base file shape: {df_base.shape}")
    print(f"New file shape: {df_new.shape}")

    # Find common companies using investee_company_beid
    common_ids = set(df_base["investee_company_beid"]).intersection(set(df_new["investee_company_beid"]))
    print(f"Common companies: {len(common_ids)}")

    # Filter to common companies
    df_base_filtered = df_base[df_base["investee_company_beid"].isin(common_ids)].copy()
    df_new_filtered = df_new[df_new["investee_company_beid"].isin(common_ids)].copy()

    # Sort by company ID to ensure alignment
    df_base_filtered = df_base_filtered.sort_values("investee_company_beid").reset_index(drop=True)
    df_new_filtered = df_new_filtered.sort_values("investee_company_beid").reset_index(drop=True)

    return df_base_filtered, df_new_filtered


def create_classification_columns(df):
    """
    Create classification columns for probability-based responses using even threshold (0.5)
    For Vertical_or_Horizontal_Response, Developer_or_Integrator_Response,
    AI_Native_or_Augmented_Response, Automation_Depth_Response
    """
    df_classified = df.copy()

    # Vertical vs Horizontal (1 if vertical > 0.5, 0 if horizontal > 0.5)
    if "vertical_ai_startup" in df.columns and "horizontal_ai_startup" in df.columns:
        df_classified["vertical_or_horizontal_classified"] = (df["vertical_ai_startup"] > 0.5).astype(int)

    # Model Developer vs Integrator (1 if developer > 0.5, 0 if integrator > 0.5)
    if "model_developer" in df.columns and "model_integrator" in df.columns:
        df_classified["model_developer_or_integrator_classified"] = (df["model_developer"] > 0.5).astype(int)

    # AI Native vs Augmented (1 if native > 0.5, 0 if augmented > 0.5)
    if "ai_native" in df.columns and "ai_augmented" in df.columns:
        df_classified["ai_native_or_augmented_classified"] = (df["ai_native"] > 0.5).astype(int)

    # Automation Depth - use argmax for multi-class
    if all(
        col in df.columns
        for col in [
            "full_automation",
            "human_in_the_loop",
            "recommendation_or_insight_only",
        ]
    ):
        automation_cols = [
            "full_automation",
            "human_in_the_loop",
            "recommendation_or_insight_only",
        ]
        df_classified["automation_depth_classified"] = (
            df[automation_cols]
            .idxmax(axis=1)
            .map(
                {
                    "full_automation": 0,
                    "human_in_the_loop": 1,
                    "recommendation_or_insight_only": 2,
                }
            )
        )

    return df_classified


def compare_int_columns(df_base, df_new):
    """
    Compare integer classification columns between the two datasets
    Based on basedata.py int field columns
    """
    # Make copies to avoid modifying original data
    df_base = df_base.copy()
    df_new = df_new.copy()

    # Fill NA values in Core_or_General_AI_Application with 1
    if "Core_or_General_AI_Application" in df_base.columns:
        df_base["Core_or_General_AI_Application"] = df_base["Core_or_General_AI_Application"].fillna(1)
    if "Core_or_General_AI_Application" in df_new.columns:
        df_new["Core_or_General_AI_Application"] = df_new["Core_or_General_AI_Application"].fillna(1)

    # Integer columns from basedata.py models
    int_columns = [
        "Core_or_General_AI_Application",  # Core_or_General_AI_Application_Response
        "Is_data_centric",  # Is_Data_Centric_Response
        "Is_niche_or_broad_market_A",  # Is_Niche_or_Broad_Market_A_Response
        "Is_niche_or_broad_market_B",  # Is_Niche_or_Broad_Market_B_Response
        "Is_product_or_platform",  # Is_Product_or_Platform_Response
        "AI_startup_type",  # AI_startup_type_Response
    ]

    # Add our newly created classification columns
    classification_columns = [
        "vertical_or_horizontal_classified",
        "model_developer_or_integrator_classified",
        "ai_native_or_augmented_classified",
        "automation_depth_classified",
    ]

    all_columns_to_compare = int_columns + classification_columns

    results = {}

    print("\n" + "=" * 80)
    print("CLASSIFICATION COMPARISON RESULTS")
    print("=" * 80)

    for col in all_columns_to_compare:
        if col in df_base.columns and col in df_new.columns:
            # Get non-null values for both datasets
            mask = df_base[col].notna() & df_new[col].notna()
            base_vals = df_base.loc[mask, col]
            new_vals = df_new.loc[mask, col]

            if len(base_vals) > 0:
                # Calculate accuracy (same classification rate)
                accuracy = accuracy_score(base_vals, new_vals)

                # Calculate Cohen's Kappa
                kappa = cohen_kappa_score(base_vals, new_vals)

                # Calculate Balanced Accuracy
                balanced_acc = balanced_accuracy_score(base_vals, new_vals)

                # Calculate Weighted F1 Score
                weighted_f1 = f1_score(base_vals, new_vals, average="weighted")

                # Calculate Pearson correlation coefficient
                try:
                    pearson_r, pearson_p = pearsonr(base_vals, new_vals)
                    pearson_r_squared = pearson_r**2
                except:
                    pearson_r, pearson_p, pearson_r_squared = np.nan, np.nan, np.nan

                # Get unique values to understand the classification space
                unique_base = sorted(base_vals.unique())
                unique_new = sorted(new_vals.unique())

                # Store results
                results[col] = {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "weighted_f1": weighted_f1,
                    "cohen_kappa": kappa,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "pearson_r_squared": pearson_r_squared,
                    "total_samples": len(base_vals),
                    "unique_values_base": unique_base,
                    "unique_values_new": unique_new,
                    "base_distribution": base_vals.value_counts().sort_index().to_dict(),
                    "new_distribution": new_vals.value_counts().sort_index().to_dict(),
                }

                # Print detailed results
                print(f"\n{col}:")
                print(f"  Same Classification Rate: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"  Balanced Accuracy: {balanced_acc:.3f}")
                print(f"  Weighted F1 Score: {weighted_f1:.3f}")
                print(f"  Cohen's Kappa: {kappa:.3f}")
                print(f"  Pearson Correlation (r): {pearson_r:.3f}")
                print(f"  Pearson R-squared: {pearson_r_squared:.3f}")
                print(f"  Pearson P-value: {pearson_p:.6f}" if not np.isnan(pearson_p) else "  Pearson P-value: N/A")
                print(f"  Total Samples Compared: {len(base_vals)}")
                print(f"  Base Distribution: {results[col]['base_distribution']}")
                print(f"  New Distribution: {results[col]['new_distribution']}")

                # Confusion matrix for detailed analysis
                if len(unique_base) <= 10 and len(unique_new) <= 10:  # Only for reasonable number of classes
                    cm = confusion_matrix(base_vals, new_vals)
                    print(f"  Confusion Matrix:")
                    print(f"    {cm}")
            else:
                print(f"\n{col}: No valid data to compare")
        else:
            missing_from = []
            if col not in df_base.columns:
                missing_from.append("base")
            if col not in df_new.columns:
                missing_from.append("new")
            print(f"\n{col}: Missing from {', '.join(missing_from)} dataset")

    return results


def generate_summary_statistics(results):
    """Generate summary statistics for all column comparisons"""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if not results:
        print("No results to summarize.")
        return

    # Create summary DataFrame
    summary_data = []
    for col, data in results.items():
        summary_data.append(
            {
                "Column": col,
                "Same_Classification_Rate": data["accuracy"],
                "Balanced_Accuracy": data["balanced_accuracy"],
                "Weighted_F1": data["weighted_f1"],
                "Cohen_Kappa": data["cohen_kappa"],
                "Pearson_r": data["pearson_r"],
                "Pearson_r_squared": data["pearson_r_squared"],
                "Pearson_p_value": data["pearson_p"],
                "Total_Samples": data["total_samples"],
                "Num_Classes_Base": len(data["unique_values_base"]),
                "Num_Classes_New": len(data["unique_values_new"]),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Same_Classification_Rate", ascending=False)

    print("\nRanked by Same Classification Rate:")
    print(summary_df.to_string(index=False, float_format="%.3f"))

    # Overall statistics
    accuracies = [data["accuracy"] for data in results.values()]
    balanced_accs = [data["balanced_accuracy"] for data in results.values()]
    weighted_f1s = [data["weighted_f1"] for data in results.values()]
    kappas = [data["cohen_kappa"] for data in results.values()]
    pearson_rs = [data["pearson_r"] for data in results.values() if not np.isnan(data["pearson_r"])]

    print(f"\nOverall Statistics - Same Classification Rate:")
    print(f"  Mean Same Classification Rate: {np.mean(accuracies):.3f}")
    print(f"  Median Same Classification Rate: {np.median(accuracies):.3f}")
    print(f"  Std Dev: {np.std(accuracies):.3f}")
    print(f"  Min: {np.min(accuracies):.3f}")
    print(f"  Max: {np.max(accuracies):.3f}")

    print(f"\nOverall Statistics - Balanced Accuracy:")
    print(f"  Mean Balanced Accuracy: {np.mean(balanced_accs):.3f}")
    print(f"  Median Balanced Accuracy: {np.median(balanced_accs):.3f}")
    print(f"  Std Dev: {np.std(balanced_accs):.3f}")
    print(f"  Min: {np.min(balanced_accs):.3f}")
    print(f"  Max: {np.max(balanced_accs):.3f}")

    print(f"\nOverall Statistics - Weighted F1 Score:")
    print(f"  Mean Weighted F1: {np.mean(weighted_f1s):.3f}")
    print(f"  Median Weighted F1: {np.median(weighted_f1s):.3f}")
    print(f"  Std Dev: {np.std(weighted_f1s):.3f}")
    print(f"  Min: {np.min(weighted_f1s):.3f}")
    print(f"  Max: {np.max(weighted_f1s):.3f}")

    print(f"\nOverall Statistics - Cohen's Kappa:")
    print(f"  Mean Kappa: {np.mean(kappas):.3f}")
    print(f"  Median Kappa: {np.median(kappas):.3f}")
    print(f"  Std Dev: {np.std(kappas):.3f}")
    print(f"  Min: {np.min(kappas):.3f}")
    print(f"  Max: {np.max(kappas):.3f}")

    if pearson_rs:
        print(f"\nOverall Statistics - Pearson Correlation:")
        print(f"  Mean Pearson r: {np.mean(pearson_rs):.3f}")
        print(f"  Median Pearson r: {np.median(pearson_rs):.3f}")
        print(f"  Std Dev: {np.std(pearson_rs):.3f}")
        print(f"  Min: {np.min(pearson_rs):.3f}")
        print(f"  Max: {np.max(pearson_rs):.3f}")

    # Categories of performance
    high_agreement = [col for col, data in results.items() if data["accuracy"] >= 0.8]
    medium_agreement = [col for col, data in results.items() if 0.6 <= data["accuracy"] < 0.8]
    low_agreement = [col for col, data in results.items() if data["accuracy"] < 0.6]

    print(f"\nPerformance Categories:")
    print(f"  High Agreement (≥80%): {len(high_agreement)} columns")
    if high_agreement:
        print(f"    {high_agreement}")
    print(f"  Medium Agreement (60-79%): {len(medium_agreement)} columns")
    if medium_agreement:
        print(f"    {medium_agreement}")
    print(f"  Low Agreement (<60%): {len(low_agreement)} columns")
    if low_agreement:
        print(f"    {low_agreement}")

    return summary_df


def create_visualization(results):
    """Create visualization of the comparison results"""
    if not results:
        print("No results to visualize.")
        return

    # Prepare data
    columns = list(results.keys())
    accuracies = [results[col]["accuracy"] for col in columns]
    balanced_accs = [results[col]["balanced_accuracy"] for col in columns]
    weighted_f1s = [results[col]["weighted_f1"] for col in columns]
    kappas = [results[col]["cohen_kappa"] for col in columns]

    # Create subplot with a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Classification Agreement Metrics Between Datasets", fontsize=20)

    metrics = {
        "Accuracy": accuracies,
        "Balanced Accuracy": balanced_accs,
        "Weighted F1 Score": weighted_f1s,
        "Cohen's Kappa": kappas,
    }

    colors = ["skyblue", "lightgreen", "salmon", "plum"]

    for i, (title, data) in enumerate(metrics.items()):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(range(len(columns)), data, color=colors[i], edgecolor="black", alpha=0.7)
        for bar, val in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([col.replace("_", "\n") for col in columns], rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    plt.savefig("classification_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved as 'classification_comparison.png'")


def correlation_test(basefile, newfile):
    """Main function to run the correlation test"""
    print("Starting Task 1 Correlation Test")
    print("=" * 50)

    # Load and prepare data
    df_base, df_new = load_and_prepare_data(basefile, newfile)

    # Create classification columns for probability-based responses
    print("\nCreating classification columns with even threshold (0.5)...")
    df_base_classified = create_classification_columns(df_base)
    df_new_classified = create_classification_columns(df_new)

    # Compare integer columns and classification columns
    results = compare_int_columns(df_base_classified, df_new_classified)

    # Generate summary statistics
    summary_df = generate_summary_statistics(results)

    # Create visualization
    create_visualization(results)

    # Save detailed results to file
    if results:
        with open("correlation_test_results.txt", "w") as f:
            f.write("Task 1 Correlation Test Results\n")
            f.write("=" * 50 + "\n\n")

            for col, data in results.items():
                f.write(f"{col}:\n")
                f.write(f"  Same Classification Rate: {data['accuracy']:.3f}\n")
                f.write(f"  Balanced Accuracy: {data['balanced_accuracy']:.3f}\n")
                f.write(f"  Weighted F1 Score: {data['weighted_f1']:.3f}\n")
                f.write(f"  Cohen's Kappa: {data['cohen_kappa']:.3f}\n")
                f.write(f"  Pearson Correlation (r): {data['pearson_r']:.3f}\n")
                f.write(f"  Pearson R-squared: {data['pearson_r_squared']:.3f}\n")
                f.write(
                    f"  Pearson P-value: {data['pearson_p']:.6f}\n"
                    if not np.isnan(data["pearson_p"])
                    else "  Pearson P-value: N/A\n"
                )
                f.write(f"  Total Samples: {data['total_samples']}\n")
                f.write(f"  Base Distribution: {data['base_distribution']}\n")
                f.write(f"  New Distribution: {data['new_distribution']}\n\n")

        summary_df.to_csv("correlation_summary.csv", index=False)
        print(f"\nDetailed results saved to 'correlation_test_results.txt'")
        print(f"Summary saved to 'correlation_summary.csv'")


if __name__ == "__main__":
    basefile = "/Users/wbik/Downloads/label-task/labeled_sample.csv"
    newfile = "/Users/wbik/Downloads/label-task/full_sample_label_.csv"
    correlation_test(basefile, newfile)
