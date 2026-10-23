import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load Data
    try:
        df = pd.read_csv("noah_regional_dengue_master.csv")
    except FileNotFoundError:
        print("Error: 'noah_regional_dengue_master.csv' not found. Please place it in the script directory.")
        return

    # Set overall aesthetic style
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Total Dengue Cases by Region ---
    plt.figure(figsize=(12, 6))
    region_cases = df.groupby("region")["dengue_cases"].sum().sort_values(ascending=False).reset_index()
    sns.barplot(data=region_cases, x="dengue_cases", y="region", palette="viridis")
    plt.title("Total Reported Dengue Cases by Region", fontsize=14, fontweight="bold")
    plt.xlabel("Total Dengue Cases")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig("07_total_dengue_cases_by_region.png", dpi=300)
    plt.close()

    # --- Plot 2: Dengue Cases vs Weather Factors (Correlation Heatmap) ---
    plt.figure(figsize=(10, 8))
    cols_to_plot = ["dengue_cases", "rainfall_estimate", "temperature_era5", "specific_humidity_era5"]
    corr = df[cols_to_plot].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix: Dengue Cases & Climate Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("08_weather_dengue_correlation_heatmap.png", dpi=300)
    plt.close()

    # --- Plot 3: Dengue Cases Distribution across Plausibility Flags ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="plausibility_flag", y="dengue_cases", palette="Set2")
    plt.yscale("log")  # Log scale due to high case counts/outliers
    plt.title("Dengue Cases Distribution by Plausibility Flag (Log Scale)", fontsize=12, fontweight="bold")
    plt.xlabel("Plausibility Flag")
    plt.ylabel("Dengue Cases (Log Scale)")
    plt.tight_layout()
    plt.savefig("09_plausibility_flag_distribution.png", dpi=300)
    plt.close()

    # --- Plot 4: Weekly Dengue Cases by Region (Stacked Subplots) ---
    regions = df["region"].unique()
    n_regions = len(regions)

    fig, axes = plt.subplots(nrows=n_regions, ncols=1, figsize=(14, 2 * n_regions), sharex=True)
    
    # Handle edge case if there is only 1 region (axes won't be an array)
    if n_regions == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        # Sort by year and week to ensure chronological multi-year plotting continuity if applicable
        sort_cols = ["year", "week"] if "year" in df.columns else ["week"]
        subset = df[df["region"] == region].sort_values(sort_cols)
        
        ax.plot(range(len(subset)), subset["dengue_cases"], color="tab:blue", linewidth=1.2)
        ax.set_title(region, fontsize=10, loc="left", pad=2)
        ax.set_ylabel("Cases", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Set x-label on the bottom subplot
    axes[-1].set_xlabel("Timeline Index / Weeks", fontsize=10)

    plt.suptitle("Weekly Dengue Cases by Region", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("11_weekly_dengue_cases_by_region.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Successfully generated all EDA plot images, including regional weekly timelines!")

if __name__ == "__main__":
    main()
