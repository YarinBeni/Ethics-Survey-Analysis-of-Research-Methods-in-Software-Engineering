I understand. Here's the revised README, incorporating the work title and making the "Generated Visualizations and Analysis" and "Detailed Plot Interpretations" sections more concise.

-----

# What Matters to Developers? A Comparative Survey of Code and Software Values

This repository contains the complete experimental materials for a research study examining the alignment between software developers' stated values and their reported actions across technical and ethical dimensions.

-----

## Project Overview

This study investigates whether software developers' declarative statements about their values align with their scenario-based actions in practice. We analyzed responses from 42 software developers across eight core values:

**Technical Values:**

  * Correctness
  * Maintainability
  * Software Standards (Best Practices)

**Ethical Values:**

  * Privacy
  * Fairness
  * Accountability
  * Human Relations
  * Social Impact

-----

## Quick Start

### Prerequisites

Ensure you have Python 3.7 or higher installed with the following packages:

```bash
pip install pandas numpy matplotlib seaborn pathlib
```

### Running the Analysis

1.  **Clone or download** this repository.
2.  **Navigate** to the project directory:
    ```bash
    cd research_methods
    ```
3.  **Run the analysis script**:
    ```bash
    python survey_analysis.py
    ```

The script will automatically:

  * Load and clean the survey data from `download.csv`.
  * Generate all plots and save them to the `plots/` directory.
  * Create 16 publication-ready visualizations.

### Expected Output

After running the script, the `plots/` directory will contain:

  * 8 declarative vs. scenario comparison plots
  * 5 mean vs. mode analysis plots
  * 2 "think vs. do" alignment plots
  * 1 aggregated beliefs vs. actions plot

-----

## Data Description

### Survey Data (`download.csv`)

  * **Participants:** 42 software developers
  * **Experience Levels:** Junior (0-2 years) to Senior (6+ years) developers
  * **Roles:** Backend, Frontend, Full-Stack, Data Science, DevOps, and others.
  * **Countries:** Primarily Israel and USA
  * **Response Scale:** 7-point Likert scale (1=Strongly Disagree, 7=Strongly Agree)

### Question Types

1.  **Declarative Questions:** Statements such as "X is highly important to me in my work."
2.  **Scenario Questions:** Specific behavioral situations requiring responses.

-----

## Generated Visualizations and Analysis

The analysis produced 16 distinct visualizations categorized as follows:

### 1\. Declarative vs. Scenario Comparison Plots

These plots, found as `declarative_vs_scenario_[value].png`, compare how developers' stated values align with their reported actions in specific scenarios. Key observations include strong alignment for **Correctness**, a notable gap for **Privacy**, and the largest discrepancy for **Social Impact**. Technical values generally show better alignment than ethical considerations.

### 2\. Mean vs. Mode Analysis Plots

These plots, named `mean_vs_mode_[1-5].png`, highlight questions where average responses differ significantly from the most common response, indicating polarized opinions. Larger differences were observed for ethical considerations like **Fairness**, while technical practices showed greater consensus.

### 3\. Think vs. Do Alignment Analysis

The plots `think_vs_do_alignment.png` and `think_vs_do_aggregated.png` examine the relationship between stated beliefs and reported actions across all categories. Technical categories typically cluster near perfect alignment, whereas ethical categories, particularly **Social Impact**, show actions falling short of beliefs.

-----

## File Structure

```
research_methods/
├── README.md                 # This file
├── survey_analysis.py        # Main analysis script
├── plot_explanations.md      # Detailed plot documentation
├── download.csv              # Raw survey data
└── plots/                    # Generated visualizations
    ├── declarative_vs_scenario_correctness.png
    ├── declarative_vs_scenario_maintainability.png
    ├── declarative_vs_scenario_software_standards.png
    ├── declarative_vs_scenario_privacy.png
    ├── declarative_vs_scenario_fairness.png
    ├── declarative_vs_scenario_accountability.png
    ├── declarative_vs_scenario_human_relations.png
    ├── declarative_vs_scenario_social_impact.png
    ├── mean_vs_mode_1.png
    ├── mean_vs_mode_2.png
    ├── mean_vs_mode_3.png
    ├── mean_vs_mode_4.png
    ├── mean_vs_mode_5.png
    ├── think_vs_do_alignment.png
    └── think_vs_do_aggregated.png
```

-----

## Research Methodology

### Survey Design

  * **Mixed Methods:** Combines declarative value statements with scenario-based questions.
  * **Comprehensive Coverage:** 8 core values spanning technical and ethical dimensions.
  * **Behavioral Focus:** Emphasis on reported actions rather than solely attitudes.
  * **Cross-Cultural:** Participants from multiple countries and backgrounds.

### Statistical Approach

  * **Descriptive Analysis:** Distribution comparisons and gap identification.
  * **Alignment Metrics:** Correlation between stated values and actions.
  * **Consensus Measurement:** Mean vs. mode analysis for opinion polarization.
  * **Error Estimation:** Standard error calculations for reliability assessment.

### Visualization Strategy

  * **Multi-Modal Presentation:** Utilizes bar charts, histograms, and scatter plots.
  * **Comparative Analysis:** Side-by-side value versus action comparisons.
  * **Publication Ready:** High-resolution (300 DPI) professional formatting.
  * **Color Coding:** Consistent technical versus ethical category distinction.

-----

## Key Research Findings

### Primary Insights

1.  **Technical-Ethical Divide:** Developers demonstrate stronger alignment between beliefs and actions in technical areas compared to ethical considerations.
2.  **Implementation Challenges:** Ethical values face greater barriers to practical implementation.
3.  **Consensus Patterns:** More agreement observed on technical practices than ethical approaches.
4.  **Social Impact Gap:** The largest discrepancy exists between stated importance and reported actions.

### Implications for Software Development

  * **Training Focus:** Greater emphasis is needed on ethical implementation strategies.
  * **Organizational Support:** Structural changes are required to enable ethical practice.
  * **Tool Development:** Better tools are necessary for implementing privacy, fairness, and social impact considerations.
  * **Community Standards:** More concrete guidelines are needed for ethical decision-making.

-----

## Technical Notes

### Code Structure

  * **Modular Design:** Separate functions for data loading, analysis, and visualization.
  * **Error Handling:** Robust handling of missing data and edge cases.
  * **Scalable:** Designed for easy extension with additional value categories or analysis types.
  * **Reproducible:** Fixed random seeds and consistent styling for reliable results.

### Customization Options

  * **Color Schemes:** Modify `palette` parameters within plotting functions.
  * **Output Formats:** Change file extensions in `plt.savefig()` calls.
  * **Plot Dimensions:** Adjust `figsize` parameters for different aspect ratios.
  * **Resolution:** Modify the `dpi` parameter for varying quality requirements.

-----

## Citation

If you use this analysis or data in your research, please cite:

```
What Matters to Developers? A Comparative Survey of Code and Software Values
Hadar Rotschield, Dorin Shteyman, Yarin Benizri
The School of Computer Science and Engineering, The Hebrew University of Jerusalem
2025
```

-----

## Contact

For questions about the methodology, data, or analysis, please contact:

  * Hadar Rotschield: hadar.rotschield@mail.huji.ac.il
  * Dorin Shteyman: dorin.shteyman@mail.huji.ac.il
  * Yarin Benizri: yarin.benizri@mail.huji.ac.il
