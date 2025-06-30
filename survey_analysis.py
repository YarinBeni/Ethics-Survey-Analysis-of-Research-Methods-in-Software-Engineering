"""
Survey Analysis Script for Software Developers' Values and Actions
This script analyzes survey data about software developers' values, beliefs, and actions.
The analysis is divided into three main tasks:
1. Comparing declarative values vs. scenario-based actions
2. Analyzing cases where mean differs from mode
3. Comparing "I think" vs. "I do" responses

Author: Yarin
Date: 2024
"""

# %% [markdown]
# # Import Required Libraries
# First, we'll import all the necessary libraries for data analysis and visualization.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')

# Create plots directory if it doesn't exist
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

# %% [markdown]
# # Data Loading and Initial Cleaning
# Load the survey data and perform initial cleaning steps.

# %%
def load_and_clean_data(file_path):
    """
    Load the survey data and perform initial cleaning.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        return None
    
    # Define new column names for better readability
    new_columns = [
        "Timestamp", "Unclear_Requirements", "Thorough_Testing", "Minor_Bug_Failure",
        "Code_Profiling", "Library_Choice_Impact", "Readable_Code", "Code_Structure_Maintainability",
        "Comments_Documentation", "Security_Best_Practices", "Scalability", "Sharing_Bad_Test_Results",
        "Helping_Teammates", "Checking_For_Bias", "Fairness_In_High_Stakes_Systems",
        "Delay_Launch_For_Data_Protection", "Privacy_Leaks_Worry", "Maintain_Respectful_Communication",
        "Shared_Responsibility_For_Failures", "Soften_Critical_Feedback", "Refuse_Morally_Questionable_Task",
        "Consider_Negative_Social_Impact", "Implement_Feature_With_Negative_Impact_If_Required",
        "Country", "Experience", "Gender", "Role", "Project_Criticality",
        "Correctness_Depends_On_Criticality", "Part_Time", "Work_Impact", "Importance_Correctness",
        "Importance_Best_Practices", "Importance_Maintainability", "Importance_Fairness",
        "Importance_Accountability", "Importance_Respect_Human_Relations",
        "Importance_Broader_Social_Impact", "Importance_Privacy", "Most_Important_Value",
        "Least_Important_Value", "Missing_Values", "Criticality_Affects_Values"
    ]
    
    # Truncate the new_columns list to match the number of columns in the dataframe
    df = df.iloc[:, :len(new_columns)]
    df.columns = new_columns
    
    # Drop the first row which is mostly empty in the original Google Forms export
    df = df.drop(0).reset_index(drop=True)
    
    # Convert Likert scale questions to numeric type
    likert_columns = [col for col in new_columns if 'Importance' in col or col in [
        "Unclear_Requirements", "Thorough_Testing", "Minor_Bug_Failure", "Code_Profiling",
        "Library_Choice_Impact", "Readable_Code", "Code_Structure_Maintainability",
        "Comments_Documentation", "Security_Best_Practices", "Scalability", "Sharing_Bad_Test_Results",
        "Helping_Teammates", "Checking_For_Bias", "Fairness_In_High_Stakes_Systems",
        "Delay_Launch_For_Data_Protection", "Privacy_Leaks_Worry", "Maintain_Respectful_Communication",
        "Shared_Responsibility_For_Failures", "Soften_Critical_Feedback", "Refuse_Morally_Questionable_Task",
        "Consider_Negative_Social_Impact", "Implement_Feature_With_Negative_Impact_If_Required"
    ]]
    
    for col in likert_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# %% [markdown]
# # Define Question Categories
# Create the mapping of questions to their categories and types.

# %%
def create_question_mapping():
    """
    Create the mapping of questions to their categories and types.
    Each category maps to:
    - declarative: Questions about stated values/importance
    - scenario: Questions about specific actions/behaviors
    - type: Whether it's a Technical or Ethical consideration
    
    The mapping is based on the following logic:
    - Correctness: Questions about code accuracy and testing
    - Maintainability: Questions about code structure and documentation
    - Software Standards: Questions about best practices and performance
    - Privacy: Questions about data protection
    - Fairness: Questions about bias and discrimination
    - Accountability: Questions about responsibility and transparency
    - Human Relations: Questions about team interaction
    - Social Impact: Questions about broader societal effects
    """
    return {
        'Correctness': {
            'declarative': ['Importance_Correctness'],  # How important is correctness
            'scenario': ['Thorough_Testing', 'Unclear_Requirements'],  # Testing behavior and handling unclear requirements
            'type': 'Technical'
        },
        'Maintainability': {
            'declarative': ['Importance_Maintainability'],  # How important is maintainability
            'scenario': ['Readable_Code', 'Code_Structure_Maintainability', 'Comments_Documentation'],  # Writing maintainable code
            'type': 'Technical'
        },
        'Software Standards': {
            'declarative': ['Importance_Best_Practices'],  # How important are best practices
            'scenario': ['Security_Best_Practices', 'Library_Choice_Impact', 'Code_Profiling', 'Scalability'],  # Following standards
            'type': 'Technical'
        },
        'Privacy': {
            'declarative': ['Importance_Privacy'],  # How important is privacy
            'scenario': ['Delay_Launch_For_Data_Protection', 'Privacy_Leaks_Worry'],  # Privacy protection actions
            'type': 'Ethical'
        },
        'Fairness': {
            'declarative': ['Importance_Fairness'],  # How important is fairness
            'scenario': ['Fairness_In_High_Stakes_Systems', 'Checking_For_Bias'],  # Fairness in practice
            'type': 'Ethical'
        },
        'Accountability': {
            'declarative': ['Importance_Accountability'],  # How important is accountability
            'scenario': ['Sharing_Bad_Test_Results', 'Minor_Bug_Failure'],  # Taking responsibility
            'type': 'Ethical'
        },
        'Human Relations': {
            'declarative': ['Importance_Respect_Human_Relations'],  # How important are human relations
            'scenario': ['Maintain_Respectful_Communication', 'Helping_Teammates', 'Soften_Critical_Feedback'],  # Team interaction
            'type': 'Ethical'
        },
        'Social Impact': {
            'declarative': ['Importance_Broader_Social_Impact'],  # How important is social impact
            'scenario': ['Consider_Negative_Social_Impact', 'Refuse_Morally_Questionable_Task', 
                        'Implement_Feature_With_Negative_Impact_If_Required'],  # Social impact actions
            'type': 'Ethical'
        }
    }

# %% [markdown]
# # Task 1: Declarative vs. Scenario Comparison
# Create comparison graphs for each value, showing the difference between what developers say they value
# and how they would act in scenarios.

# %%
def plot_declarative_vs_scenario(df, question_map):
    """
    Create comparison graphs for each value, showing declarative vs. scenario responses.
    
    Args:
        df (pd.DataFrame): The survey data
        question_map (dict): Mapping of questions to their categories
    """
    for value, mapping in question_map.items():
        plt.figure(figsize=(12, 7))
        
        # Get declarative scores
        declarative_scores = df[mapping['declarative'][0]].dropna()
        
        # Calculate average scenario scores
        scenario_scores = df[mapping['scenario']].mean(axis=1).dropna()
        
        # Prepare data for grouped bar chart
        declarative_counts = declarative_scores.value_counts(normalize=True).sort_index()
        scenario_counts = scenario_scores.round().value_counts(normalize=True).sort_index()
        
        plot_df = pd.DataFrame({
            'Declarative': declarative_counts,
            'Scenario-Based': scenario_counts
        }).fillna(0).reset_index()
        plot_df = plot_df.melt(id_vars='index', var_name='Response Type', value_name='Percentage')
        
        # Create the plot
        sns.barplot(data=plot_df, x='index', y='Percentage', hue='Response Type', palette='viridis')
        
        plt.title(f'Value Analysis: {value}\nDeclarative Statement vs. Scenario-Based Actions', 
                 fontsize=16, pad=20)
        plt.xlabel('Response Score (1=Strongly Disagree, 7=Strongly Agree)', fontsize=12)
        plt.ylabel('Percentage of Respondents', fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.legend(title='Question Type')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(PLOTS_DIR / f'declarative_vs_scenario_{value.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# %% [markdown]
# # Task 2: Mean vs. Mode Analysis
# Find and visualize the cases where the mean and mode differ the most.

# %%
def plot_mean_vs_mode(df, likert_columns):
    """
    Create histograms for questions where mean and mode differ significantly.
    Shows the top 5 differences and includes the actual difference values.
    
    Args:
        df (pd.DataFrame): The survey data
        likert_columns (list): List of Likert scale question columns
    """
    # Calculate differences between mean and mode
    differences = []
    for col in likert_columns:
        if df[col].notna().sum() > 0:
            mean = df[col].mean()
            mode = df[col].mode()[0]
            diff = abs(mean - mode)
            differences.append((diff, col, mean, mode))
    
    # Get top 5 differences
    top_5_diffs = sorted(differences, key=lambda x: x[0], reverse=True)[:5]
    
    # Create plots
    for i, (diff, question, mean, mode) in enumerate(top_5_diffs, 1):
        plt.figure(figsize=(12, 7))
        sns.histplot(df[question], kde=False, bins=np.arange(1, 9)-0.5, stat='count')
        
        plt.axvline(mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean:.2f}')
        plt.axvline(mode, color='blue', linestyle=':', linewidth=2, 
                   label=f'Mode: {mode}')
        
        plt.title(f'Distribution Analysis for:\n"{question}"\n'
                 f'Difference between Mean and Mode: {diff:.2f}', 
                 fontsize=16, pad=20)
        plt.xlabel('Response Score', fontsize=12)
        plt.ylabel('Number of Respondents', fontsize=12)
        plt.legend()
        plt.xticks(range(1, 8))
        
        # Save the plot
        plt.savefig(PLOTS_DIR / f'mean_vs_mode_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print summary of all differences
    print("\nSummary of Mean vs Mode Differences:")
    print("------------------------------------")
    for diff, question, mean, mode in sorted(differences, key=lambda x: x[0], reverse=True):
        print(f"{question}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Mode: {mode}")
        print(f"  Difference: {diff:.2f}")
        print()

# %% [markdown]
# # Task 3: "I think" vs. "I do" Analysis
# Create a scatter plot comparing developers' stated beliefs vs. their reported actions.

# %%
def plot_think_vs_do(df, question_map):
    """
    Create a scatter plot comparing "I think" vs. "I do" responses.
    
    Args:
        df (pd.DataFrame): The survey data
        question_map (dict): Mapping of questions to their categories
    """
    # Create a mapping of questions to their categories
    question_to_category = {}
    for category, mapping in question_map.items():
        for q in mapping['declarative'] + mapping['scenario']:
            question_to_category[q] = category
    
    # Categorize questions into "I think" vs "I do"
    i_think_questions = [
        'Minor_Bug_Failure', 'Privacy_Leaks_Worry', 'Fairness_In_High_Stakes_Systems', 
        'Consider_Negative_Social_Impact'
    ] + [q['declarative'][0] for q in question_map.values()]
    
    i_do_questions = [
        'Unclear_Requirements', 'Thorough_Testing', 'Code_Profiling', 'Library_Choice_Impact',
        'Readable_Code', 'Code_Structure_Maintainability', 'Comments_Documentation',
        'Security_Best_Practices', 'Scalability', 'Sharing_Bad_Test_Results', 'Helping_Teammates',
        'Checking_For_Bias', 'Delay_Launch_For_Data_Protection', 'Maintain_Respectful_Communication',
        'Soften_Critical_Feedback', 'Refuse_Morally_Questionable_Task', 
        'Implement_Feature_With_Negative_Impact_If_Required'
    ]
    
    # Calculate average scores for each category
    analysis_df = pd.DataFrame()
    
    # Calculate scores for each category
    for category in question_map.keys():
        # Get questions for this category
        cat_think_questions = [q for q in i_think_questions if question_to_category.get(q) == category]
        cat_do_questions = [q for q in i_do_questions if question_to_category.get(q) == category]
        
        if cat_think_questions and cat_do_questions:
            analysis_df[f'{category}_think'] = df[cat_think_questions].mean(axis=1)
            analysis_df[f'{category}_do'] = df[cat_do_questions].mean(axis=1)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot each category with a different color
    colors = sns.color_palette('husl', n_colors=len(question_map))
    for (category, color) in zip(question_map.keys(), colors):
        plt.scatter(analysis_df[f'{category}_think'], 
                   analysis_df[f'{category}_do'],
                   label=category,
                   color=color,
                   alpha=0.7)
    
    # Add the y=x line for reference
    min_val = min(analysis_df.min().min(), 1) - 0.2
    max_val = max(analysis_df.max().max(), 7) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', 
             label='Perfect Alignment (y=x)')
    
    plt.title('Alignment of Beliefs vs. Actions\nAverage "I think" score vs. Average "I do" score', 
             fontsize=16, pad=20)
    plt.xlabel('Average "I think..." Score (Stated Beliefs)', fontsize=12)
    plt.ylabel('Average "I will do..." Score (Reported Actions)', fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig(PLOTS_DIR / 'think_vs_do_alignment.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_think_vs_do_aggregated(df, question_map):
    """
    Create an aggregated plot comparing "I think" vs. "I do" responses for each category,
    showing mean values and standard error bars.
    
    Args:
        df (pd.DataFrame): The survey data
        question_map (dict): Mapping of questions to their categories
    """
    # Create a mapping of questions to their categories
    question_to_category = {}
    for category, mapping in question_map.items():
        for q in mapping['declarative'] + mapping['scenario']:
            question_to_category[q] = category
    
    # Categorize questions into "I think" vs "I do"
    i_think_questions = [
        'Minor_Bug_Failure', 'Privacy_Leaks_Worry', 'Fairness_In_High_Stakes_Systems', 
        'Consider_Negative_Social_Impact'
    ] + [q['declarative'][0] for q in question_map.values()]
    
    i_do_questions = [
        'Unclear_Requirements', 'Thorough_Testing', 'Code_Profiling', 'Library_Choice_Impact',
        'Readable_Code', 'Code_Structure_Maintainability', 'Comments_Documentation',
        'Security_Best_Practices', 'Scalability', 'Sharing_Bad_Test_Results', 'Helping_Teammates',
        'Checking_For_Bias', 'Delay_Launch_For_Data_Protection', 'Maintain_Respectful_Communication',
        'Soften_Critical_Feedback', 'Refuse_Morally_Questionable_Task', 
        'Implement_Feature_With_Negative_Impact_If_Required'
    ]
    
    # Calculate aggregated scores for each category
    category_data = []
    for category in question_map.keys():
        # Get questions for this category
        cat_think_questions = [q for q in i_think_questions if question_to_category.get(q) == category]
        cat_do_questions = [q for q in i_do_questions if question_to_category.get(q) == category]
        
        if cat_think_questions and cat_do_questions:
            # Calculate means
            think_means = df[cat_think_questions].mean(axis=1)
            do_means = df[cat_do_questions].mean(axis=1)
            
            # Calculate standard error (SE = SD / sqrt(n))
            n = len(think_means.dropna())
            think_se = think_means.std() / np.sqrt(n)
            do_se = do_means.std() / np.sqrt(n)
            
            category_data.append({
                'Category': category,
                'Think_Mean': think_means.mean(),
                'Think_SE': think_se,
                'Do_Mean': do_means.mean(),
                'Do_SE': do_se,
                'Type': question_map[category]['type']
            })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(category_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each category
    for category_type in ['Technical', 'Ethical']:
        mask = plot_df['Type'] == category_type
        plt.errorbar(plot_df[mask]['Think_Mean'], 
                    plot_df[mask]['Do_Mean'],
                    xerr=plot_df[mask]['Think_SE'],
                    yerr=plot_df[mask]['Do_SE'],
                    fmt='o',
                    label=category_type,
                    capsize=0,  # Remove caps
                    elinewidth=1,
                    markersize=8,
                    alpha=0.7)
        
        # Add category labels with slight offset to avoid overlap
        for i, (_, row) in enumerate(plot_df[mask].iterrows()):
            # Alternate between top and bottom offset to reduce overlap
            y_offset = 0.2 if i % 2 == 0 else -0.2
            plt.annotate(row['Category'],
                        (row['Think_Mean'], row['Do_Mean']),
                        xytext=(5, 5 + y_offset),
                        textcoords='offset points',
                        fontsize=9)  # Slightly smaller font
    
    # Add the y=x line for reference
    min_val = min(plot_df['Think_Mean'].min(), plot_df['Do_Mean'].min()) - 0.5
    max_val = max(plot_df['Think_Mean'].max(), plot_df['Do_Mean'].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', 
             label='Perfect Alignment (y=x)', alpha=0.7)
    
    plt.title('Aggregated Alignment of Beliefs vs. Actions\nMean "I think" vs. Mean "I do" scores by Category\n'
             'Error bars show standard error', 
             fontsize=16, pad=20)
    plt.xlabel('Mean "I think..." Score (Stated Beliefs)', fontsize=12)
    plt.ylabel('Mean "I will do..." Score (Reported Actions)', fontsize=12)
    plt.grid(True, alpha=0.3)  # Lighter grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig(PLOTS_DIR / 'think_vs_do_aggregated.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# # Main Execution
# Run the complete analysis.

# %%
def main():
    # Set the file path
    file_path = 'download.csv'
    
    # Load and clean the data
    print("Loading and cleaning data...")
    df = load_and_clean_data(file_path)
    if df is None:
        return
    
    # Create question mapping
    print("Creating question mapping...")
    question_map = create_question_mapping()
    
    # Get Likert scale columns
    likert_columns = [col for col in df.columns if 'Importance' in col or col in [
        "Unclear_Requirements", "Thorough_Testing", "Minor_Bug_Failure", "Code_Profiling",
        "Library_Choice_Impact", "Readable_Code", "Code_Structure_Maintainability",
        "Comments_Documentation", "Security_Best_Practices", "Scalability", "Sharing_Bad_Test_Results",
        "Helping_Teammates", "Checking_For_Bias", "Fairness_In_High_Stakes_Systems",
        "Delay_Launch_For_Data_Protection", "Privacy_Leaks_Worry", "Maintain_Respectful_Communication",
        "Shared_Responsibility_For_Failures", "Soften_Critical_Feedback", "Refuse_Morally_Questionable_Task",
        "Consider_Negative_Social_Impact", "Implement_Feature_With_Negative_Impact_If_Required"
    ]]
    
    # Task 1: Declarative vs. Scenario Comparison
    print("\nGenerating Task 1: Declarative vs. Scenario Plots...")
    plot_declarative_vs_scenario(df, question_map)
    
    # Task 2: Mean vs. Mode Analysis
    print("\nGenerating Task 2: Mean vs. Mode Analysis Plots...")
    plot_mean_vs_mode(df, likert_columns)
    
    # Task 3: "I think" vs. "I do" Analysis
    print("\nGenerating Task 3: 'I think' vs. 'I do' Analysis...")
    plot_think_vs_do(df, question_map)
    plot_think_vs_do_aggregated(df, question_map)

if __name__ == "__main__":
    main() 