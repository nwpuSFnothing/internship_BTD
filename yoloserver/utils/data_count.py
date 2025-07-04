#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Dataset Label Statistics and Visualization Analysis Script
Count the number of labels for each category in train, val, test datasets
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import matplotlib.font_manager as fm

# Set font for visualization
font_path = r'C:\Users\14854\Desktop\BTD\yoloserver\utils\LXGWWenKai-Bold.ttf'
if os.path.exists(font_path):
    # Register custom font
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"Custom font loaded: {font_prop.get_name()}")
else:
    # Use system default font for English
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    print("Using system default English font")

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

class YOLODatasetAnalyzer:
    def __init__(self, data_root):
        """
        Initialize analyzer
        
        Args:
            data_root: Dataset root directory path
        """
        self.data_root = data_root
        self.datasets = ['train', 'val', 'test']
        self.class_names = {
            0: 'G_Category',  # Files starting with G
            1: 'M_Category',  # Files starting with M  
            2: 'P_Category'   # Files starting with P
        }
        self.results = {}
        
    def count_labels_in_dataset(self, dataset_name):
        labels_dir = os.path.join(self.data_root, dataset_name, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Directory {labels_dir} does not exist")
            return {}
            
        # Count labels for each category
        class_counts = Counter()
        total_files = 0
        
        # Get all label files
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            if os.path.basename(label_file) == 'labels.cache':
                continue
                
            total_files += 1
            
            # Read label file
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line:
                        # YOLO format: class_id x_center y_center width height
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            
            except Exception as e:
                print(f"Error reading file {label_file}: {e}")
                continue
        
        print(f"{dataset_name} Dataset:")
        print(f"  Total files: {total_files}")
        print(f"  Total labels: {sum(class_counts.values())}")
        
        return dict(class_counts)
    
    def analyze_all_datasets(self):
        """
        Analyze all datasets
        """
        print("Starting YOLO dataset analysis...")
        print("=" * 50)
        
        for dataset in self.datasets:
            print(f"\nAnalyzing {dataset} dataset...")
            counts = self.count_labels_in_dataset(dataset)
            self.results[dataset] = counts
            
            # Display current dataset statistics
            for class_id, count in sorted(counts.items()):
                class_name = self.class_names.get(class_id, f'Unknown_Class_{class_id}')
                print(f"  {class_name} (ID: {class_id}): {count} labels")
        
        print("\n" + "=" * 50)
        print("Analysis completed!")
        
    def create_dataframe(self):
        """
        Create DataFrame for visualization
        """
        # Collect all class IDs that appear
        all_classes = set()
        for counts in self.results.values():
            all_classes.update(counts.keys())
        
        # Create data table
        data = []
        for dataset in self.datasets:
            counts = self.results.get(dataset, {})
            for class_id in sorted(all_classes):
                count = counts.get(class_id, 0)
                class_name = self.class_names.get(class_id, f'Unknown_Class_{class_id}')
                data.append({
                    'Dataset': dataset,
                    'Class_ID': class_id,
                    'Class_Name': class_name,
                    'Label_Count': count
                })
        
        return pd.DataFrame(data)
    
    def create_visualizations(self, output_dir):
        """
        Create visualization charts
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.create_dataframe()
        
        # Set chart style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # 1. Stacked Bar Chart - Distribution of different categories in each dataset
        plt.figure(figsize=fig_size)
        pivot_df = df.pivot(index='Dataset', columns='Class_Name', values='Label_Count').fillna(0)
        
        ax = pivot_df.plot(kind='bar', stacked=True, figsize=fig_size, 
                          color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('YOLO Dataset Label Distribution by Category (Stacked Chart)', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Label Count', fontsize=12)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_Label_Distribution_Stacked.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Grouped Bar Chart - Comparison of different categories across datasets
        plt.figure(figsize=fig_size)
        pivot_df.plot(kind='bar', figsize=fig_size, width=0.8,
                     color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('YOLO Dataset Label Distribution by Category (Grouped Comparison)', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Label Count', fontsize=12)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_Label_Distribution_Grouped.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap - Dataset vs Category heatmap
        plt.figure(figsize=(10, 6))
        heatmap_data = pivot_df.T  # Transpose: categories as rows, datasets as columns
        sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd', 
                   cbar_kws={'label': 'Label Count'})
        plt.title('YOLO Dataset Label Distribution Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_Label_Distribution_Heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Pie Chart - Proportion of each category in the overall dataset
        plt.figure(figsize=(12, 8))
        total_counts = df.groupby('Class_Name')['Label_Count'].sum()
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        plt.pie(total_counts.values, labels=total_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors[:len(total_counts)])
        plt.title('Category Proportion in Overall Dataset', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_Category_Proportion_Pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Dataset Total Comparison - Total labels per dataset
        plt.figure(figsize=(10, 6))
        dataset_totals = df.groupby('Dataset')['Label_Count'].sum()
        bars = plt.bar(dataset_totals.index, dataset_totals.values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Display values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=12)
        
        plt.title('Total Label Count Comparison by Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Total Label Count', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_Dataset_Total_Comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Detailed Statistics Table
        plt.figure(figsize=(12, 8))
        plt.axis('tight')
        plt.axis('off')
        
        # Create statistics table
        table_data = []
        for dataset in self.datasets:
            counts = self.results.get(dataset, {})
            total = sum(counts.values())
            row = [dataset, total]
            for class_id in sorted(self.class_names.keys()):
                count = counts.get(class_id, 0)
                percentage = (count / total * 100) if total > 0 else 0
                row.append(f"{count} ({percentage:.1f}%)")
            table_data.append(row)
        
        # Add total row
        total_row = ['Total']
        grand_total = sum(sum(counts.values()) for counts in self.results.values())
        total_row.append(grand_total)
        for class_id in sorted(self.class_names.keys()):
            class_total = sum(counts.get(class_id, 0) for counts in self.results.values())
            percentage = (class_total / grand_total * 100) if grand_total > 0 else 0
            total_row.append(f"{class_total} ({percentage:.1f}%)")
        table_data.append(total_row)
        
        columns = ['Dataset', 'Total_Labels'] + [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        table = plt.table(cellText=table_data, colLabels=columns,
                         cellLoc='center', loc='center', 
                         colWidths=[0.15] + [0.2] * (len(columns) - 1))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Set table style
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Set total row style
        for i in range(len(columns)):
            table[(len(table_data), i)].set_facecolor('#E0E0E0')
            table[(len(table_data), i)].set_text_props(weight='bold')
        
        plt.title('YOLO Dataset Detailed Statistics Table', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_Detailed_Statistics_Table.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAll visualization charts saved to: {output_dir}")
        
        # Save CSV statistics data
        csv_path = os.path.join(output_dir, 'dataset_statistics.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Detailed statistics data saved to: {csv_path}")
    def create_total_summary_table(self, output_dir):
        """
        Create total summary statistics table
        
        Args:
            output_dir: Output directory
        """
        plt.figure(figsize=(14, 10))
        plt.axis('tight')
        plt.axis('off')
        
        # Calculate various statistics
        dataset_stats = []
        class_stats = []
        
        # 1. Statistics by dataset
        for dataset in self.datasets:
            counts = self.results.get(dataset, {})
            total = sum(counts.values())
            
            # Calculate count and percentage for each category
            class_details = []
            for class_id in sorted(self.class_names.keys()):
                count = counts.get(class_id, 0)
                percentage = (count / total * 100) if total > 0 else 0
                class_details.append(f"{count} ({percentage:.1f}%)")
            
            dataset_stats.append([
                dataset.upper(),
                total,
                *class_details
            ])
        
        # 2. Calculate totals
        grand_total = sum(sum(counts.values()) for counts in self.results.values())
        total_class_details = []
        for class_id in sorted(self.class_names.keys()):
            class_total = sum(counts.get(class_id, 0) for counts in self.results.values())
            percentage = (class_total / grand_total * 100) if grand_total > 0 else 0
            total_class_details.append(f"{class_total} ({percentage:.1f}%)")
            
            # Prepare data for class statistics table
            class_stats.append([
                self.class_names[class_id],
                class_total,
                f"{percentage:.1f}%"
            ])
        
        dataset_stats.append([
            "TOTAL",
            grand_total,
            *total_class_details
        ])
        
        # Create main statistics table
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Table 1: Detailed dataset statistics
        ax1.axis('tight')
        ax1.axis('off')
        
        columns1 = ['Dataset', 'Total_Labels'] + [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        table1 = ax1.table(cellText=dataset_stats, colLabels=columns1,
                          cellLoc='center', loc='center',
                          colWidths=[0.15, 0.15] + [0.23] * (len(self.class_names)))
        
        table1.auto_set_font_size(False)
        table1.set_fontsize(11)
        table1.scale(1, 2.5)
        
        # Set table 1 style
        for i in range(len(columns1)):
            table1[(0, i)].set_facecolor('#2E7D32')  # Dark green
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        # Set data row style
        for i in range(1, len(dataset_stats)):
            for j in range(len(columns1)):
                if i == len(dataset_stats) - 1:  # Total row
                    table1[(i, j)].set_facecolor('#E8F5E8')
                    table1[(i, j)].set_text_props(weight='bold')
                else:
                    if i % 2 == 0:
                        table1[(i, j)].set_facecolor('#F5F5F5')
                    else:
                        table1[(i, j)].set_facecolor('#FFFFFF')
        
        ax1.set_title('YOLO Dataset Detailed Statistics - Grouped by Dataset', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Table 2: Category summary statistics
        ax2.axis('tight')
        ax2.axis('off')
        
        # Add additional statistical information
        summary_stats = [
            ['Total Datasets', len(self.datasets), 'items'],
            ['Total Categories', len(self.class_names), 'items'],
            ['Total Labels', grand_total, 'items'],
            ['Average Labels per Category', grand_total // len(self.class_names) if len(self.class_names) > 0 else 0, 'items'],
        ]
        
        # Add file count statistics for each dataset
        for dataset in self.datasets:
            labels_dir = os.path.join(self.data_root, dataset, 'labels')
            if os.path.exists(labels_dir):
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'labels.cache']
                file_count = len(label_files)
                summary_stats.append([f'{dataset.upper()} Files', file_count, 'items'])
        
        # Combine table data
        combined_stats = []
        
        # Left side: Category statistics
        combined_stats.append(['Category Statistics', '', '', 'Basic Information', '', ''])
        combined_stats.append(['Category Name', 'Label Count', 'Percentage', 'Statistics Item', 'Value', 'Unit'])
        
        max_rows = max(len(class_stats), len(summary_stats))
        for i in range(max_rows):
            left_row = class_stats[i] if i < len(class_stats) else ['', '', '']
            right_row = summary_stats[i] if i < len(summary_stats) else ['', '', '']
            combined_stats.append(left_row + right_row)
        
        table2 = ax2.table(cellText=combined_stats[2:], colLabels=combined_stats[1],
                          cellLoc='center', loc='center',
                          colWidths=[0.15, 0.12, 0.08, 0.15, 0.12, 0.08])
        
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1, 2.2)
        
        # Set table 2 style
        for i in range(6):
            table2[(0, i)].set_facecolor('#1976D2')  # Blue
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        # Set data row style
        for i in range(1, len(combined_stats) - 1):
            for j in range(6):
                if j < 3:  # Left side category statistics
                    table2[(i, j)].set_facecolor('#E3F2FD')
                else:  # Right side basic information
                    table2[(i, j)].set_facecolor('#FFF3E0')
        
        ax2.set_title('YOLO Dataset Summary Statistics - Category Distribution & Basic Info', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7_Total_Summary_Statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create simplified overview table
        plt.figure(figsize=(12, 8))
        plt.axis('tight')
        plt.axis('off')
        
        # Create overview data
        overview_data = [
            ['Metric', 'Value', 'Note'],
            ['Dataset Count', f"{len(self.datasets)} items", 'train, val, test'],
            ['Category Count', f"{len(self.class_names)} items", ', '.join(self.class_names.values())],
            ['Total Labels', f"{grand_total:,} items", 'Sum of all dataset labels'],
        ]
        
        # Add label count for each dataset
        for dataset in self.datasets:
            counts = self.results.get(dataset, {})
            total = sum(counts.values())
            percentage = (total / grand_total * 100) if grand_total > 0 else 0
            overview_data.append([
                f'{dataset.upper()} Dataset',
                f"{total:,} items",
                f"Accounts for {percentage:.1f}% of total"
            ])
        
        # Add statistics for each category
        for class_id in sorted(self.class_names.keys()):
            class_total = sum(counts.get(class_id, 0) for counts in self.results.values())
            percentage = (class_total / grand_total * 100) if grand_total > 0 else 0
            overview_data.append([
                self.class_names[class_id],
                f"{class_total:,} items",
                f"Accounts for {percentage:.1f}% of total"
            ])
        
        table = plt.table(cellText=overview_data[1:], colLabels=overview_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.45])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)
        
        # Set style
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Set data row style
        for i in range(1, len(overview_data)):
            for j in range(3):
                if i <= 3:  # Basic information rows
                    table[(i, j)].set_facecolor('#E8F5E8')
                elif i <= 3 + len(self.datasets):  # Dataset information rows
                    table[(i, j)].set_facecolor('#FFF3E0')
                else:  # Category information rows
                    table[(i, j)].set_facecolor('#E3F2FD')
                
                if j == 1:  # Bold value column
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.title('YOLO Dataset Overview Statistics', fontsize=18, fontweight='bold', pad=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '8_Dataset_Overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Total summary statistics tables saved to: {output_dir}")
        print("  - 7_Total_Summary_Statistics.png: Detailed grouped statistics")
        print("  - 8_Dataset_Overview.png: Simplified overview statistics")
        
    def print_summary(self):
        """
        Print statistical summary
        """
        print("\n" + "=" * 60)
        print("YOLO Dataset Statistics Summary")
        print("=" * 60)
        
        total_labels = 0
        for dataset, counts in self.results.items():
            dataset_total = sum(counts.values())
            total_labels += dataset_total
            print(f"\n{dataset.upper()} Dataset:")
            print(f"  Total Labels: {dataset_total}")
            
            for class_id in sorted(self.class_names.keys()):
                count = counts.get(class_id, 0)
                percentage = (count / dataset_total * 100) if dataset_total > 0 else 0
                class_name = self.class_names[class_id]
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nOverall Statistics:")
        print(f"  Total Datasets: {len(self.datasets)}")
        print(f"  Total Categories: {len(self.class_names)}")
        print(f"  Total Labels: {total_labels}")
        
        # Total count statistics for each category
        print(f"\nTotal Count by Category:")
        for class_id in sorted(self.class_names.keys()):
            class_total = sum(counts.get(class_id, 0) for counts in self.results.values())
            percentage = (class_total / total_labels * 100) if total_labels > 0 else 0
            class_name = self.class_names[class_id]
            print(f"  {class_name}: {class_total} ({percentage:.1f}%)")


def main():
    """
    Main function
    """
    # Dataset root directory
    data_root = r"C:\Users\14854\Desktop\BTD\yoloserver\data"
    output_dir = os.path.join(data_root, "data_count")
    
    # Create analyzer
    analyzer = YOLODatasetAnalyzer(data_root)
    
    # Execute analysis
    analyzer.analyze_all_datasets()
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualization charts
    print(f"\nGenerating visualization charts...")
    analyzer.create_visualizations(output_dir)
    
    # Create total summary statistics table
    print(f"\nGenerating total summary statistics table...")
    analyzer.create_total_summary_table(output_dir)
    
    print(f"\nAnalysis completed! Check results:")
    print(f"  - Chart files: {output_dir}")
    print(f"  - Statistics data: {os.path.join(output_dir, 'dataset_statistics.csv')}")
    print(f"  - Total summary table: {output_dir}/7_Total_Summary_Statistics.png")
    print(f"  - Dataset overview: {output_dir}/8_Dataset_Overview.png")


if __name__ == "__main__":
    main()
