import os
import pandas as pd
import matplotlib.pyplot as plt

# Load your generated CSV
df = pd.read_csv("output/final_test_dice_scores_fixed.csv")

# Create the plot
plt.figure(figsize=(10, 6))
df.boxplot(column=['Necrotic', 'Edema', 'Enhancing'])

plt.title('Dice Scores by Tumor Region (T1-only Model)')
plt.ylabel('Dice Similarity Coefficient')
plt.ylim(-0.05, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)


os.makedirs("output", exist_ok=True)
plt.savefig('output/dice_boxplot_summary.png')
print("Boxplot saved as output/dice_boxplot_summary.png")