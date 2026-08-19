import pandas as pd

# Load the fixed scores
df = pd.read_csv("output/final_test_dice_scores_fixed.csv")

# Sort by Mean_Dice
df_sorted = df.sort_values(by="Mean_Dice", ascending=False)

top_3 = df_sorted.head(3)
bottom_3 = df_sorted.tail(3)

print("--- TOP 3 PERFORMING CASES ---")
print(top_3[['Case_ID', 'Mean_Dice']])

print("\n--- BOTTOM 3 PERFORMING CASES ---")
print(bottom_3[['Case_ID', 'Mean_Dice']])