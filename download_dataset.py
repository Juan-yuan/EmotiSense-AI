from datasets import load_dataset
from pathlib import Path

DATASET_ID = "Sp1786/multiclass-sentiment-analysis-dataset"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(DATASET_ID)

print("\nDataset information:")
print(dataset)

for split_name, split_data in dataset.items():
    print(f"\n{split_name}: {len(split_data)} rows")
    print("Columns:", split_data.column_names)

    output_file = OUTPUT_DIR / f"{split_name}.csv"
    split_data.to_csv(str(output_file))

    print(f"Saved to: {output_file}")

print("\nDataset download completed!")