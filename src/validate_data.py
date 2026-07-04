import pandas as pd

def validate_dataset(filepath="data/cleaned_employee_dataset.csv"):
    """Automated data validation: checks the cleaned dataset
    before model training or dashboard display."""
    df = pd.read_csv(filepath)
    results = {}

    critical_cols = ["Age", "Salary", "Department", "Region"]
    results["No missing values in critical columns"] = \
        df[critical_cols].isnull().sum().sum() == 0
    results["No duplicate Employee_IDs"] = \
        df["Employee_ID"].duplicated().sum() == 0
    results["Salary within valid range (20k-200k)"] = \
        df["Salary"].between(20000, 200000).all()
    results["Age within valid range (18-70)"] = \
        df["Age"].between(18, 70).all()

    for check, passed in results.items():
        print(f"[{'PASS' if passed else 'FAIL'}]  {check}")
    return all(results.values())

if __name__ == "__main__":
    ok = validate_dataset()
    print("OVERALL:", "ALL CHECKS PASSED" if ok else "VALIDATION FAILED")
