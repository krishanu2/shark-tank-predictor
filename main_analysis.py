import pandas as pd

# ✅ Step 1: Read the CSV exported from Excel
df = pd.read_csv('shark_tank_us.csv', encoding='ISO-8859-1', engine='python', on_bad_lines='skip')

# ✅ Step 2: Clean column names (remove spaces, lowercase)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# ✅ Step 3: Preview raw data
print("🔍 Raw Data Preview:")
print(df.head())
print("\n📊 Column Types:")
print(df.dtypes)
print("\n🧼 Null Value Count:")
print(df.isnull().sum())

# ✅ Step 4: Clean monetary columns (remove $, commas) if present
money_cols = ['original_ask_amount', 'total_deal_amount', 'valuation_requested', 'deal_valuation']
for col in money_cols:
    if col in df.columns:
        try:
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)
        except:
            print(f"⚠️ Could not clean: {col}")

# ✅ Step 5: Compute implied valuation (optional insight)
if 'original_ask_amount' in df.columns and 'original_offered_equity' in df.columns:
    try:
        df['implied_valuation'] = df['original_ask_amount'] / (df['original_offered_equity'] / 100)
    except:
        df['implied_valuation'] = None
        print("⚠️ Error in implied valuation calculation")

# ✅ Step 6: Preview selected cleaned columns
preview_cols = ['startup_name', 'industry', 'original_ask_amount', 'original_offered_equity', 'got_deal', 'implied_valuation']
preview_cols = [col for col in preview_cols if col in df.columns]
print("\n✅ Cleaned Data Sample:")
print(df[preview_cols].head())

# ✅ Step 7: Export to Excel for dashboarding
df.to_excel('Cleaned_SharkTank.xlsx', index=False)
print("\n📁 Export successful: 'Cleaned_SharkTank.xlsx' created.")
