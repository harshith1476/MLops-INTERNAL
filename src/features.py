"""Shared cleaning + feature engineering (used by training AND the API)."""
import numpy as np
import pandas as pd

NUMERIC = [
    "Product_Price_INR", "Discount_Percent", "Expected_Delivery_Days",
    "Actual_Delivery_Days", "Customer_Rating_PreOrder", "Previous_Returns",
    "Previous_Orders", "Discount_Amount_INR", "Final_Price_INR",
    "Delivery_Delay", "Is_Late", "Return_Frequency",
]
CATEGORICAL = [
    "Product_Category", "Payment_Method", "Damaged_Package_Flag",
    "Package_Size", "Promotion_Active",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Return_Status"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn invalid values into NaN (they are imputed later in the pipeline)."""
    df = df.copy()
    df.loc[df["Product_Price_INR"] <= 0, "Product_Price_INR"] = np.nan
    df.loc[~df["Discount_Percent"].between(0, 100), "Discount_Percent"] = np.nan
    df.loc[~df["Customer_Rating_PreOrder"].between(1, 5), "Customer_Rating_PreOrder"] = np.nan
    for c in ["Expected_Delivery_Days", "Actual_Delivery_Days", "Previous_Returns", "Previous_Orders"]:
        df.loc[df[c] < 0, c] = np.nan
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add discount amount, final price, delivery delay, late flag, return frequency."""
    df = clean(df)
    df["Discount_Amount_INR"] = df["Product_Price_INR"] * df["Discount_Percent"] / 100
    df["Final_Price_INR"] = df["Product_Price_INR"] - df["Discount_Amount_INR"]
    df["Delivery_Delay"] = df["Actual_Delivery_Days"] - df["Expected_Delivery_Days"]
    df["Is_Late"] = (df["Delivery_Delay"] > 0).astype(float).where(df["Delivery_Delay"].notna())
    orders = df["Previous_Orders"].replace(0, np.nan)
    df["Return_Frequency"] = (df["Previous_Returns"] / orders).clip(upper=1)
    return df
