import pandas as pd

from src.features import FEATURES, engineer


def _row(**kw):
    base = dict(Product_Category="Home", Product_Price_INR=1000.0, Discount_Percent=10.0,
                Expected_Delivery_Days=5, Actual_Delivery_Days=8, Customer_Rating_PreOrder=4.0,
                Previous_Returns=2, Previous_Orders=10, Payment_Method="UPI",
                Damaged_Package_Flag="No", Package_Size="Small", Promotion_Active="No")
    base.update(kw)
    return pd.DataFrame([base])


def test_engineered_features():
    df = engineer(_row())
    assert df.loc[0, "Delivery_Delay"] == 3
    assert df.loc[0, "Is_Late"] == 1
    assert df.loc[0, "Discount_Amount_INR"] == 100
    assert df.loc[0, "Final_Price_INR"] == 900
    assert df.loc[0, "Return_Frequency"] == 0.2
    assert all(c in df.columns for c in FEATURES)


def test_invalid_values_become_nan():
    df = engineer(_row(Product_Price_INR=-200, Discount_Percent=125))
    assert pd.isna(df.loc[0, "Product_Price_INR"])
    assert pd.isna(df.loc[0, "Discount_Percent"])
