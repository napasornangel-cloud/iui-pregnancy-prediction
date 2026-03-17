import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from feature_engineering import run_feature_engineering


def clean_basic_values(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 1. Basic Cleaning ---")
    data = df.copy()

    weird_na = {"", " ", "na", "n/a", "nan", "none", "null", "-"}

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            data[col] = data[col].apply(
                lambda x: np.nan if isinstance(x, str) and x.lower() in weird_na else x
            )

    return data


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 2. Coerce Numeric Columns ---")
    data = df.copy()

    numeric_cols = [
        "Age_Female", "Age_Male", "Total_infertile_duration",
        "Pregnancy_History", "Number_Of_Alive_Children", "Number_Of_Miscarriages",
        "Infertility_Type", "Body_Mass_Index", "Menstrual",
        "Menstrual_Interval_Days", "Menstrual_Duration_Days", "Dysmenorrhea",
        "FSH_Baseline", "LH_Baseline", "E2_Baseline", "PRL_Baseline",
        "Uterine_Factors", "Tubal_Factors", "Ovarian_Factors", "Ovulatory_Factors",
        "Cervical_Factors", "Endometriosis_Factors", "Multisystem_Factors",
        "Alcohol", "Smoke",
        "First_Volume", "First_Count", "First_Motile", "First_Progressive_Motile",
        "First_Normal_Morpho",
        "Pre_Volume", "Pre_Count", "Pre_Motile", "Pre_Progressive_Motile", "Pre_TPMSC",
        "Post_Count", "Post_Motile", "Post_Progressive_Motile", "Post_TPMSC",
        "Cycle_Type", "OI_Clomiphene", "OI_Letrozole", "OI_Gonadotropins",
        "Cycle_Day", "Cycle_Number", "Ovary_Stimulation_Round", "hCG_Dose",
        "Mature_Follicle_Count", "Endometrium_Thickness",
        "Endo_Type_Triple", "Endo_Type_Intermediate", "Endo_Type_Mixed",
        "Result",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def add_hcg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    hCG_Dose ใน Excel เป็น 0/1/2:
      0 = ไม่ได้ฉีด (Natural)
      1 = Pregnyl 5000 IU
      2 = Ovidrel 6500 IU

    แตกเป็น 2 columns:
      hCG_Used : 0 = ไม่ได้ฉีด, 1 = ฉีด, NaN = ไม่มีข้อมูล
      hCG_Type : 0/1/2 ตาม encoding เดิม, NaN = ไม่มีข้อมูล
    """
    print("--- 3. Add hCG Features ---")
    data = df.copy()

    if "hCG_Dose" in data.columns:
        x = pd.to_numeric(data["hCG_Dose"], errors="coerce")

        data["hCG_Used"] = np.where(
            x.isna(), np.nan,
            np.where(x == 0, 0, 1)
        )

        data["hCG_Type"] = np.where(
            x.isna(), np.nan,
            np.where(x.isin([0, 1, 2]), x, np.nan)
        )

        unknown = x[(x.notna()) & (~x.isin([0, 1, 2]))]
        if len(unknown) > 0:
            print(f"  ⚠️ hCG_Dose พบค่านอก 0/1/2: {sorted(unknown.unique())}")

        data = data.drop(columns=["hCG_Dose"])

    return data


def clean_gynecological_surgical_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง Gynecological_Surgical_History เป็น binary 0/1:
      0 = ไม่มีประวัติผ่าตัด
      1 = มีประวัติผ่าตัดอย่างน้อย 1 อย่าง
      NaN = ไม่มีข้อมูล
    """
    print("--- 4. Clean Gynecological_Surgical_History ---")
    data = df.copy()

    if "Gynecological_Surgical_History" in data.columns:
        false_like = {"0", "0.0", "none", "false", "no", ""}
        missing_like = {"nan", "na", "n/a", "null"}

        s = data["Gynecological_Surgical_History"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

        data["Gynecological_Surgical_History"] = s.apply(
            lambda x: 0 if (x in false_like or x == 0)
            else (np.nan if pd.isna(x) or x in missing_like else 1)
        )

    return data


def filter_cycle_1_to_3(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 5. Filter Cycle_Number 1-3 ---")
    data = df.copy()

    if "Cycle_Number" in data.columns:
        data["Cycle_Number"] = pd.to_numeric(data["Cycle_Number"], errors="coerce")

        before = len(data)
        data = data[data["Cycle_Number"].isin([1, 2, 3])]
        print(f"  Dropped {before - len(data)} rows (Cycle_Number not in 1-3 or NaN)")

    return data


def final_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 6. Final Sanity Checks ---")
    data = df.copy()

    if "Result" in data.columns:
        data["Result"] = pd.to_numeric(data["Result"], errors="coerce")
        data = data.dropna(subset=["Result"])

        valid_vals = {0, 1, 0.0, 1.0}
        bad_vals = set(data["Result"].unique()) - valid_vals
        if bad_vals:
            raise ValueError(f"Result มีค่าที่ไม่ใช่ 0/1: {sorted(bad_vals)}")

    if "HN" in data.columns:
        data = data.dropna(subset=["HN"])
        data["HN"] = data["HN"].astype(str).str.strip()
        data = data[data["HN"] != ""]

    if {"HN", "Cycle_Number"}.issubset(data.columns):
        dup = data.duplicated(subset=["HN", "Cycle_Number"]).sum()
        print(f"  Duplicate (HN, Cycle_Number): {dup}")
        if dup > 0:
            print(f"  ⚠️ พบ duplicate {dup} rows — ควรตรวจสอบก่อน proceed")

    print(f"  Has Date column: {'Date' in data.columns}")
    print(f"  Result distribution:\n{data['Result'].value_counts(dropna=False).to_string()}")

    return data


if __name__ == "__main__":
    raw_path        = "data/raw/final_coding.xlsx"
    clean_path      = "data/processed/cycle_level_ready_for_ml.csv"
    features_path   = "data/processed/cycle_level_features.csv"

    try:
        df = pd.read_excel(raw_path, sheet_name="final")
        print(f"โหลดข้อมูลสำเร็จ รูปแบบข้อมูลเริ่มต้น: {df.shape}")

        df = clean_basic_values(df)
        print(f"  shape: {df.shape}")

        df = coerce_numeric_columns(df)
        print(f"  shape: {df.shape}")

        df = add_hcg_features(df)
        print(f"  shape: {df.shape}")

        df = clean_gynecological_surgical_history(df)
        print(f"  shape: {df.shape}")

        df = filter_cycle_1_to_3(df)
        print(f"  shape: {df.shape}")

        df = final_sanity_checks(df)
        print(f"\nหลัง cleaning และกรอง cycle 1-3: {df.shape}")

        # save clean version ก่อน
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        df.to_csv(clean_path, index=False)
        print(f"\n✅ Clean data บันทึกที่: {clean_path}")

        # ต่อด้วย feature engineering ทันที
        print("\n--- Feature Engineering ---")
        run_feature_engineering(clean_path, features_path)

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        raise