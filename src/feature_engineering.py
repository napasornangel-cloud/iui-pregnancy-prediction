import pandas as pd
import numpy as np
import os


def add_sperm_wash_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sperm wash improvement ratios
    ดูว่าหลัง wash สเปิร์มดีขึ้นแค่ไหนเทียบกับก่อน wash
    """
    print("--- 1. Sperm Wash Ratios ---")
    data = df.copy()

    # ratio TPMSC หลัง/ก่อน wash — ยิ่งสูงยิ่งดี
    if {'Post_TPMSC', 'Pre_TPMSC'}.issubset(data.columns):
        data['Ratio_TPMSC'] = np.where(
            data['Pre_TPMSC'] > 0,
            data['Post_TPMSC'] / data['Pre_TPMSC'],
            np.nan
        )

    # delta progressive motility หลัง - ก่อน wash
    if {'Post_Progressive_Motile', 'Pre_Progressive_Motile'}.issubset(data.columns):
        data['Delta_Progressive_Motile'] = (
            data['Post_Progressive_Motile'] - data['Pre_Progressive_Motile']
        )

    # delta motility หลัง - ก่อน wash
    if {'Post_Motile', 'Pre_Motile'}.issubset(data.columns):
        data['Delta_Motile'] = data['Post_Motile'] - data['Pre_Motile']

    return data


def add_cycle_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cycle quality features
    ดู readiness ของ cycle นั้นๆ สำหรับ IUI
    """
    print("--- 2. Cycle Quality Features ---")
    data = df.copy()

    # combined cycle quality — follicle ใหญ่พอ + endometrium หนาพอ
    if {'Mature_Follicle_Count', 'Endometrium_Thickness'}.issubset(data.columns):
        data['Follicle_Endo_Product'] = (
            data['Mature_Follicle_Count'] * data['Endometrium_Thickness']
        )

    # cumulative treatment burden — ทำมากี่รอบแล้ว
    if {'Cycle_Number', 'Ovary_Stimulation_Round'}.issubset(data.columns):
        data['Cumulative_Treatment'] = (
            data['Cycle_Number'] * data['Ovary_Stimulation_Round']
        )

    return data


def add_female_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Female factor interaction features
    ดู interaction ระหว่าง factors ของฝ่ายหญิง
    """
    print("--- 3. Female Interaction Features ---")
    data = df.copy()

    # อายุ x FSH — ovarian reserve proxy
    # FSH สูง + อายุมาก = ovarian reserve ต่ำ
    if {'Age_Female', 'FSH_Baseline'}.issubset(data.columns):
        data['Age_FSH_Interaction'] = (
            data['Age_Female'] * data['FSH_Baseline']
        )

    # BMI x Infertility_Type
    if {'Body_Mass_Index', 'Infertility_Type'}.issubset(data.columns):
        data['BMI_InfertilityType_Interaction'] = (
            data['Body_Mass_Index'] * data['Infertility_Type']
        )

    # total female pathology burden — นับจำนวน factors ที่มีปัญหา
    factor_cols = [
        'Uterine_Factors', 'Tubal_Factors', 'Ovarian_Factors',
        'Ovulatory_Factors', 'Cervical_Factors',
        'Endometriosis_Factors', 'Multisystem_Factors'
    ]
    available = [c for c in factor_cols if c in data.columns]
    if available:
        data['Total_Female_Pathology'] = data[available].sum(axis=1, min_count=1)

    return data


def add_sperm_quality_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sperm quality index จาก first sample
    ดูคุณภาพโดยรวมของสเปิร์มตอนมาตรวจครั้งแรก
    """
    print("--- 4. Sperm Quality Index ---")
    data = df.copy()

    # total motile sperm จาก first sample
    if {'First_Count', 'First_Motile', 'First_Volume'}.issubset(data.columns):
        data['First_TotalMotile'] = (
            data['First_Count'] *
            data['First_Motile'] / 100 *
            data['First_Volume']
        )

    return data


def run_feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print(f"โหลดข้อมูล: {df.shape}")

    df = add_sperm_wash_ratios(df)
    print(f"  shape: {df.shape}")

    df = add_cycle_quality_features(df)
    print(f"  shape: {df.shape}")

    df = add_female_interaction_features(df)
    print(f"  shape: {df.shape}")

    df = add_sperm_quality_index(df)
    print(f"  shape: {df.shape}")

    # log features ที่เพิ่มมา
    new_features = [
        'Ratio_TPMSC', 'Delta_Progressive_Motile', 'Delta_Motile',
        'Follicle_Endo_Product', 'Cumulative_Treatment',
        'Age_FSH_Interaction', 'BMI_InfertilityType_Interaction',
        'Total_Female_Pathology', 'First_TotalMotile'
    ]
    added = [f for f in new_features if f in df.columns]
    print(f"\nFeatures ที่เพิ่มมาทั้งหมด ({len(added)}):")
    for f in added:
        missing_pct = df[f].isna().mean() * 100
        print(f"  {f}: missing {missing_pct:.1f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ บันทึกไฟล์ที่: {output_path}")
    print(f"Shape สุดท้าย: {df.shape}")

    return df


if __name__ == "__main__":
    input_path  = "data/processed/cycle_level_ready_for_ml.csv"
    output_path = "data/processed/cycle_level_features.csv"
    df = run_feature_engineering(input_path, output_path)