# IUI PhD Project: Infertility Prediction Model (XAI Approach)

โปรเจค Machine Learning สำหรับวิทยานิพนธ์ระดับปริญญาเอก เน้นกระบวนการ Explainable AI (SHAP) 
และการหา Feature Budget ที่เหมาะสมจากการเปรียบเทียบ Imbalanced Data Techniques

## วิธีการใช้งาน (How to Run)
1. นำไฟล์ `final_coding.xlsx - final.csv` ไปวางในโฟลเดอร์ `data/raw/`
2. ติดตั้งไลบรารี: `pip install -r requirements.txt`
3. รันการเตรียมข้อมูล: `python src/data_prep.py`
4. รันการเทรนและวิเคราะห์โมเดลผ่าน Jupyter Notebook: 
   เปิด Terminal แล้วพิมพ์ `jupyter notebook` 
   จากนั้นเปิดไฟล์ `src/model_training.ipynb` และรันทีละ Cell
