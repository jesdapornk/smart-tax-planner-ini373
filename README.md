# Smart Tax Planner (INI 373)
เว็บวางแผนภาษีเงินได้บุคคลธรรมดาด้วย K-Means · ทีม "อ้วนกลมอารมณ์เสีย"

## ไฟล์
- `INI373_SmartTaxPlanner.ipynb` Notebook (ข้อมูลจำลอง → ภาษี → K-Means → ประเมิน → บันทึกโมเดล)
- `app.py` หน้าเว็บ Streamlit · `tax_core.py` ตรรกะภาษี/โมเดล · `test_logic.py` ทดสอบ (`python test_logic.py`)
- `outputs/` โมเดล `.joblib`, `model_info.json`, `users_synthetic.csv`

## รันในเครื่อง
```
pip install -r requirements.txt
streamlit run app.py
```

## นำขึ้น Streamlit Community Cloud
1. อัปโหลดทุกไฟล์ในโฟลเดอร์นี้ขึ้น GitHub repository (public)
2. เข้า share.streamlit.io → New app → เลือก repo, branch `main`, Main file path `app.py` → Deploy
3. คัดลอกลิงก์เว็บไปใส่สไลด์สุดท้ายและรายงาน

> ข้อมูลเป็นข้อมูลจำลอง และไม่ใช่คำแนะนำทางการเงิน/ภาษีอย่างเป็นทางการ
