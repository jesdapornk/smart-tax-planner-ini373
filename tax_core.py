"""ตรรกะหลักของ Smart Tax Planner (ภาษี + โมเดล K-Means + ระบบแนะนำ) แยกจากหน้าเว็บเพื่อทดสอบได้"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "log_income", "risk_score", "deduction_ratio"]

BRACKETS = [  # (เพดานของขั้น, อัตรา) ตรวจกับประกาศกรมสรรพากรของปีภาษีที่ใช้ก่อนนำไปใช้จริง
    (150_000, 0.00), (300_000, 0.05), (500_000, 0.10), (750_000, 0.15),
    (1_000_000, 0.20), (2_000_000, 0.25), (5_000_000, 0.30), (float("inf"), 0.35),
]

TRAIN_RANGE = {"age": (20, 75), "income": (120_000, 6_000_000)}   # ช่วงข้อมูลที่ใช้ฝึกโมเดล
RISK_LABELS = {1: "1 ต่ำมาก", 2: "2 ต่ำ", 3: "3 ปานกลาง", 4: "4 สูง", 5: "5 สูงมาก"}


def income_tax(net_income):
    tax, lower = 0.0, 0.0
    for upper, rate in BRACKETS:
        if net_income > lower:
            tax += (min(net_income, upper) - lower) * rate
        lower = upper
    return tax


def net_income_before_funds(income, dependents, insurance, pvd, age=40):
    senior_exempt = 190_000 if age >= 65 else 0
    expense = min(0.5 * income, 100_000)
    deductions = 60_000 + 30_000 * dependents + 9_000 + insurance + pvd
    return max(income - senior_exempt - expense - deductions, 0.0)


def equity_share_from_risk(risk):
    return float(np.clip(0.10 + 0.80 * (risk - 1) / 4, 0.10, 0.90))


def describe_cluster(age_mean, risk_mean, income_mean):
    age_label = "วัยทำงาน (20-49)" if age_mean < 50 else "วัยใกล้เกษียณ (50-59)" if age_mean < 60 else "ผู้สูงอายุ (60+)"
    risk_label = "รับความเสี่ยงต่ำ" if risk_mean < 2.5 else "รับความเสี่ยงปานกลาง" if risk_mean < 3.5 else "รับความเสี่ยงสูง"
    income_label = "รายได้สูง" if income_mean >= 800_000 else "รายได้ปานกลาง" if income_mean >= 450_000 else "รายได้น้อย"
    return f"{age_label} · {income_label} · {risk_label}"


def load_model(root):
    """โหลดโมเดล; ถ้าเวอร์ชัน scikit-learn ไม่ตรงจนโหลดไม่ได้ ให้ฝึกใหม่จากข้อมูลชุดเดิมด้วย seed เดิม"""
    root = Path(root)
    # หาไฟล์ในโฟลเดอร์ outputs ก่อน ถ้าไม่มีให้หาที่หน้าแรกของ repo (กรณีอัปโหลดไฟล์ไว้ชั้นนอกสุด)
    base = next((d for d in (root / "outputs", root) if (d / "model_info.json").exists() and (d / "users_synthetic.csv").exists()), None)
    if base is None:
        raise FileNotFoundError("ไม่พบ model_info.json และ users_synthetic.csv ทั้งในโฟลเดอร์ outputs และหน้าแรกของ repo")
    info = json.loads((base / "model_info.json").read_text(encoding="utf-8"))
    df = pd.read_csv(base / "users_synthetic.csv")
    df["log_income"] = np.log(df["income"])
    train, _ = train_test_split(df, test_size=0.2, random_state=info["random_seed"])
    try:
        model = joblib.load(base / "kmeans_model.joblib")
        model.predict(train[FEATURES].head(2))
        source = "โหลดจากไฟล์ .joblib"
    except Exception:
        model = Pipeline([("scaler", StandardScaler()),
                          ("kmeans", KMeans(n_clusters=info["n_clusters"], n_init=20, random_state=info["random_seed"]))]).fit(train[FEATURES])
        source = "ฝึกใหม่จากข้อมูลชุดเดิม (เวอร์ชันไลบรารีไม่ตรงกับไฟล์โมเดล)"
    train = train.assign(cluster=model.predict(train[FEATURES]))
    profile = train.groupby("cluster").agg(n=("age", "size"), age=("age", "mean"), income=("income", "mean"),
                                           risk_score=("risk_score", "mean"), deduction_ratio=("deduction_ratio", "mean"))
    profile["description"] = [describe_cluster(r.age, r.risk_score, r.income) for r in profile.itertuples()]
    profile["equity_share"] = [equity_share_from_risk(r) for r in profile["risk_score"]]
    return model, info, profile, source


def assign_cluster(model, age, income, risk, insurance, pvd, dependents):
    inc = max(float(income), 1.0)
    row = pd.DataFrame([{"age": age, "log_income": np.log(inc), "risk_score": risk,
                         "deduction_ratio": (insurance + pvd + 30_000 * dependents) / inc}])
    return int(model.predict(row[FEATURES])[0])


def plan(age, income, dependents, insurance, pvd, budget, equity_share, invest_override=None):
    """คืนแผนลงทุนใน RMF/ThaiESG และภาษีที่ประหยัดได้"""
    net = net_income_before_funds(income, dependents, insurance, pvd, age)
    rmf_cap = max(0.0, min(0.30 * income, 500_000 - pvd))
    esg_cap = max(0.0, min(0.30 * income, 300_000))
    cap_total = rmf_cap + esg_cap
    wanted = budget if invest_override is None else invest_override
    invest = float(min(max(wanted, 0.0), cap_total))
    if age >= 45:                                   # ใช้ RMF ก่อน (เน้นออมเกษียณ) สมมติฐานการออกแบบของทีม
        rmf = min(invest, rmf_cap); esg = invest - rmf
    else:
        esg = min(invest, esg_cap); rmf = invest - esg
    tax_before = income_tax(net)
    tax_after = income_tax(max(net - invest, 0.0))
    tax_max = income_tax(max(net - cap_total, 0.0))
    return {"net_income": net, "rmf_cap": rmf_cap, "esg_cap": esg_cap, "cap_total": cap_total,
            "invest": invest, "rmf": rmf, "thaiesg": esg,
            "equity_amount": invest * equity_share, "bond_amount": invest * (1 - equity_share),
            "tax_before": tax_before, "tax_after": tax_after, "tax_saved": tax_before - tax_after,
            "max_saved": tax_before - tax_max, "limited_by_cap": wanted > cap_total + 1e-9}
