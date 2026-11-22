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

# สมมติฐานการหักของทีม (ผู้มีเงินเดือน เงินได้ประเภท 40(1)) ต้องตรวจกับประกาศกรมสรรพากรของปีภาษีที่ใช้
PERSONAL_ALLOWANCE = 60_000          # ค่าลดหย่อนส่วนตัว
DEPENDENT_ALLOWANCE = 30_000         # ค่าลดหย่อนผู้อยู่ในอุปการะ คนละ
SOCIAL_SECURITY = 9_000              # เงินสมทบประกันสังคม: ทีมสมมติให้ทุกคนจ่ายเต็มเพดานต่อปี
EXPENSE_RATE, EXPENSE_CAP = 0.5, 100_000   # ค่าใช้จ่ายร้อยละ 50 ไม่เกิน 100,000
SENIOR_EXEMPT = 190_000              # เงินได้ที่ยกเว้นสำหรับผู้มีอายุ 65 ปีขึ้นไป
FUND_RATE = 0.30                     # RMF/ThaiESG ไม่เกินร้อยละ 30 ของเงินได้
RMF_PVD_CAP, ESG_CAP = 500_000, 300_000  # RMF รวม PVD ไม่เกิน 500,000 / ThaiESG ไม่เกิน 300,000

TRAIN_RANGE = {"age": (20, 75), "income": (120_000, 6_000_000)}   # ช่วงข้อมูลที่ใช้ฝึกโมเดล
RISK_LABELS = {1: "1 ต่ำมาก", 2: "2 ต่ำ", 3: "3 ปานกลาง", 4: "4 สูง", 5: "5 สูงมาก"}


def income_tax(net_income):
    tax, lower = 0.0, 0.0
    for upper, rate in BRACKETS:
        if net_income > lower:
            tax += (min(net_income, upper) - lower) * rate
        lower = upper
    return tax


def deduction_detail(income, dependents, insurance, pvd, age=40):
    """แจกแจงการหักค่าใช้จ่ายและค่าลดหย่อน (ใช้ทั้งคำนวณและแสดงบนหน้าเว็บ)"""
    items = [
        ("เงินได้ที่ยกเว้น (อายุ 65 ปีขึ้นไป)", SENIOR_EXEMPT if age >= 65 else 0.0),
        (f"ค่าใช้จ่าย ({EXPENSE_RATE:.0%} ไม่เกิน {EXPENSE_CAP:,})", min(EXPENSE_RATE * income, EXPENSE_CAP)),
        ("ค่าลดหย่อนส่วนตัว", PERSONAL_ALLOWANCE),
        (f"ค่าลดหย่อนผู้อยู่ในอุปการะ ({dependents} คน)", DEPENDENT_ALLOWANCE * dependents),
        ("ประกันสังคม (สมมติฐานของทีม)", SOCIAL_SECURITY),
        ("เบี้ยประกันชีวิต/สุขภาพ", float(insurance)),
        ("กองทุนสำรองเลี้ยงชีพ (PVD)", float(pvd)),
    ]
    return items


def net_income_before_funds(income, dependents, insurance, pvd, age=40):
    total = sum(v for _, v in deduction_detail(income, dependents, insurance, pvd, age))
    return max(income - total, 0.0)


def tax_by_bracket(net_income):
    """ตารางภาษีรายขั้นบันได: (ช่วงเงินได้สุทธิ, อัตรา, เงินได้ในขั้น, ภาษีในขั้น)"""
    rows, lower = [], 0.0
    for upper, rate in BRACKETS:
        part = max(min(net_income, upper) - lower, 0.0)
        label = f"{lower:,.0f} - {upper:,.0f}" if upper != float("inf") else f"เกิน {lower:,.0f}"
        rows.append((label, rate, part, part * rate))
        lower = upper
    return rows


def k_curve(root, k_max=8):
    """Elbow (inertia) และ Silhouette ของแต่ละ k บนชุดฝึก ด้วย pipeline/seed เดียวกับโมเดล"""
    from sklearn.metrics import silhouette_score
    root = Path(root)
    base = next(d for d in (root / "outputs", root) if (d / "users_synthetic.csv").exists())
    info = json.loads((base / "model_info.json").read_text(encoding="utf-8"))
    df = pd.read_csv(base / "users_synthetic.csv"); df["log_income"] = np.log(df["income"])
    train, _ = train_test_split(df, test_size=0.2, random_state=info["random_seed"])
    X = StandardScaler().fit_transform(train[FEATURES])
    out = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=info["random_seed"]).fit(X)
        out.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(out)


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
    rmf_cap = max(0.0, min(FUND_RATE * income, RMF_PVD_CAP - pvd))
    esg_cap = max(0.0, min(FUND_RATE * income, ESG_CAP))
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
            "deductions": deduction_detail(income, dependents, insurance, pvd, age),
            "tax_before": tax_before, "tax_after": tax_after, "tax_saved": tax_before - tax_after,
            "max_saved": tax_before - tax_max, "limited_by_cap": wanted > cap_total + 1e-9}
