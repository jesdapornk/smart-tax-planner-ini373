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
SPOUSE_ALLOWANCE = 60_000            # ค่าลดหย่อนคู่สมรสที่ไม่มีเงินได้
DEPENDENT_ALLOWANCE = 30_000         # ค่าลดหย่อนผู้อยู่ในอุปการะ คนละ
SOCIAL_SECURITY = 9_000              # ค่าเริ่มต้นของช่องประกันสังคม (เพดานเงินสมทบต่อปี) ผู้ใช้แก้ได้
EXPENSE_RATE, EXPENSE_CAP = 0.5, 100_000   # ค่าใช้จ่ายร้อยละ 50 ไม่เกิน 100,000
SENIOR_EXEMPT = 190_000              # เงินได้ที่ยกเว้นสำหรับผู้มีอายุ 65 ปีขึ้นไป
FUND_RATE = 0.30                     # RMF/ThaiESG ไม่เกินร้อยละ 30 ของเงินได้
RMF_PVD_CAP, ESG_CAP = 500_000, 300_000  # RMF รวม PVD ไม่เกิน 500,000 / ThaiESG ไม่เกิน 300,000

# เกณฑ์รายปีภาษี: ทีมตั้งสมมติฐานว่าพารามิเตอร์ด้านล่างเท่ากันทุกปีที่ให้เลือก (ต้องตรวจกับประกาศกรมสรรพากรก่อนใช้จริง)
# หากปีใดเกณฑ์เปลี่ยน ให้เพิ่ม/แก้ค่าในตารางนี้ ฟังก์ชันคำนวณทั้งหมดอ่านค่าจากตารางนี้ตามปีที่เลือก
_BASE = dict(brackets=BRACKETS, personal=PERSONAL_ALLOWANCE, spouse=SPOUSE_ALLOWANCE, dependent=DEPENDENT_ALLOWANCE,
             social=SOCIAL_SECURITY, expense_rate=EXPENSE_RATE, expense_cap=EXPENSE_CAP, senior=SENIOR_EXEMPT,
             fund_rate=FUND_RATE, rmf_pvd_cap=RMF_PVD_CAP, esg_cap=ESG_CAP)
TAX_RULES = {
    2567: dict(_BASE, verified=False, note="ทีมยังไม่ได้ตรวจเกณฑ์ปีนี้ ใช้ค่าเดียวกับปี 2568"),
    2568: dict(_BASE, verified=True, note="ตรวจเทียบกับคู่มือ ภ.ง.ด.90 ปีภาษี 2568 ของกรมสรรพากร (rd.go.th) แล้ว: ค่าใช้จ่าย 50% ไม่เกิน 100,000, ส่วนตัว/คู่สมรส 60,000, บุตรคนละ 30,000, ผู้มีอายุ 65 ปีขึ้นไปยกเว้น 190,000, RMF 30% ไม่เกิน 500,000 (รวมกลุ่มเกษียณ), ThaiESG 30% ไม่เกิน 300,000 (ซื้อปี 2567-2569) ประกันสังคมหักตามที่จ่ายจริง (ผู้ใช้กรอก) และบุตรคนที่ 2 ที่เกิดตั้งแต่ปี 2561 เพิ่มอีกคนละ 30,000"),
    2569: dict(_BASE, verified=False, note="ยังไม่ยืนยันเกณฑ์ปีนี้ทั้งหมด: เพดาน ThaiESG/ThaiESGX 30% ไม่เกิน 300,000 เท่าเดิม แต่ผลิตภัณฑ์ที่ใช้สิทธิและเงื่อนไขอาจต่างจากปี 2568 ต้องตรวจกับกรมสรรพากร"),
}
DEFAULT_YEAR = 2568

TRAIN_RANGE = {"age": (20, 75), "income": (120_000, 6_000_000)}   # ช่วงข้อมูลที่ใช้ฝึกโมเดล
RISK_LABELS = {1: "1 ต่ำมาก", 2: "2 ต่ำ", 3: "3 ปานกลาง", 4: "4 สูง", 5: "5 สูงมาก"}


def income_tax(net_income, year=DEFAULT_YEAR):
    tax, lower = 0.0, 0.0
    for upper, rate in TAX_RULES[year]["brackets"]:
        if net_income > lower:
            tax += (min(net_income, upper) - lower) * rate
        lower = upper
    return tax


def deduction_detail(income, dependents, insurance, pvd, age=40, spouse=False, year=DEFAULT_YEAR, social=None, kids_2561=0):
    """แจกแจงการหักค่าใช้จ่ายและค่าลดหย่อน (ใช้ทั้งคำนวณและแสดงบนหน้าเว็บ)"""
    r = TAX_RULES[year]
    items = [
        ("เงินได้ที่ยกเว้น (อายุ 65 ปีขึ้นไป)", r["senior"] if age >= 65 else 0.0),
        (f"ค่าใช้จ่าย ({r['expense_rate']:.0%} ไม่เกิน {r['expense_cap']:,})", min(r["expense_rate"] * income, r["expense_cap"])),
        ("ค่าลดหย่อนส่วนตัว", r["personal"]),
        ("ค่าลดหย่อนคู่สมรส (ไม่มีเงินได้)", r["spouse"] if spouse else 0.0),
        (f"ค่าลดหย่อนผู้อยู่ในอุปการะ ({dependents} คน)", r["dependent"] * dependents),
        (f"บุตรคนที่ 2 เป็นต้นไปที่เกิดตั้งแต่ปี 2561 เพิ่มอีกคนละ ({min(kids_2561, dependents)} คน)", r["dependent"] * min(kids_2561, dependents)),
        ("ประกันสังคม (ตามที่จ่ายจริง)", float(r["social"] if social is None else social)),
        ("เบี้ยประกันชีวิต/สุขภาพ", float(insurance)),
        ("กองทุนสำรองเลี้ยงชีพ (PVD)", float(pvd)),
    ]
    return items


def net_income_before_funds(income, dependents, insurance, pvd, age=40, spouse=False, year=DEFAULT_YEAR, social=None, kids_2561=0):
    total = sum(v for _, v in deduction_detail(income, dependents, insurance, pvd, age, spouse, year, social, kids_2561))
    return max(income - total, 0.0)


def tax_by_bracket(net_income, year=DEFAULT_YEAR):
    """ตารางภาษีรายขั้นบันได: (ช่วงเงินได้สุทธิ, อัตรา, เงินได้ในขั้น, ภาษีในขั้น)"""
    rows, lower = [], 0.0
    for upper, rate in TAX_RULES[year]["brackets"]:
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


def plan(age, income, dependents, insurance, pvd, budget, equity_share, invest_override=None, spouse=False, year=DEFAULT_YEAR, social=None, kids_2561=0):
    r = TAX_RULES[year]
    """คืนแผนลงทุนใน RMF/ThaiESG และภาษีที่ประหยัดได้"""
    net = net_income_before_funds(income, dependents, insurance, pvd, age, spouse, year, social, kids_2561)
    rmf_cap = max(0.0, min(r["fund_rate"] * income, r["rmf_pvd_cap"] - pvd))
    esg_cap = max(0.0, min(r["fund_rate"] * income, r["esg_cap"]))
    cap_total = rmf_cap + esg_cap
    wanted = budget if invest_override is None else invest_override
    invest = float(min(max(wanted, 0.0), cap_total))
    if age >= 45:                                   # ใช้ RMF ก่อน (เน้นออมเกษียณ) สมมติฐานการออกแบบของทีม
        rmf = min(invest, rmf_cap); esg = invest - rmf
    else:
        esg = min(invest, esg_cap); rmf = invest - esg
    tax_before = income_tax(net, year)
    tax_after = income_tax(max(net - invest, 0.0), year)
    max_invest = float(min(cap_total, net))          # ลงทุนเกินเงินได้สุทธิไม่ช่วยลดภาษีเพิ่ม
    if age >= 45:
        rmf_max = min(max_invest, rmf_cap); esg_max = max_invest - rmf_max
    else:
        esg_max = min(max_invest, esg_cap); rmf_max = max_invest - esg_max
    tax_max = income_tax(max(net - max_invest, 0.0), year)
    return {"net_income": net, "rmf_cap": rmf_cap, "esg_cap": esg_cap, "cap_total": cap_total,
            "invest": invest, "rmf": rmf, "thaiesg": esg,
            "equity_amount": invest * equity_share, "bond_amount": invest * (1 - equity_share),
            "deductions": deduction_detail(income, dependents, insurance, pvd, age, spouse, year, social, kids_2561),
            "tax_before": tax_before, "tax_after": tax_after, "tax_saved": tax_before - tax_after,
            "max_saved": tax_before - tax_max, "max_invest": max_invest, "rmf_max": rmf_max, "esg_max": esg_max, "tax_after_max": tax_max, "limited_by_cap": wanted > cap_total + 1e-9}


def fund_type_for(equity_share):
    """จับคู่สัดส่วนหุ้นที่แนะนำกับประเภทกองทุนในรายการตัวอย่าง"""
    return "หุ้น" if equity_share >= 0.60 else "ผสม" if equity_share >= 0.35 else "ตราสารหนี้"


def load_funds(root):
    root = Path(root)
    for d in (root / "outputs", root):
        if (d / "funds_reference.csv").exists():
            return pd.read_csv(d / "funds_reference.csv")
    return pd.DataFrame(columns=["fund_name", "scheme", "fund_type", "policy_summary", "source_url", "retrieved"])
