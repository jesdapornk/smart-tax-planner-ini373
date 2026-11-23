import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import tax_core as tc

st.set_page_config(page_title="Smart Tax Planner", page_icon="💰", layout="centered")
st.markdown("""
<style>
html, body, [class*="st-"] { font-size: 18px; }
h1 { font-size: 2.0rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.7rem; }
.stButton button { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_k_curve():
    return tc.k_curve(ROOT)


@st.cache_resource
def get_model():
    return tc.load_model(ROOT)


model, info, profile, source = get_model()
baht = lambda x: f"{x:,.0f} บาท"

PERSONAS = {
    "กรอกข้อมูลเอง": None,
    "คุณสมชาย (55 ปี เงินเดือนสูง รับความเสี่ยงต่ำ)": dict(age=55, income=700_000, dependents=1, insurance=20_000, pvd=0, risk=2, budget=70_000),
    "คุณนที (27 ปี เพิ่งเริ่มงาน รับความเสี่ยงสูง)": dict(age=27, income=420_000, dependents=0, insurance=0, pvd=0, risk=4, budget=25_000),
    "คุณป้าสมศรี (68 ปี มีบำนาญและเงินลงทุน)": dict(age=68, income=500_000, dependents=0, insurance=15_000, pvd=0, risk=1, budget=40_000),
}
DEFAULTS = dict(year=tc.DEFAULT_YEAR, spouse=False, age=35, income=600_000, dependents=0, insurance=0, pvd=0, risk=3, budget=50_000)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_persona():
    p = PERSONAS[st.session_state["persona"]]
    if p:
        for k, v in p.items():
            st.session_state[k] = v


st.title("💰 Smart Tax Planner")
st.caption("ประเมินภาษีเงินได้บุคคลธรรมดา และแนะนำกองทุนลดหย่อนภาษี (RMF / ThaiESG) ที่เหมาะกับอายุ รายได้ และความเสี่ยงของคุณ")

st.subheader("ขั้นที่ 1: กรอกข้อมูลของคุณ")
st.selectbox("เริ่มจากตัวอย่าง หรือกรอกเอง", list(PERSONAS), key="persona", on_change=apply_persona)
year = st.selectbox("ปีภาษีที่ใช้คำนวณ", sorted(tc.TAX_RULES), key="year", format_func=lambda y: f"ปีภาษี {y}")
rule = tc.TAX_RULES[year]
(st.info if rule["verified"] else st.warning)(("✅ " if rule["verified"] else "⚠️ ") + rule["note"])
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("อายุ (ปี)", min_value=20, max_value=100, step=1, key="age")
    income = st.number_input("เงินได้ต่อปี (บาท)", min_value=0, max_value=50_000_000, step=10_000, key="income")
    dependents = st.number_input("ผู้อยู่ในอุปการะ เช่น บุตร (คน, ลดหย่อนคนละ 30,000)", min_value=0, max_value=10, step=1, key="dependents")
with c2:
    insurance = st.number_input("เบี้ยประกันชีวิต/สุขภาพต่อปี (บาท)", min_value=0, max_value=1_000_000, step=1_000, key="insurance")
    pvd = st.number_input("เงินสะสมกองทุนสำรองเลี้ยงชีพต่อปี (บาท)", min_value=0, max_value=1_000_000, step=1_000, key="pvd")
    budget = st.number_input("งบที่ตั้งใจลงทุนในกองทุนลดหย่อนต่อปี (บาท)", min_value=0, max_value=10_000_000, step=5_000, key="budget")
spouse = st.checkbox("มีคู่สมรสที่ไม่มีเงินได้ (ลดหย่อนคู่สมรสได้)", key="spouse")
risk = st.select_slider("ความเสี่ยงที่คุณรับได้", options=[1, 2, 3, 4, 5], format_func=lambda r: tc.RISK_LABELS[r], key="risk")

lo_a, hi_a = tc.TRAIN_RANGE["age"]; lo_i, hi_i = tc.TRAIN_RANGE["income"]
if not (lo_a <= age <= hi_a) or not (lo_i <= income <= hi_i):
    st.warning(f"ค่าที่กรอกอยู่นอกช่วงข้อมูลที่ใช้ฝึกโมเดล (อายุ {lo_a}-{hi_a} ปี, เงินได้ {lo_i:,}-{hi_i:,} บาท) ผลการจัดกลุ่มอาจไม่น่าเชื่อถือ")

cluster = tc.assign_cluster(model, age, income, risk, insurance, pvd, dependents)
row = profile.loc[cluster]
result = tc.plan(age, income, dependents, insurance, pvd, budget, row["equity_share"], spouse=spouse, year=year)

st.subheader("ขั้นที่ 2: ผลการวิเคราะห์")
st.info(f"**โมเดลจัดคุณอยู่ในกลุ่ม:** {row['description']}  \n"
        f"(กลุ่มนี้มีอายุเฉลี่ย {row['age']:.0f} ปี ความเสี่ยงเฉลี่ย {row['risk_score']:.1f} จาก 5)")

m1, m2, m3 = st.columns(3)
m1.metric("ภาษีก่อนใช้กองทุน", baht(result["tax_before"]))
m2.metric("ภาษีหลังใช้กองทุน", baht(result["tax_after"]))
m3.metric("ภาษีที่ประหยัดได้", baht(result["tax_saved"]))

st.markdown("**วิธีคิดเงินได้สุทธิ (ก่อนซื้อกองทุนลดหย่อน)**")
ded = pd.DataFrame([("เงินได้ต่อปี", income)] + [(f"หัก: {n}", v) for n, v in result["deductions"] if v > 0]
                   + [("เงินได้สุทธิก่อนซื้อกองทุน", result["net_income"])], columns=["รายการ", "บาท"])
ded["บาท"] = ded["บาท"].map(lambda x: f"{x:,.0f}")
st.dataframe(ded, hide_index=True, use_container_width=True)
st.caption("เกณฑ์ที่ใช้: ผู้มีเงินเดือน (เงินได้ประเภท 40(1)) รองรับค่าลดหย่อนคู่สมรส ไม่รวมรายการพิเศษอื่น ๆ · สมมติฐานเพิ่มเติมของทีม (ไม่ใช่ข้อมูลที่ผู้ใช้กรอก): ประกันสังคม 9,000 บาทต่อปีทุกคน และผู้อยู่ในอุปการะ (รวมบุตร) นับคนละ 30,000 บาท")

with st.expander("ดูการคิดภาษีแบบขั้นบันได (0% - 35%) เทียบก่อน/หลังใช้กองทุน"):
    before, after = tc.tax_by_bracket(result["net_income"], year), tc.tax_by_bracket(max(result["net_income"] - result["invest"], 0), year)
    ladder = pd.DataFrame({"ช่วงเงินได้สุทธิ (บาท)": [b[0] for b in before], "อัตรา": [f"{b[1]:.0%}" for b in before],
                           "ภาษีก่อนใช้กองทุน": [f"{b[3]:,.0f}" for b in before], "ภาษีหลังใช้กองทุน": [f"{a[3]:,.0f}" for a in after]})
    ladder.loc[len(ladder)] = ["รวม", "", f"{result['tax_before']:,.0f}", f"{result['tax_after']:,.0f}"]
    st.dataframe(ladder, hide_index=True, use_container_width=True)

st.subheader("ขั้นที่ 3: คำแนะนำสำหรับคุณ")
if result["tax_before"] <= 0:
    st.success("รายได้สุทธิของคุณยังไม่ถึงเกณฑ์เสียภาษี **การซื้อกองทุนลดหย่อนจึงไม่ช่วยลดภาษี** ไม่จำเป็นต้องซื้อเพื่อลดหย่อน ถ้าจะลงทุนให้พิจารณาจากเป้าหมายการออมของคุณเอง")
elif result["invest"] <= 0:
    st.warning("ยังไม่ได้ตั้งงบลงทุน ลองกรอกงบที่ตั้งใจลงทุนด้านบน หรือดูตัวอย่างการลองเปลี่ยนค่าด้านล่าง")
else:
    per_baht = result["tax_saved"] / result["invest"]
    st.success(f"**แนะนำให้ลงทุน {baht(result['invest'])}** แบ่งเป็น RMF {baht(result['rmf'])} และ ThaiESG {baht(result['thaiesg'])} "
               f"จะลดภาษีได้ **{baht(result['tax_saved'])}** (ทุก 1 บาทที่ลงทุน ลดภาษีได้ประมาณ {per_baht * 100:.0f} สตางค์)")
    st.write(f"**ลักษณะกองทุนที่เหมาะกับกลุ่มของคุณ:** ตราสารทุนประมาณ {row['equity_share'] * 100:.0f}% "
             f"({baht(result['equity_amount'])}) และตราสารหนี้/ตลาดเงินประมาณ {(1 - row['equity_share']) * 100:.0f}% ({baht(result['bond_amount'])})")
    if result["limited_by_cap"]:
        st.warning(f"งบที่ตั้งไว้เกินวงเงินลดหย่อนที่ใช้ได้ ({baht(result['cap_total'])}) ระบบจึงแนะนำเพียงเท่าวงเงิน")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = ["RMF", "ThaiESG", "Equity", "Bond"]
    vals = [result["rmf"], result["thaiesg"], result["equity_amount"], result["bond_amount"]]
    bars = ax.bar(labels, vals, color=["#1f77b4", "#1f77b4", "#2ca02c", "#ff7f0e"])
    for b_, v_ in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Baht"); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig); plt.close(fig)
    st.caption("แท่ง RMF/ThaiESG แสดงช่องทางลดหย่อน ส่วนแท่งตราสารทุน/หนี้แสดงลักษณะสินทรัพย์ในกองทุน (นับรวมแยกกัน ไม่ได้บวกกัน)")

if result["max_invest"] > 0 and result["tax_before"] > 0:
    st.markdown("**แผนที่ประหยัดภาษีสูงสุดภายใต้เพดานสิทธิ**")
    st.success(f"ลงทุน **{baht(result['max_invest'])}** (RMF {baht(result['rmf_max'])} + ThaiESG {baht(result['esg_max'])}) "
               f"ภาษีจะลดจาก {baht(result['tax_before'])} เหลือ {baht(result['tax_after_max'])} ประหยัดสูงสุด **{baht(result['max_saved'])}**")
    st.caption("ยอดนี้คือเพดานสิทธิหรือเงินได้สุทธิ แล้วแต่อย่างใดต่ำกว่า ลงทุนเกินนี้ไม่ช่วยลดภาษีเพิ่ม และต้องดูว่าเงินก้อนนี้เหมาะกับสภาพคล่องของคุณ (ThaiESG ต้องถือ 5 ปีขึ้นไป)")
    if row["equity_share"] is not None:
        ftype = tc.fund_type_for(row["equity_share"])
        funds = tc.load_funds(ROOT)
        ex = funds[funds["fund_type"] == ftype]
        if len(ex):
            st.markdown(f"**ตัวอย่างกองทุน RMF และ ThaiESG ประเภท{ftype} จากแหล่งสาธารณะ** (กลุ่มของคุณเหมาะกับกองทุนแนวหุ้นประมาณ {row['equity_share'] * 100:.0f}%)")
            st.dataframe(ex[["fund_name", "scheme", "fund_type", "policy_summary"]].rename(columns={"fund_name": "ชื่อกองทุน", "scheme": "ประเภทสิทธิ", "fund_type": "ประเภทกองทุน", "policy_summary": "นโยบายโดยสรุป"}), hide_index=True, use_container_width=True)
            st.caption("ที่มา: เว็บไซต์/บทความสาธารณะของ InnovestX (ThaiESG: Café Invest, RMF: หน้ากองทุน RMF เก็บข้อมูล 22-23 พ.ย. 2569) เป็นเพียงตัวอย่างประกอบ ไม่ใช่คำแนะนำซื้อขาย ระบบไม่มีข้อมูลผลตอบแทนและไม่พยากรณ์ผลตอบแทนของกองทุน")

st.subheader("ลองเปลี่ยนค่า (What-if)")
if result["cap_total"] >= 1_000 and result["tax_before"] > 0:
    step = max(1_000, int(result["cap_total"] // 100 // 1_000 * 1_000)) or 1_000
    amounts = list(range(0, int(result["cap_total"]) + 1, step))
    what_if = st.slider("จำนวนเงินที่ลงทุนในกองทุนลดหย่อน (บาท)", 0, int(result["cap_total"]), min(int(result["invest"]), int(result["cap_total"])), step=1_000)
    w = tc.plan(age, income, dependents, insurance, pvd, budget, row["equity_share"], invest_override=what_if, spouse=spouse, year=year)
    st.write(f"ลงทุน {baht(w['invest'])} → ภาษีที่ประหยัดได้ **{baht(w['tax_saved'])}**")
    curve = [tc.plan(age, income, dependents, insurance, pvd, budget, row["equity_share"], invest_override=a, spouse=spouse, year=year)["tax_saved"] for a in amounts]
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    ax2.plot(amounts, curve, color="#1f77b4"); ax2.scatter([w["invest"]], [w["tax_saved"]], color="#d62728", zorder=3)
    ax2.set_xlabel("Fund investment (baht)"); ax2.set_ylabel("Tax saved (baht)"); ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2); plt.close(fig2)
elif result["tax_before"] <= 0:
    st.write("รายได้สุทธิของคุณไม่ถึงเกณฑ์เสียภาษี จึงไม่มีภาษีให้ลด การเปลี่ยนจำนวนเงินลงทุนจะไม่ทำให้ภาษีเปลี่ยน")
else:
    st.write("วงเงินลดหย่อนที่ใช้ได้น้อยเกินกว่าจะจำลอง")

with st.expander("โมเดลทำงานอย่างไร และข้อจำกัด (อ่านก่อนใช้)"):
    mt = info["metrics_test"]
    st.markdown(f"""
- **วิธีการ:** จัดกลุ่มผู้ใช้ด้วย K-Means ({info['n_clusters']} กลุ่ม) จากอายุ รายได้ ความเสี่ยง และภาระลดหย่อน แล้วเลือกสัดส่วนกองทุนตราสารทุน/หนี้ตามกลุ่ม
- **ภาษีที่ประหยัดได้** คำนวณจากสูตรภาษีตามกฎหมาย ไม่ได้มาจาก ML ส่วน ML ช่วยเลือกลักษณะกองทุนที่เหมาะกับกลุ่ม
- **ผลทดสอบ:** Silhouette {mt['silhouette_model']:.3f} (baseline อายุอย่างเดียว {mt['silhouette_age_only_baseline']:.3f}, สุ่ม {mt['silhouette_random_baseline']:.3f}) ·
  สัดส่วนหุ้นคลาดเคลื่อนจากความเสี่ยงของผู้ใช้ลดลงจาก {mt['equity_mae_one_size']:.3f} เป็น {mt['equity_mae_model']:.3f}
- **การเลือกจำนวนกลุ่ม:** ดู Elbow Method และ Silhouette Score ในตารางด้านล่าง ทีมเลือก k = {info['n_clusters']} เพราะแยกกลุ่มผู้สูงอายุออกมาได้ แม้ Silhouette ต่ำกว่า k = 2 ซึ่งสูงสุดเล็กน้อย (ผลจัดกลุ่มไม่ได้ตรงกับ 3 ช่วงวัยในโครงร่างทุกกลุ่ม)
- **ข้อจำกัด:** ฝึกด้วย **ข้อมูลจำลอง** ผลจึงยืนยันว่าระบบทำงานได้ แต่ยังไม่ได้พิสูจน์กับผู้ใช้จริง · กติกาภาษีเป็นแบบย่อสำหรับผู้มีเงินเดือน (ไม่รวมเงินได้ประเภทอื่น ฯลฯ) ·
  วงเงินและอัตราต้องตรวจกับประกาศกรมสรรพากรของปีภาษีนั้น
- **ไม่ควรเชื่อโมเดลเมื่อ:** ค่าที่กรอกอยู่นอกช่วงข้อมูลฝึก หรือมีรายการลดหย่อนพิเศษที่ระบบไม่รองรับ
- แหล่งโมเดล: {source}
""")
    kc = get_k_curve()
    e1, e2 = st.columns(2)
    e1.caption("Elbow Method: inertia ของแต่ละ k (ยิ่งต่ำยิ่งแน่น มองหาจุดที่ลดลงช้าลง)")
    f1, x1 = plt.subplots(figsize=(4, 2.8)); x1.plot(kc["k"], kc["inertia"], marker="o"); x1.set_xlabel("k"); x1.set_ylabel("Inertia")
    x1.axvline(info["n_clusters"], ls="--", color="gray"); e1.pyplot(f1); plt.close(f1)
    e2.caption("Silhouette Score ของแต่ละ k (ยิ่งสูงยิ่งแยกกลุ่มชัด)")
    f2, x2 = plt.subplots(figsize=(4, 2.8)); x2.plot(kc["k"], kc["silhouette"], marker="o", color="#2ca02c"); x2.set_xlabel("k"); x2.set_ylabel("Silhouette")
    x2.axvline(info["n_clusters"], ls="--", color="gray"); e2.pyplot(f2); plt.close(f2)
    st.dataframe(kc.round(3), hide_index=True, use_container_width=True)

st.caption("⚠️ เครื่องมือนี้เป็นสื่อการเรียนรู้ของโครงงานวิชา INI 373 ไม่ใช่คำแนะนำทางการเงินหรือภาษีอย่างเป็นทางการ ไม่เก็บข้อมูลส่วนบุคคลของผู้ใช้")
