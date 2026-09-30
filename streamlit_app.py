import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import io
from bs4 import BeautifulSoup
import re
import warnings

warnings.filterwarnings('ignore')

# 網頁基本設定
st.set_page_config(
    page_title="賽馬 AI 戰術預測系統 (Ranker 17維度)",
    page_icon="🐎",
    layout="wide" # 改為 wide 讓排版更寬敞
)

# --- 輔助函式區域 ---
def clean_person_name(val):
    return re.sub(r'\(.*?\)', '', str(val)).strip()

def extract_rating(val):
    nums = re.findall(r'\d+', str(val))
    return float(nums[0]) if nums else 52.0

def extract_horse_no(val):
    nums = re.findall(r'\d+', str(val))
    return int(nums[0]) if nums else 0

@st.cache_resource
def load_ai_model():
    """ 載入 17 維度 Ranker 排序 AI 模型並加上快取 """
    model = xgb.XGBRanker()
    model.load_model("horse_racing_ai_model.json")
    return model

@st.cache_data
def load_memory_databases():
    """ 載入各種歷史記憶庫 (包含 17 維度新增的同程同地勝率) """
    try:
        history_db = pd.read_csv("advanced_features.csv")
        horse_mem = history_db.drop_duplicates(subset=['馬匹編號'], keep='last').set_index('馬匹編號')
    except:
        horse_mem = pd.DataFrame()

    try:
        jockey_db = pd.read_csv("jockey_stats.csv").set_index('騎師')
        trainer_db = pd.read_csv("trainer_stats.csv").set_index('練馬師')
        hj_dict = pd.read_csv("horse_jockey_stats.csv").set_index(['馬匹編號', '騎師'])['勝率'].to_dict()
    except:
        jockey_db, trainer_db, hj_dict = pd.DataFrame(columns=['勝率']), pd.DataFrame(columns=['勝率']), {}
        
    try:
        hv_dict = pd.read_csv("horse_venue_stats.csv").set_index(['馬匹編號', '場地'])['勝率'].to_dict()
    except: hv_dict = {}
        
    try:
        hd_dict = pd.read_csv("horse_distance_stats.csv").set_index(['馬匹編號', '途程_數值'])['勝率'].to_dict()
    except: hd_dict = {}
        
    return horse_mem, jockey_db, trainer_db, hj_dict, hv_dict, hd_dict

# --- 核心：排位表抓取與 17 維度特徵運算 ---
def fetch_race_cards(date_str, venue_str, ignore_jockey=False):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36'}
    
    # 抓取馬匹 ID
    entries_url = f"https://racing.hkjc.com/zh-hk/local/information/entries?racedate={date_str}"
    name_to_id = {}
    try:
        res_entries = requests.get(entries_url, headers=headers, timeout=5)
        soup = BeautifulSoup(res_entries.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if "horse?horseid=" in link['href']:
                h_name = link.text.strip()
                if h_name: name_to_id[h_name] = link['href'].split("=")[-1]
    except: pass

    model = load_ai_model()
    horse_memory, jockey_db, trainer_db, hj_dict, hv_dict, hd_dict = load_memory_databases()
    
    predictions_list = []
    
    for race_no in range(1, 12):
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?RaceDate={date_str}&Racecourse={venue_str}&RaceNo={race_no}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if "找不到" in res.text or "沒有賽事" in res.text: continue
            
            # 抓取當前途程
            current_dist = 1200
            dist_match = re.search(r'(\d{4})\s*米', res.text)
            if dist_match: current_dist = int(dist_match.group(1))

            tables = pd.read_html(io.StringIO(res.text))
            target_tbl = next((tbl for tbl in tables if ('馬名' in str(tbl.columns) or '馬匹' in str(tbl.columns)) and '檔位' in str(tbl.columns)), None)
            
            if target_tbl is not None:
                df_race = target_tbl.copy()
                if isinstance(df_race.columns, pd.MultiIndex): df_race.columns = df_race.columns.get_level_values(-1)
                df_race.columns = [str(col).replace(' ', '').replace('\n', '') for col in df_race.columns]
                
                if '馬名' in df_race.columns: df_race.rename(columns={'馬名': '馬匹'}, inplace=True)
                if '配磅' in df_race.columns: df_race.rename(columns={'配磅': '負磅'}, inplace=True)
                
                horse_no_col = next((c for c in df_race.columns if '馬號' in str(c) or 'NO' in str(c).upper()), df_race.columns[0])
                df_race['馬號'] = df_race[horse_no_col].apply(extract_horse_no)
                df_race['檔位'] = pd.to_numeric(df_race['檔位'], errors='coerce')
                df_race['負磅'] = pd.to_numeric(df_race['負磅'], errors='coerce')
                rating_col = [c for c in df_race.columns if '評分' in c]
                df_race['評分'] = df_race[rating_col[0]].apply(extract_rating) if rating_col else 52.0
                df_race['騎師_清理'] = df_race['騎師'].apply(clean_person_name) if '騎師' in df_race.columns else '未知'
                df_race['練馬師_清理'] = df_race['練馬師'].apply(clean_person_name) if '練馬師' in df_race.columns else '未知'
                
                df_race = df_race.dropna(subset=['檔位', '負磅'])
                df_race = df_race[df_race['馬號'] > 0]
                
                race_avg_rating = df_race['評分'].mean()
                race_features = []
                horse_info = []
                
                # 計算 17 維度
                for _, row in df_race.iterrows():
                    h_id = name_to_id.get(row['馬匹'], "")
                    rating = row['評分']
                    jockey = row.get('騎師_清理', '未知')
                    trainer = row.get('練馬師_清理', '未知')
                    
                    j_w = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                    t_w = trainer_db.loc[trainer, '勝率'] if trainer in trainer_db.index else 0.08
                    hj_w = hj_dict.get((h_id, jockey), 0.08)
                    
                    if ignore_jockey: j_w, t_w, hj_w = 0.08, 0.08, 0.08
                    
                    rating_db, r_rank, h_win, l_rating, p_rank, a_last, l_margin, r_style = 52.0, 7.0, 0.08, 52.0, 7.0, 7.0, 5.0, 7.0
                    if h_id in horse_memory.index:
                        mem = horse_memory.loc[h_id]
                        rating_db, r_rank, h_win = mem.get('評分_數值', 52.0), mem.get('近三仗平均名次', 7.0), mem.get('歷史勝率', 0.08)
                        l_rating, p_rank, a_last = mem.get('評分_數值', rating), mem.get('上仗名次', 7.0), mem.get('名次_數值', 7.0)
                        l_margin, r_style = mem.get('頭馬距離_數值', 5.0), mem.get('近三仗早段走位', 7.0)
                        
                    features = {
                        '檔位_數值': row['檔位'], '負磅_數值': row['負磅'], '休賽天數': 30.0, '體重變化': 0.0,
                        '近三仗平均名次': r_rank, '歷史勝率': h_win, '評分_數值': rating, '騎師勝率': float(j_w), '練馬師勝率': float(t_w), 
                        '人馬合作勝率': float(hj_w), '升降班幅度': rating - l_rating, '相對場次優勢': rating - race_avg_rating,
                        '近況動能': float(p_rank - a_last), '同地勝率': float(hv_dict.get((h_id, venue_str), 0.08)), 
                        '同程勝率': float(hd_dict.get((h_id, current_dist), 0.08)), '上仗頭馬距離': float(l_margin), '近況跑法': float(r_style)
                    }
                    race_features.append(features)
                    horse_info.append({'場次': f"第 {race_no} 場", '馬號': int(row['馬號']), '馬匹': row['馬匹'], '騎師': jockey, '檔位': int(row['檔位']), '負磅': row['負磅'], '評分': rating})
                
                if race_features:
                    raw_scores = model.predict(pd.DataFrame(race_features))
                    win_probs = np.exp((raw_scores - np.max(raw_scores)) / 0.5) / np.sum(np.exp((raw_scores - np.max(raw_scores)) / 0.5))
                    for i, info in enumerate(horse_info):
                        info['勝率'] = win_probs[i]
                        info['AI預測勝率(%)'] = round(win_probs[i] * 100, 2)
                        predictions_list.append(info)
        except: pass
        
    return pd.DataFrame(predictions_list) if predictions_list else pd.DataFrame()
# --- 側邊欄：系統維護與資料庫管理 ---
st.sidebar.header("🛠️ 第一步：資料庫與 AI 訓練")

if st.sidebar.button("1. 抓取最新現役馬匹資料", use_container_width=True):
    with st.spinner("正在執行 main.py 抓取資料中..."):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        st.sidebar.code(result.stdout[-500:], language='text')
        st.sidebar.success("抓取完成！")

if st.sidebar.button("2. 更新 AI (基礎訓練)", use_container_width=True):
    with st.spinner("正在執行 update_ai.py..."):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "update_ai.py"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        st.sidebar.code(result.stdout[-500:], language='text')
        st.sidebar.success("AI 更新完成！")

if st.sidebar.button("3. 17維度 AI 自動尋優", use_container_width=True):
    with st.spinner("正在執行 auto_tune_ai.py 進行深度調參與特徵權重優化..."):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "auto_tune_ai.py"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        st.sidebar.code(result.stdout, language='text')
        st.sidebar.success("AI 自動尋優與模型儲存完成！")

st.sidebar.markdown("---")
st.sidebar.header("🏆 系統級總回測")
if st.sidebar.button("啟動總歷史回測 (30天)", use_container_width=True):
    with st.spinner("正在自動尋找賽馬日並進行 17 維度歷史回測..."):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "auto_backtest.py"], capture_output=True, text=True, encoding='utf-8', errors='replace')
        st.sidebar.code(result.stdout, language='text')


# --- 主畫面：UI 介面設計 ---
st.title("🐎 賽馬 AI 戰術預測系統 (Ranker 17維度)")
st.markdown("基於 XGBRanker 機器學習，結合 **17 維度終極特徵**（含同程勝率、頭馬距離、近況跑法）進行同場精準排序。")

st.divider()

st.subheader("🗓️ 設定目標賽事")
col1, col2 = st.columns(2)

with col1:
    target_date = st.text_input("賽事日期 (格式: YYYY/MM/DD)", value="2026/10/01")
with col2:
    venue = st.selectbox("賽事場地", options=["ST (沙田)", "HV (跑馬地)"])
    venue_code = venue.split(" ")[0]

ignore_jockey = st.checkbox("🚫 忽視騎師權重 (純馬匹實力模式)", value=False)
if ignore_jockey:
    st.warning("已啟動「忽視騎師權重」模式：騎師與練馬師的勝率影響力已歸零。")

# --- 按鈕佈局 ---
col_btn1, col_btn2, col_btn3 = st.columns(3)
action = None

with col_btn1:
    if st.button("🚀 實戰預測", use_container_width=True):
        action = "predict"
with col_btn2:
    if st.button("📊 單日回測", use_container_width=True):
        action = "backtest"
with col_btn3:
    if st.button("🎯 策略推薦", use_container_width=True):
        action = "recommend"

# --- 功能 1：實戰預測 ---
if action == "predict":
    with st.spinner("AI 正在連線賽馬會，進行 17 維度運算..."):
        df_pred = fetch_race_cards(target_date, venue_code, ignore_jockey)
        if df_pred.empty:
            st.error(f"無法抓取 {target_date} 的排位表。可能是日期錯誤或賽事尚未公佈。")
        else:
            st.success("預測運算完成！")
            for name, group in sorted(df_pred.groupby('場次'), key=lambda x: int(x[0].replace('第 ', '').replace(' 場', ''))):
                st.markdown(f"### 🏆 [ {name} ] AI 排序推薦前四名")
                display_df = group.sort_values(by='勝率', ascending=False).head(4)
                st.dataframe(display_df[['馬號', '馬匹', '騎師', '檔位', '負磅', '評分', 'AI預測勝率(%)']], hide_index=True, use_container_width=True)

# --- 功能 2：單日歷史回測 ---
elif action == "backtest":
    st.subheader(f"📊 單日歷史回測報告 (17維度) - {target_date} ({venue_code})")
    with st.spinner("正在執行回測分析..."):
        import subprocess
        import sys
        cmd = [sys.executable, "backtest_ai_2.py", target_date, venue_code]
        if ignore_jockey: cmd.append("--ignore-jockey")
        process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        st.code(process.stdout, language='text')

# --- 功能 3：策略推薦 (基礎彩池 + 最佳策略) ---
elif action == "recommend":
    with st.spinner("AI 正在連線賽馬會，計算全彩池與最佳策略..."):
        df_pred = fetch_race_cards(target_date, venue_code, ignore_jockey)
        if df_pred.empty:
            st.error(f"無法抓取 {target_date} 的排位表。")
        else:
            st.success("策略計算完成！")
            for name, group in sorted(df_pred.groupby('場次'), key=lambda x: int(x[0].replace('第 ', '').replace(' 場', ''))):
                group = group.sort_values(by='勝率', ascending=False).reset_index(drop=True)
                if len(group) >= 4:
                    h1, h2, h3, h4 = group.iloc[0], group.iloc[1], group.iloc[2], group.iloc[3]
                    top_win_prob = h1['勝率']
                    
                    st.markdown(f"### 🎯 [ {name} ] 📊 AI 勝率分佈與全彩池推薦")
                    st.markdown(f"**🥇 核心首選**: ({h1['馬號']}) {h1['馬匹']} [勝率: {h1['AI預測勝率(%)']}%]")
                    st.markdown(f"**🥈 實力配腳**: ({h2['馬號']}) {h2['馬匹']}, ({h3['馬號']}) {h3['馬匹']}, ({h4['馬號']}) {h4['馬匹']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### 📋 基礎全彩池組合")
                        st.markdown(f"- **獨贏/位置 (W/P)**: 投注 ({h1['馬號']})")
                        st.markdown(f"- **連贏/位置Q**: 以 ({h1['馬號']}) 為膽，拖 ({h2['馬號']})、({h3['馬號']})、({h4['馬號']})")
                        st.markdown(f"- **三重彩/單T**: 複式包注 ➔ ({h1['馬號']}), ({h2['馬號']}), ({h3['馬號']}), ({h4['馬號']})")
                        st.markdown(f"- **四連環/四重彩**: 複式互聯 ➔ ({h1['馬號']}), ({h2['馬號']}), ({h3['馬號']}), ({h4['馬號']})")
                    
                    with col_b:
                        st.markdown("#### ⭐ AI 最佳策略推薦")
                        if top_win_prob > 0.25:
                            st.success(f"**👑 判斷: 超級馬膽** (首選實力超群，具絕對統治力)\n\n**💰 推薦**: 強攻【獨贏】與【連贏/位置Q膽拖】")
                        elif top_win_prob >= 0.15:
                            st.info(f"**⚖️ 判斷: 勢均力敵** (雙雄鼎立或前列馬匹實力接近)\n\n**💰 推薦**: 主攻【連贏/位置Q互串】與【單T複式包注】")
                        else:
                            st.warning(f"**🎲 判斷: 混戰格局** (群龍無首，極易爆出大冷門)\n\n**💰 推薦**: 略過單邊獨贏，專攻大彩池【四連環複式互聯】")
                    st.divider()