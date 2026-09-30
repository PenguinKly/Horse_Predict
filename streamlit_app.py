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

st.set_page_config(
    page_title="賽馬 AI 戰術預測系統 (17維度矩陣版)",
    page_icon="🐎",
    layout="wide"
)

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
    model = xgb.XGBRanker()
    try: model.load_model("horse_racing_ai_model.json")
    except: pass
    return model

@st.cache_data
def load_memory_databases():
    try:
        history_db = pd.read_csv("advanced_features.csv")
        horse_mem = history_db.drop_duplicates(subset=['馬匹編號'], keep='last').set_index('馬匹編號')
    except: horse_mem = pd.DataFrame()
    try: jockey_db = pd.read_csv("jockey_stats.csv").set_index('騎師')
    except: jockey_db = pd.DataFrame(columns=['勝率'])
    try: trainer_db = pd.read_csv("trainer_stats.csv").set_index('練馬師')
    except: trainer_db = pd.DataFrame(columns=['勝率'])
    try: hj_dict = pd.read_csv("horse_jockey_stats.csv").set_index(['馬匹編號', '騎師'])['勝率'].to_dict()
    except: hj_dict = {}
    try: hv_dict = pd.read_csv("horse_venue_stats.csv").set_index(['馬匹編號', '場地'])['勝率'].to_dict()
    except: hv_dict = {}
    try: hd_dict = pd.read_csv("horse_distance_stats.csv").set_index(['馬匹編號', '途程_數值'])['勝率'].to_dict()
    except: hd_dict = {}
    return horse_mem, jockey_db, trainer_db, hj_dict, hv_dict, hd_dict

def fetch_race_cards(date_str, venue_str, ignore_jockey=False):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/119.0.0.0 Safari/537.36'}
    session = requests.Session()
    
    entries_url = f"https://racing.hkjc.com/zh-hk/local/information/entries?racedate={date_str}"
    name_to_id = {}
    try:
        res_entries = session.get(entries_url, headers=headers, timeout=5)
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
            res = session.get(url, headers=headers, timeout=5)
            if "找不到" in res.text or "沒有賽事" in res.text: continue
            
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
                race_features, horse_info = [], []
                
                for _, row in df_race.iterrows():
                    h_id = name_to_id.get(row['馬匹'], "")
                    rating, jockey, trainer = row['評分'], row.get('騎師_清理', '未知'), row.get('練馬師_清理', '未知')
                    
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
                    
                    ability_score = (rating + h_win * 100) / 2
                    jt_score = (j_w + t_w) * 500
                    venue_dist_score = (hv_dict.get((h_id, venue_str), 0.08) + hd_dict.get((h_id, current_dist), 0.08)) * 500
                    
                    horse_info.append({
                        '場次': f"第 {race_no} 場", '馬號': int(row['馬號']), '馬匹': row['馬匹'], '騎師': jockey, 
                        '檔位': int(row['檔位']), '負磅': row['負磅'], '評分': rating,
                        '實力分': ability_score, '騎練分': jt_score, '同場往績': venue_dist_score
                    })
                
                if race_features:
                    raw_scores = model.predict(pd.DataFrame(race_features))
                    win_probs = np.exp((raw_scores - np.max(raw_scores)) / 0.5) / np.sum(np.exp((raw_scores - np.max(raw_scores)) / 0.5))
                    for i, info in enumerate(horse_info):
                        info['勝率'] = win_probs[i]
                        info['IH指數'] = round(win_probs[i] * 100, 1)
                        info['AI預測勝率(%)'] = round(win_probs[i] * 100, 2)
                        predictions_list.append(info)
        except: pass
    return pd.DataFrame(predictions_list) if predictions_list else pd.DataFrame()

def get_level(val, low_th, high_th):
    if val >= high_th: return "高"
    elif val <= low_th: return "低"
    else: return "中"

def get_stars(prob):
    if prob >= 0.20: return "⭐⭐⭐⭐"
    elif prob >= 0.12: return "⭐⭐⭐"
    elif prob >= 0.06: return "⭐⭐"
    else: return "⭐"

# --- UI 介面設計 ---
st.title("🐎 賽馬 AI 戰術預測系統 (17維度矩陣版)")
st.markdown("基於 XGBRanker 機器學習，結合 **17 維度終極特徵**，並提供專業級的**單場多維度能力矩陣對比**。")
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

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
action = None

with col_btn1:
    if st.button("🚀 實戰預測", use_container_width=True): action = "predict"
with col_btn2:
    if st.button("📊 單日回測", use_container_width=True): action = "backtest"
with col_btn3:
    if st.button("🎯 策略推薦", use_container_width=True): action = "recommend"
with col_btn4:
    if st.button("📋 單場能力矩陣", use_container_width=True): action = "matrix"

st.divider()

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

elif action == "backtest":
    st.subheader(f"📊 單日歷史回測報告 (17維度) - {target_date} ({venue_code})")
    with st.spinner("正在讀取歷史賽果並執行高速回測分析..."):
        model = load_ai_model()
        horse_memory, jockey_db, trainer_db, hj_dict, hv_dict, hd_dict = load_memory_databases()
        session = requests.Session()
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        total_races, ai_top1_hit, ai_top3_catch_win, exact_match_count = 0, 0, 0, 0
        ai_top4_catch_counts = []
        
        for race_no in range(1, 12):
            url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={target_date}&Racecourse={venue_code}&RaceNo={race_no}"
            try:
                res = session.get(url, headers=headers, timeout=5)
                if "找不到" in res.text or "沒有賽事" in res.text: continue
                
                current_dist = 1200
                dist_match = re.search(r'(\d{4})\s*米', res.text)
                if dist_match: current_dist = int(dist_match.group(1))

                soup = BeautifulSoup(res.text, 'html.parser')
                name_to_id = {re.sub(r'\(.*?\)', '', link.text).strip(): link['href'].split('=')[-1].split('&')[0].strip() for link in soup.find_all('a', href=True) if 'horse' in link['href'].lower() and link.text.strip()}
                
                tables = pd.read_html(io.StringIO(res.text))
                target_tbl = next((tbl for tbl in tables if '名次' in str(tbl.columns) and '馬名' in str(tbl.columns)), None)
                
                if target_tbl is not None:
                    df_race = target_tbl.copy()
                    if isinstance(df_race.columns, pd.MultiIndex): df_race.columns = df_race.columns.get_level_values(-1)
                    df_race = df_race[pd.to_numeric(df_race['名次'], errors='coerce').notnull()] 
                    
                    race_features, horse_info = [], []
                    temp_ratings = [horse_memory.loc[name_to_id.get(re.sub(r'\(.*?\)', '', row['馬名']).strip(), "")].get('評分_數值', 52.0) if name_to_id.get(re.sub(r'\(.*?\)', '', row['馬名']).strip(), "") in horse_memory.index else 52.0 for _, row in df_race.iterrows()]
                    race_avg_rating = sum(temp_ratings) / len(temp_ratings) if temp_ratings else 52.0
                    
                    for _, row in df_race.iterrows():
                        h_name = re.sub(r'\(.*?\)', '', row['馬名']).strip()
                        h_id = name_to_id.get(h_name, "")
                        actual_rank = int(row['名次'])
                        jockey = re.sub(r'\(.*?\)', '', str(row.get('騎師', '未知'))).strip()
                        trainer = re.sub(r'\(.*?\)', '', str(row.get('練馬師', '未知'))).strip()
                        
                        j_w = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                        t_w = trainer_db.loc[trainer, '勝率'] if trainer in trainer_db.index else 0.08
                        hj_w = hj_dict.get((h_id, jockey), 0.08)
                        if ignore_jockey: j_w, t_w, hj_w = 0.08, 0.08, 0.08
                        
                        draw = pd.to_numeric(row.get('檔位', 7), errors='coerce')
                        weight = pd.to_numeric(row.get('實際負磅', 125), errors='coerce')
                        
                        rating, r_rank, h_win, l_rating, p_rank, a_last, l_margin, r_style = 52.0, 7.0, 0.08, 52.0, 7.0, 7.0, 5.0, 7.0
                        if h_id in horse_memory.index:
                            mem = horse_memory.loc[h_id]
                            rating, r_rank, h_win = mem.get('評分_數值', 52.0), mem.get('近三仗平均名次', 7.0), mem.get('歷史勝率', 0.08)
                            l_rating, p_rank, a_last = mem.get('評分_數值', rating), mem.get('上仗名次', 7.0), mem.get('名次_數值', 7.0)
                            l_margin, r_style = mem.get('頭馬距離_數值', 5.0), mem.get('近三仗早段走位', 7.0)
                        
                        features = {
                            '檔位_數值': float(draw) if pd.notna(draw) else 7.0, '負磅_數值': float(weight) if pd.notna(weight) else 125.0,
                            '休賽天數': 30.0, '體重變化': 0.0, '近三仗平均名次': r_rank, '歷史勝率': h_win, '評分_數值': float(rating),
                            '騎師勝率': float(j_w), '練馬師勝率': float(t_w), '人馬合作勝率': float(hj_w),
                            '升降班幅度': rating - l_rating, '相對場次優勢': rating - race_avg_rating, '近況動能': float(p_rank - a_last),
                            '同地勝率': float(hv_dict.get((h_id, venue_code), 0.08)), '同程勝率': float(hd_dict.get((h_id, current_dist), 0.08)),
                            '上仗頭馬距離': float(l_margin), '近況跑法': float(r_style)
                        }
                        race_features.append(features)
                        horse_info.append({'馬匹': h_name, '真實名次': actual_rank})

                    if race_features:
                        raw_scores = model.predict(pd.DataFrame(race_features))
                        win_probs = np.exp((raw_scores - np.max(raw_scores)) / 0.5) / np.sum(np.exp((raw_scores - np.max(raw_scores)) / 0.5))
                        for i, info in enumerate(horse_info): info['AI預測勝率'] = win_probs[i]

                    df_pred = pd.DataFrame(horse_info).sort_values(by='AI預測勝率', ascending=False)
                    ai_top4 = df_pred.head(4)['馬匹'].tolist()
                    ai_top3 = df_pred.head(3)['馬匹'].tolist()
                    actual_top4 = df_pred[df_pred['真實名次'] <= 4]['馬匹'].tolist()
                    actual_winner = df_pred[df_pred['真實名次'] == 1]['馬匹'].tolist()[0] if not df_pred[df_pred['真實名次']==1].empty else "無資料"
                    actual_rank_dict = dict(zip(df_pred['馬匹'], df_pred['真實名次']))
                    
                    st.markdown(f"### 🏁 [ 第 {race_no} 場 ]")
                    st.markdown(f"* **真實冠軍**: `{actual_winner}`")
                    st.markdown("* **AI 推薦前四名**:")
                    
                    for idx, h in enumerate(ai_top4):
                        r_rank = actual_rank_dict.get(h, 99)
                        ai_rank = idx + 1
                        if ai_rank == r_rank:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🟢 **#{ai_rank} {h}** (真實 #{r_rank})")
                            exact_match_count += 1
                        elif r_rank <= 4:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🟡 **#{ai_rank} {h}** (真實 #{r_rank})")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;⬛ **#{ai_rank} {h}** (真實 #{r_rank})")
                    
                    match_count = len(set(ai_top4) & set(actual_top4))
                    ai_top4_catch_counts.append(match_count)
                    st.markdown(f"* **前四名命中數**: `{match_count}/4 匹`")
                    st.divider()
                    
                    total_races += 1
                    if ai_top4 and actual_winner != "無資料":
                        if ai_top4[0] == actual_winner: ai_top1_hit += 1
                        if actual_winner in ai_top3: ai_top3_catch_win += 1
            except: pass

        if total_races > 0:
            st.subheader("📈 回測績效總結報告")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("單邊獨贏命中率", f"{ai_top1_hit}/{total_races}", f"{round((ai_top1_hit/total_races)*100,1)}%")
            m2.metric("頭馬涵蓋率 (前3名)", f"{ai_top3_catch_win}/{total_races}", f"{round((ai_top3_catch_win/total_races)*100,1)}%")
            m3.metric("前四名平均命中", f"{round(sum(ai_top4_catch_counts)/total_races, 2)} 匹")
            m4.metric("名次完全吻合", f"{exact_match_count} 匹")
        else:
            st.warning("找不到該日期的歷史賽果或尚未有完賽資料。")

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

elif action == "matrix":
    st.subheader(f"📋 單場多維度能力比較矩陣 - {target_date} ({venue_code})")
    with st.spinner("正在讀取排位並建構多維能力矩陣..."):
        df_pred = fetch_race_cards(target_date, venue_code, ignore_jockey)
        if df_pred.empty:
            st.error(f"無法取得 {target_date} 的排位資料。")
        else:
            races = sorted(df_pred['場次'].unique(), key=lambda x: int(x.replace('第 ', '').replace(' 場', '')))
            selected_race = st.selectbox("請選擇要檢視的場次", options=races)
            
            sub_df = df_pred[df_pred['場次'] == selected_race].sort_values(by='勝率', ascending=False).reset_index(drop=True)
            
            matrix_data = []
            for _, row in sub_df.iterrows():
                matrix_data.append({
                    '馬號': row['馬號'],
                    '馬名': row['馬匹'],
                    'IH指數': row['IH指數'],
                    '預估勝率': f"{row['AI預測勝率(%)']}%",
                    '評級': get_stars(row['勝率']),
                    '實力分': get_level(row['實力分'], 55, 75),
                    '騎練分': get_level(row['騎練分'], 40, 60),
                    '檔位分': "高" if row['檔位'] <= 5 else ("低" if row['檔位'] >= 11 else "中"),
                    '同場往績': get_level(row['同場往績'], 40, 60)
                })
            
            df_matrix = pd.DataFrame(matrix_data)
            st.markdown(f"### 📊 [ {selected_race} ] 馬匹多維度能力對比表")
            st.dataframe(df_matrix, hide_index=True, use_container_width=True)
            st.info("💡 **指標說明**：IH指數由 AI 綜合排序轉化；實力分結合評分與歷史勝率；騎練分結合騎師與練馬師勝率；檔位分依內外檔自動評定；同場往績結合同地與同程勝率。")
