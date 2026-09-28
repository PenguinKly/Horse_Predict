import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import io
from bs4 import BeautifulSoup
import re
import warnings

warnings.filterwarnings('ignore')

# 網頁基本設定 (設定標題、圖示、以及適應手機螢幕)
st.set_page_config(
    page_title="賽馬 AI 戰術預測系統",
    page_icon="🐎",
    layout="centered"
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
    """ 載入 AI 模型並加上快取，避免每次點擊按鈕都要重載 """
    model = xgb.XGBClassifier()
    model.load_model("horse_racing_ai_model.json")
    return model

@st.cache_data
def load_memory_databases():
    """ 載入各種歷史記憶庫並加上快取 """
    try:
        history_db = pd.read_csv("advanced_features.csv")
        horse_mem = history_db.drop_duplicates(subset=['馬匹編號'], keep='last').set_index('馬匹編號')
    except:
        horse_mem = pd.DataFrame()

    try:
        jockey_db = pd.read_csv("jockey_stats.csv").set_index('騎師')
        trainer_db = pd.read_csv("trainer_stats.csv").set_index('練馬師')
        hj_db = pd.read_csv("horse_jockey_stats.csv")
        hj_dict = hj_db.set_index(['馬匹編號', '騎師'])['勝率'].to_dict()
    except:
        jockey_db = pd.DataFrame(columns=['勝率'])
        trainer_db = pd.DataFrame(columns=['勝率'])
        hj_dict = {}
        
    return horse_mem, jockey_db, trainer_db, hj_dict


# --- UI 介面設計 ---

st.title("🐎 賽馬 AI 戰術預測系統")
st.markdown("基於 XGBoost 機器學習，整合 12 維度特徵，包含人馬默契與步速形勢。")

st.divider()

st.subheader("🗓️ 設定目標賽事")
col1, col2 = st.columns(2)

with col1:
    # 預設為今天的日期格式 YYYY/MM/DD
    target_date = st.text_input("賽事日期 (格式: YYYY/MM/DD)", value="2026/10/01")
with col2:
    venue = st.selectbox("賽事場地", options=["ST (沙田)", "HV (跑馬地)"])
    venue_code = venue.split(" ")[0]

# --- 核心預測邏輯 ---
if st.button("🚀 開始預測", type="primary", use_container_width=True):
    with st.spinner("AI 正在連線賽馬會，讀取排位表與官方即時評分..."):
        model = load_ai_model()
        horse_memory, jockey_db, trainer_db, hj_dict = load_memory_databases()
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        entries_url = f"https://racing.hkjc.com/zh-hk/local/information/entries?racedate={target_date}"
        res_entries = requests.get(entries_url, headers=headers)
        soup = BeautifulSoup(res_entries.text, 'html.parser')

        name_to_id = {}
        for link in soup.find_all('a', href=True):
            if "horse?horseid=" in link['href']:
                h_id = link['href'].split("=")[-1]
                h_name = link.text.strip()
                if h_name:
                    name_to_id[h_name] = h_id

        today_races = []
        
        # 爬取 1 到 11 場
        for race_no in range(1, 12):
            url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?RaceDate={target_date}&Racecourse={venue_code}&RaceNo={race_no}"
            try:
                res = requests.get(url, headers=headers)
                if "找不到" in res.text or "沒有賽事" in res.text:
                    break
                    
                tables = pd.read_html(io.StringIO(res.text))
                target_tbl = None
                for tbl in tables:
                    columns_str = str(tbl.columns)
                    if ('馬名' in columns_str or '馬匹' in columns_str) and '檔位' in columns_str:
                        target_tbl = tbl
                        break
                        
                if target_tbl is not None:
                    df_race = target_tbl.copy()
                    if isinstance(df_race.columns, pd.MultiIndex):
                        df_race.columns = df_race.columns.get_level_values(-1)
                    
                    df_race.columns = [str(col).replace(' ', '').replace('\n', '') for col in df_race.columns]
                    df_race['場次'] = f"第 {race_no} 場"
                    
                    if '馬名' in df_race.columns: df_race.rename(columns={'馬名': '馬匹'}, inplace=True)
                    if '配磅' in df_race.columns: df_race.rename(columns={'配磅': '負磅'}, inplace=True)
                        
                    horse_no_col = next((c for c in df_race.columns if '馬號' in str(c) or 'NO' in str(c).upper()), df_race.columns[0])
                    df_race['馬號'] = df_race[horse_no_col].apply(extract_horse_no)
                    df_race['檔位'] = pd.to_numeric(df_race['檔位'], errors='coerce')
                    df_race['負磅'] = pd.to_numeric(df_race['負磅'], errors='coerce')
                    
                    rating_col = [c for c in df_race.columns if '評分' in c]
                    df_race['評分'] = df_race[rating_col[0]].apply(extract_rating) if rating_col else 52.0
                    
                    if '騎師' in df_race.columns:
                        df_race['騎師_清理'] = df_race['騎師'].apply(clean_person_name)
                    else:
                        df_race['騎師_清理'] = '未知'
                        
                    if '練馬師' in df_race.columns:
                        df_race['練馬師_清理'] = df_race['練馬師'].apply(clean_person_name)
                    else:
                        df_race['練馬師_清理'] = '未知'
                        
                    df_race = df_race.dropna(subset=['檔位', '負磅'])
                    df_race = df_race[df_race['馬號'] > 0]
                    
                    today_races.append(df_race)
            except Exception:
                pass

        if not today_races:
            st.error(f"無法抓取 {target_date} 的排位表。可能是輸入錯誤，或是賽事尚未公佈。")
        else:
            st.success(f"成功連線！正在為 {len(today_races)} 場賽事進行 AI 勝率運算...")
            
            predictions_list = []
            
            # 開始預測
            for race_df in today_races:
                race_name = race_df['場次'].iloc[0]
                
                front_runners_count = 0
                for _, row in race_df.iterrows():
                    h_id = name_to_id.get(row['馬匹'], "")
                    style = int(horse_memory.loc[h_id].get('跑法傾向', 2)) if h_id in horse_memory.index else 2
                    if style == 1: front_runners_count += 1
                        
                pace_pressure = float(front_runners_count)
                
                for _, row in race_df.iterrows():
                    horse_name = row['馬匹']
                    horse_id = name_to_id.get(horse_name, "")
                    rating = row['評分']
                    horse_no = int(row['馬號'])
                    
                    jockey = row.get('騎師_清理', '未知')
                    trainer = row.get('練馬師_清理', '未知')
                    
                    j_win_rate = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                    t_win_rate = trainer_db.loc[trainer, '勝率'] if trainer in trainer_db.index else 0.08
                    hj_win_rate = hj_dict.get((horse_id, jockey), 0.08)
                    
                    if rating >= 100 and horse_id not in horse_memory.index:
                        recent_rank, hist_win_rate, rest_days, weight_change, running_style = 3.0, 0.25, 60.0, 0.0, 2
                    else:
                        recent_rank, hist_win_rate, rest_days, weight_change, running_style = 7.0, 0.08, 30.0, 0.0, 2
                    
                    if horse_id in horse_memory.index:
                        mem = horse_memory.loc[horse_id]
                        recent_rank = mem.get('近三仗平均名次', 7.0)
                        hist_win_rate = mem.get('歷史勝率', 0.08)
                        rest_days = mem.get('休賽天數', 30.0)
                        weight_change = mem.get('體重變化', 0.0)
                        running_style = int(mem.get('跑法傾向', 2))
                    
                    horse_features = {
                        '檔位_數值': row['檔位'],
                        '負磅_數值': row['負磅'],
                        '休賽天數': rest_days,
                        '體重變化': weight_change,
                        '近三仗平均名次': recent_rank,
                        '歷史勝率': hist_win_rate,
                        '評分_數值': rating,
                        '跑法傾向': running_style,
                        '同場步速壓力': pace_pressure,
                        '騎師勝率': float(j_win_rate),
                        '練馬師勝率': float(t_win_rate),
                        '人馬合作勝率': float(hj_win_rate)
                    }
                    
                    features_df = pd.DataFrame([horse_features])
                    win_prob = model.predict_proba(features_df)[:, 1][0]
                    
                    predictions_list.append({
                        '場次': race_name,
                        '馬號': horse_no,
                        '馬匹': horse_name,
                        '騎師': jockey,
                        '檔位': int(row['檔位']),
                        '負磅': row['負磅'],
                        '評分': int(rating),
                        'AI預測勝率(%)': round(win_prob * 100, 2)
                    })

            final_predictions = pd.DataFrame(predictions_list)
            
            # 整理並顯示在前端畫面
            grouped = final_predictions.groupby('場次')
            sorted_groups = sorted(grouped, key=lambda x: int(x[0].replace('第 ', '').replace(' 場', '')))
            
            st.divider()
            st.subheader(f"🏆 {target_date} {venue_code} 預測結果")
            
            # 使用 Streamlit 內建的資料表格式顯示
            for name, group in sorted_groups:
                st.markdown(f"**[ {name} ] AI 戰術推薦前四名**")
                
                # 排序並取前 4 名
                sorted_group = group.sort_values(by='AI預測勝率(%)', ascending=False).head(4)
                
                # 為了顯示美觀，把場次欄位藏起來，並轉換勝率為字串加上 %
                display_df = sorted_group.drop(columns=['場次'])
                display_df['AI預測勝率(%)'] = display_df['AI預測勝率(%)'].apply(lambda x: f"{x:.2f}%")
                
                # 將檔位、負磅、評分轉為無小數點的整數或漂亮格式
                display_df['檔位'] = display_df['檔位'].astype(int)
                display_df['負磅'] = display_df['負磅'].astype(int)
                display_df['評分'] = display_df['評分'].astype(int)
                
                # 隱藏左側的預設 index，並全寬度顯示
                st.dataframe(
                    display_df, 
                    hide_index=True, 
                    use_container_width=True
                )