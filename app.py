import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os

# ====== データ読み込み ======
@st.cache_data
def load_soraru_data():
    df = pd.read_csv("soraru_data.csv")
    return df

df = load_soraru_data()

# ====== 音声特徴量抽出 ======
def extract_user_features_from_file(uploaded_file, duration=15):
    # 一時ファイルに保存して librosa で読む
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, duration=duration)
    os.remove(tmp_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])  # 26次元

# ====== 距離 → スコア変換（絶対スケール） ======
def convert_to_score(dist, min_dist, max_dist):
    score = 1 - (dist - min_dist) / (max_dist - max_dist if max_dist == min_dist else (max_dist - min_dist))
    return max(0, min(score * 100, 100))  # 0〜100にクリップ

# ====== 総合そらる率 + 曲ランキング ======
def analyze_all(user_feat, df):
    # 全曲の特徴量（26次元）
    song_feats = df[[f"mfcc_{i}" for i in range(26)]].values

    # --- そらる曲同士の距離分布を作る（絶対スケールの基準） ---
    all_dists = []
    for i in range(len(song_feats)):
        for j in range(i + 1, len(song_feats)):
            all_dists.append(np.linalg.norm(song_feats[i] - song_feats[j]))

    min_dist = min(all_dists)
    max_dist = max(all_dists)

    # --- 総合そらる率 ---
    soraru_center = song_feats.mean(axis=0)
    dist_total = np.linalg.norm(user_feat - soraru_center)
    soraru_rate = convert_to_score(dist_total, min_dist, max_dist)

    # --- 曲ランキング ---
    results = []
    for (_, row), song_feat in zip(df.iterrows(), song_feats):
        dist = np.linalg.norm(user_feat - song_feat)
        score = convert_to_score(dist, min_dist, max_dist)
        results.append({
            "song": row["song"],
            "url": row["youtube_url"],
            "score": score
        })

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)

    return soraru_rate, df_res

# ====== コメント生成（そらる率に応じて） ======
def generate_comment(rate: float) -> str:
    if rate >= 90:
        return "転生したそらる様レベル。声質のニュアンスまでほぼ完全一致です。"
    elif rate >= 80:
        return "かなりの高シンクロ率。中音域の息成分や響き方がとても近いです。"
    elif rate >= 70:
        return "かなり似てます。歌い方や声の抜け感にそらる味があります。"
    elif rate >= 60:
        return "ところどころにそらる成分を感じます。意識して寄せたらまだ伸びそう。"
    elif rate >= 50:
        return "一部の帯域やニュアンスに共通点があります。自分の個性も強く出ているタイプ。"
    else:
        return "そらるとは違う方向性の声質ですが、唯一無二の個性があります。"

# ====== Streamlit UI ======
st.set_page_config(page_title="【精密解析】そらる・シンクロ率チェッカー", layout="centered")

# ① タイトル
st.title("【精密解析】 そらる・シンクロ率チェッカー")
st.caption("あなたの声に最も近い楽曲も判定！")

st.markdown("---")

# ② 音声アップロード
st.subheader("① 音声ファイルをアップロード")
st.write("**対応形式：** wav / mp3 / m4a")
st.write("**推奨：** 10〜20秒のサビや盛り上がり部分（声が大きいところ）")
st.write("※ 声だけ・アカペラだとより精度が上がります")

uploaded_file = st.file_uploader("ここに音声ファイルをドラッグ＆ドロップしてください", type=["wav", "mp3", "m4a"])

st.markdown("---")

# ③ 判定ボタン
st.subheader("② 精密解析スタート")

analyze_button = st.button("🔍 精密解析スタート")

if analyze_button:
    if uploaded_file is None:
        st.warning("先に音声ファイルをアップロードしてください。")
    else:
        with st.spinner("解析中です… 音声ファイルの長さによって少し時間がかかる場合があります。"):
            try:
                user_feat = extract_user_features_from_file(uploaded_file, duration=15)
                soraru_rate, result = analyze_all(user_feat, df)
            except Exception as e:
                st.error(f"解析中にエラーが発生しました：{e}")
                st.stop()

        st.success("解析が完了しました！")
        st.caption("※ あなたの音声ファイルは診断後に破棄されます。他者に利用されることはありません。")

        st.markdown("---")

        # ④ 結果表示
        st.subheader("③ 結果")

        st.markdown(f"### あなたのそらる・シンクロ率： **{soraru_rate:.1f}%**")
        comment = generate_comment(soraru_rate)
        st.write(comment)

        st.markdown("---")

        # ⑤ おすすめ曲ランキング（TOP5）
        st.subheader("④ あなたに近い そらる楽曲 TOP5")

        top5 = result.head(5).reset_index(drop=True)

        # 1位だけ大きく＋YouTube埋め込み
        top1 = top5.iloc[0]
        st.markdown(f"#### 🥇 第1位：{top1['song']}  （{top1['score']:.1f}%）")
        if isinstance(top1["url"], str) and top1["url"]:
            st.video(top1["url"])
            st.write(f"[YouTubeで開く]({top1['url']})")

        # 2〜5位
        for i in range(1, len(top5)):
            row = top5.iloc[i]
            st.markdown(f"#### 🥈 第{i+1}位：{row['song']}  （{row['score']:.1f}%）")
            if isinstance(row["url"], str) and row["url"]:
                st.write(f"[YouTubeで開く]({row['url']})")

        st.markdown("---")

        # ⑥ Xでシェア（ここはあとで一緒に実装）
        st.subheader("⑤ Xでシェア")
        st.write("※ この部分は、診断結果画面を画像として保存してXに投稿できるように、あとで一緒に作り込みましょう。")
        # ここに将来：
        # - 結果画面を画像として保存
        # - その画像とテキストをX投稿用URLに埋め込む
        # などを実装予定
else:
    st.info("音声ファイルをアップロードしてから「精密解析スタート」を押してください。")