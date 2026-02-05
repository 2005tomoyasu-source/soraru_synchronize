# ====== Streamlit UI ======
st.set_page_config(page_title="【精密解析】そらる・シンクロ率チェッカー", layout="centered")

# ====== カスタムCSS（そらるテーマ） ======
st.markdown("""
<style>

body {
    background-color: #f4f8ff;
    font-family: 'Hiragino Maru Gothic ProN', 'Yu Gothic', sans-serif;
}

/* タイトルカード */
.title-card {
    background: linear-gradient(135deg, #e8f2ff, #cfe2ff);
    padding: 35px 20px;
    border-radius: 18px;
    border: 1px solid #aac8ff;
    margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(150, 180, 255, 0.25);
}

/* タイトル文字 */
.title-text {
    color: #1a3d7c;
    font-weight: 800;
    font-size: 2.3rem;
    line-height: 1.3;
    text-align: center;
    text-shadow: 0px 1px 3px rgba(255,255,255,0.9);
}

/* サブタイトル */
.subtitle-text {
    color: #3d5fa3;
    font-size: 1.1rem;
    text-align: center;
    margin-top: 8px;
}

/* コメントボックス */
.result-box {
    background: #e8f2ff;
    padding: 22px;
    border-left: 6px solid #7fbfff;
    border-radius: 10px;
    margin: 20px 0;
    box-shadow: 0 3px 10px rgba(150, 180, 255, 0.2);
}

/* ランキングカード */
.song-card {
    background: #ffffff;
    border: 1px solid #cfe2ff;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 14px;
    box-shadow: 0 3px 10px rgba(180, 200, 255, 0.25);
}

/* フォント統一 */
h1, h2, h3, h4, p, div {
    font-family: 'Hiragino Maru Gothic ProN', 'Yu Gothic', sans-serif;
}

</style>
""", unsafe_allow_html=True)

# ====== タイトル ======
st.markdown("""
<div class="title-card">
    <div class="title-text">
        【精密解析】<br>そらる・シンクロ率チェッカー
    </div>
    <div class="subtitle-text">
        あなたの声に最も近い楽曲も判定！
    </div>
</div>
""", unsafe_allow_html=True)

# ====== SNSアイコン ======
st.markdown("""
<div style="text-align:center;">
<a href="https://twitter.com/soraruru" target="_blank">
    <img src="https://abs.twimg.com/favicons/twitter.ico" width="32">
</a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@soraru" target="_blank">
    <img src="https://www.youtube.com/s/desktop/fe2f1f8e/img/favicon_32x32.png" width="32">
</a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ====== アップロード ======
st.subheader("① 音声ファイルをアップロード")
uploaded_file = st.file_uploader("対応形式：wav / mp3 / m4a", type=["wav", "mp3", "m4a"])

st.markdown("---")

# ====== 精密解析 ======
st.subheader("② 精密解析")
analyze_button = st.button("🔍 精密解析スタート")

if analyze_button:
    if uploaded_file is None:
        st.warning("先に音声ファイルをアップロードしてください。")
    else:
        with st.spinner("解析中…"):
            try:
                user_feat = extract_user_features_from_file(uploaded_file, duration=15)
                soraru_rate, result = analyze_all(user_feat, df)
            except Exception as e:
                st.error(f"解析中にエラーが発生しました：{e}")
                st.stop()

        st.success("解析が完了しました！")

        # ====== 結果 ======
        st.subheader("③ 結果")

        st.markdown(f"""
        <div class="result-box">
            <h2>あなたのそらる・シンクロ率： {soraru_rate:.1f}%</h2>
            <p>{generate_comment(soraru_rate)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ====== ランキング ======
        st.subheader("④ あなたに近い そらる楽曲 TOP5")

        top5 = result.head(5).reset_index(drop=True)

        # 1位
        top1 = top5.iloc[0]
        st.markdown(f"""
        <div class="song-card">
            <h3>🥇 第1位：{top1['song']}（{top1['score']:.1f}%）</h3>
        </div>
        """, unsafe_allow_html=True)
        st.video(top1["url"])
        st.write(f"[YouTubeで開く]({top1['url']})")

        # 2〜5位
        for i in range(1, len(top5)):
            row = top5.iloc[i]
            st.markdown(f"""
            <div class="song-card">
                <h4>🥈 第{i+1}位：{row['song']}（{row['score']:.1f}%）</h4>
                <a href="{row['url']}" target="_blank">YouTubeで開く</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("⑤ Xでシェア（後で実装）")

else:
    st.info("音声ファイルをアップロードしてから「精密解析スタート」を押してください。")