from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

DATA_FILE = Path("data/sunspots.csv")
REQUIRED_COLUMNS = {"YEAR", "SUNACTIVITY"}


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV file is missing required columns: {missing_list}")

    cleaned_df = df.copy()
    cleaned_df["YEAR"] = pd.to_numeric(cleaned_df["YEAR"], errors="coerce")
    cleaned_df["SUNACTIVITY"] = pd.to_numeric(cleaned_df["SUNACTIVITY"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["YEAR", "SUNACTIVITY"]).copy()

    if cleaned_df.empty:
        raise ValueError("No valid rows were found after cleaning YEAR and SUNACTIVITY.")

    cleaned_df["YEAR_INT"] = cleaned_df["YEAR"].astype(int)
    cleaned_df["DATE"] = pd.to_datetime(cleaned_df["YEAR_INT"].astype(str), format="%Y")
    cleaned_df = cleaned_df.sort_values("DATE").set_index("DATE")
    return cleaned_df


@st.cache_data
def load_data_from_path(file_path: str) -> pd.DataFrame:
    return prepare_data(pd.read_csv(file_path))


def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    return prepare_data(pd.read_csv(uploaded_file))


def plot_advanced_sunspot_visualizations(
    df: pd.DataFrame, sunactivity_col: str = "SUNACTIVITY"
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    axs[0, 0].plot(df.index, df[sunactivity_col], color="blue", linewidth=1.5)
    axs[0, 0].set_title("Sunspot Activity Over Time")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Sunspot Count")
    axs[0, 0].grid(True)

    data = df[sunactivity_col].dropna().to_numpy()
    if len(data) > 0:
        axs[0, 1].hist(
            data,
            bins=min(30, max(5, len(data) // 2)),
            density=True,
            alpha=0.6,
            color="gray",
            label="Histogram",
        )

        if len(data) > 1 and np.unique(data).size > 1:
            xs = np.linspace(data.min(), data.max(), 200)
            density = gaussian_kde(data)
            axs[0, 1].plot(xs, density(xs), color="red", linewidth=2, label="Density")

    axs[0, 1].set_title("Distribution of Sunspot Activity")
    axs[0, 1].set_xlabel("Sunspot Count")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    df_20th = df.loc["1900":"2000"]
    if not df_20th.empty:
        axs[1, 0].boxplot(df_20th[sunactivity_col].dropna(), vert=False)
    axs[1, 0].set_title("Boxplot of Sunspot Activity (1900-2000)")
    axs[1, 0].set_xlabel("Sunspot Count")

    years = df["YEAR_INT"].to_numpy()
    sun_activity = df[sunactivity_col].to_numpy()

    mask = ~np.isnan(sun_activity)
    years_clean = years[mask]
    sun_activity_clean = sun_activity[mask]

    if len(years_clean) > 0:
        axs[1, 1].scatter(
            years_clean,
            sun_activity_clean,
            s=10,
            alpha=0.5,
            label="Data Points",
        )

    if len(years_clean) > 1 and np.unique(years_clean).size > 1:
        coef = np.polyfit(years_clean, sun_activity_clean, 1)
        trend = np.poly1d(coef)
        axs[1, 1].plot(
            years_clean,
            trend(years_clean),
            color="red",
            linewidth=2,
            label="Trend Line",
        )

    axs[1, 1].set_title("Trend of Sunspot Activity")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("Sunspot Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def render_data_source_help():
    st.info(
        "기본 데이터 파일 `data/sunspots.csv`를 찾지 못했습니다. "
        "사이드바에서 CSV를 업로드하면 바로 시각화를 볼 수 있습니다."
    )
    st.code("YEAR,SUNACTIVITY\n1900,9.5\n1901,2.7\n1902,5.0", language="csv")


st.set_page_config(page_title="Sunspots Dashboard", layout="wide")
st.title("태양흑점 데이터 분석 대시보드")
st.markdown("이 대시보드는 태양흑점 데이터를 다양한 시각화 방법으로 보여줍니다.")

st.sidebar.header("데이터 불러오기")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

df = None
data_source_label = None

try:
    if uploaded_file is not None:
        df = load_data_from_upload(uploaded_file)
        data_source_label = f"업로드 파일: {uploaded_file.name}"
    elif DATA_FILE.exists():
        df = load_data_from_path(str(DATA_FILE))
        data_source_label = f"기본 파일: {DATA_FILE.as_posix()}"
    else:
        render_data_source_help()
        st.stop()

    st.caption(f"데이터 소스: {data_source_label}")
    st.subheader("태양흑점 데이터 종합 시각화")
    fig = plot_advanced_sunspot_visualizations(df)
    st.pyplot(fig)
    st.dataframe(df[["YEAR_INT", "SUNACTIVITY"]].rename(columns={"YEAR_INT": "YEAR"}))

except Exception as exc:
    st.error(f"오류가 발생했습니다: {exc}")
    st.info(
        "CSV 파일에는 `YEAR`와 `SUNACTIVITY` 컬럼이 있어야 하며, "
        "두 컬럼 모두 숫자로 변환 가능한 값이어야 합니다."
    )
