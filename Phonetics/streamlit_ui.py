import ast
import json
import re
import time

import pandas as pd
import streamlit as st

from st_keyup import st_keyup
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

st.set_page_config(
    page_title="NamingChecker",
    layout="wide",
)


@st.cache_data(show_spinner="Загрузка базы данных...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv("db.csv", low_memory=False)

    def parse_classes(val):
        if pd.isna(val):
            return set()
        try:
            return set(int(x) for x in json.loads(val))
        except Exception:
            try:
                parsed = ast.literal_eval(str(val))
                if isinstance(parsed, (list, tuple)):
                    return set(int(x) for x in parsed)
            except Exception:
                pass
        return set()

    df["mktu_classes"] = df["class_number"].apply(parse_classes)
    df["name_clean"] = (
        df["mark_preprocessed"].fillna(df["mark_significant_normal"]).fillna("").astype(str).str.strip()
    )
    return df[df["name_clean"] != ""].copy().reset_index(drop=True)


TRANSLIT_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

REVERSE_TRANSLIT = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з",
}

PHONETIC_GROUPS = {
    "b": "P",
    "p": "P",
    "d": "T",
    "t": "T",
    "g": "K",
    "k": "K",
    "c": "K",
    "q": "K",
    "v": "F",
    "f": "F",
    "w": "F",
    "z": "S",
    "s": "S",
    "x": "KS",
    "j": "J",
    "l": "L",
    "r": "R",
    "m": "M",
    "n": "N",
    "h": "H",
    "a": "A",
    "e": "E",
    "i": "I",
    "o": "O",
    "u": "U",
    "y": "I",
}


def to_latin(text: str) -> str:
    result = []
    for ch in text.lower():
        result.append(TRANSLIT_MAP.get(ch, ch))
    return "".join(result)


def phonetic_code(text: str) -> str:
    latin = re.sub(r"[^a-z]", "", to_latin(text))
    code, prev = [], None
    for ch in latin:
        mapped = PHONETIC_GROUPS.get(ch, ch.upper())
        if mapped != prev:
            code.append(mapped)
            prev = mapped
    return "".join(code)


def phonetic_sim(q: str, c: str) -> float:
    q_lat = re.sub(r"[^a-z]", "", to_latin(q))
    c_lat = re.sub(r"[^a-z]", "", to_latin(c))
    if not q_lat or not c_lat:
        return 0.0
    jw = JaroWinkler.similarity(q_lat, c_lat)
    qc, cc = phonetic_code(q), phonetic_code(c)
    jw_p = JaroWinkler.similarity(qc, cc) if qc and cc else 0.0
    return 0.6 * jw + 0.4 * jw_p


def combined_score(query: str, candidate: str) -> float:
    q, c = query.lower().strip(), candidate.lower().strip()
    if not q or not c:
        return 0.0

    lev = Levenshtein.normalized_similarity(q, c)
    jw = JaroWinkler.similarity(q, c)
    tsrt = fuzz.token_sort_ratio(q, c) / 100.0
    tset = fuzz.token_set_ratio(q, c) / 100.0
    part = fuzz.partial_ratio(q, c) / 100.0

    min_len = min(len(q), len(c))
    max_len = max(len(q), len(c))
    prefix = sum(1 for i in range(min_len) if q[i] == c[i]) / max_len if max_len else 0.0

    text_best = max(lev, jw, tsrt)
    containment = min(max(part, tset, prefix * 1.5), 1.0)
    phon = phonetic_sim(query, candidate)

    return 0.40 * text_best + 0.30 * phon + 0.30 * containment


def search(query: str, mktu_filter: list[int], top_n: int, df: pd.DataFrame):
    if mktu_filter:
        mktu_set = set(mktu_filter)
        df = df[df["mktu_classes"].apply(lambda x: bool(x & mktu_set))]

    if df.empty:
        return pd.DataFrame()

    from rapidfuzz.process import extract

    names = df["name_clean"].tolist()
    prefilter_n = min(len(names), top_n * 4)
    quick = extract(query.lower(), names, scorer=fuzz.WRatio, limit=prefilter_n)

    rows = []
    for match_name, _, idx in quick:
        row = df.iloc[idx]
        score = combined_score(query, match_name) * 100
        rows.append(
            {
                "Сходство, %": round(score, 1),
                "Нейминг": match_name,
                "Оригинал": row.get("mark_significant", ""),
                "Классы МКТУ": ", ".join(str(x) for x in sorted(row["mktu_classes"])),
                "Реестр": row.get("certificate_link", ""),
            }
        )

    result = pd.DataFrame(rows).sort_values("Сходство, %", ascending=False).head(top_n).reset_index(drop=True)
    result.index += 1
    return result


st.title("NamingChecker")
st.caption("Проверка нейминга по базе товарных знаков РФ")

df_all = load_data()

with st.sidebar:
    st.header("Параметры поиска")

    top_n = st.slider("Максимум результатов", min_value=10, max_value=200, value=50, step=10)

    all_classes = sorted({c for classes in df_all["mktu_classes"] for c in classes})
    mktu_filter = st.multiselect(
        "Классы МКТУ (пусто = все)",
        options=all_classes,
        format_func=lambda x: f"Класс {x}",
    )

    min_score = st.slider("Мин. порог сходства, %", min_value=0, max_value=100, value=0, step=5)

    st.divider()
    st.metric("Всего записей в базе", f"{len(df_all):,}")

query: str = str(
    st_keyup(  # type: ignore[arg-type]
        "Введите наименование для проверки",
        placeholder="Например: EUROPLEX, Честное Мясо, АРКЛАЙН...",
        debounce=300,
        label_visibility="collapsed",
    )
    or ""
)

if query.strip():
    t0 = time.time()
    with st.spinner("Поиск…"):
        results = search(query.strip(), list(mktu_filter), top_n, df_all)

    elapsed = time.time() - t0

    if results.empty:
        st.warning("Совпадений не найдено. Попробуйте изменить параметры фильтрации.")
    else:
        if min_score > 0:
            results = results[results["Сходство, %"] >= min_score]

        st.success(f"Найдено **{len(results)}** результатов за {elapsed:.2f}с")

        col1, col2, col3 = st.columns(3)
        col1.metric("Макс. сходство", f"{results['Сходство, %'].max():.1f}%")
        col2.metric("Ср. сходство", f"{results['Сходство, %'].mean():.1f}%")
        col3.metric("≥ 80%", int((results["Сходство, %"] >= 80).sum()))

        def color_score(val):
            if val >= 80:
                return "background-color: #ffe0e0; font-weight: bold"
            elif val >= 60:
                return "background-color: #fff3cd"
            else:
                return ""

        display_df = results.copy()

        display_df["Реестр"] = display_df["Реестр"].apply(
            lambda url: (
                f'<a href="{url}" target="_blank">↗</a>'
                if pd.notna(url) and str(url).startswith("http")
                else ""
            )
        )

        st.dataframe(
            results.drop(columns=["Реестр"]).style.applymap(color_score, subset=["Сходство, %"]),
            use_container_width=True,
            height=600,
        )

        has_links = results["Реестр"].apply(lambda u: str(u).startswith("http")).any()
        if has_links:
            with st.expander("Ссылки на реестр"):
                for _, row in results.iterrows():
                    url = row["Реестр"]
                    if str(url).startswith("http"):
                        st.markdown(f"**{row['Нейминг']}** — [{url}]({url})")
else:
    st.info("Введите название в строку поиска выше для начала проверки")
