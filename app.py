"""
Personal Nutrition & Workout Tracker — Streamlit App
---------------------------------------------------
Run locally:
  1) pip install streamlit pandas numpy
  2) streamlit run app.py

This single-file app stores data in a local SQLite database (tracker.db).
It lets you:
  • Log weight daily
  • Log meals + foods (with macros + key vitamins/minerals)
  • Track workouts and calories burned
  • See daily macro/micro totals vs. your targets
  • Get smart food suggestions to hit remaining goals

Notes:
  • A small starter food library is included (per 100 g). Add your own foods anytime.
  • Targets are editable in the Settings tab. Defaults are reasonable for a 22 y/o male.
  • Units: grams for solids, milliliters for liquids (treat 1 ml ≈ 1 g if density ~ water).
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = "tracker.db"

# -----------------------------
# Nutrient Schema
# -----------------------------
# All nutrient amounts are per 100 g for foods in the library.
# We keep a compact but useful set of micronutrients.
NUTRIENTS = [
    # Macros
    "calories", "protein_g", "carb_g", "fat_g", "fiber_g", "sugar_g",
    # Minerals
    "calcium_mg", "iron_mg", "magnesium_mg", "potassium_mg", "sodium_mg",
    # Vitamins
    "vitamin_a_mcg", "vitamin_c_mg", "vitamin_d_mcg", "vitamin_e_mg",
    "vitamin_k_mcg", "vitamin_b1_mg", "vitamin_b2_mg", "vitamin_b3_mg",
    "vitamin_b6_mg", "vitamin_b12_mcg", "folate_mcg",
]

DEFAULT_TARGETS = {
    # Change these in Settings as needed
    "calories": 2600,
    "protein_g": 160,  # ~2 g/kg if ~80 kg
    "carb_g": 300,
    "fat_g": 80,
    "fiber_g": 30,
    "sugar_g": 40,
    # Minerals (general adult targets)
    "calcium_mg": 1000,
    "iron_mg": 8,
    "magnesium_mg": 400,
    "potassium_mg": 3500,
    "sodium_mg": 2300,
    # Vitamins
    "vitamin_a_mcg": 900,
    "vitamin_c_mg": 90,
    "vitamin_d_mcg": 15,
    "vitamin_e_mg": 15,
    "vitamin_k_mcg": 120,
    "vitamin_b1_mg": 1.2,
    "vitamin_b2_mg": 1.3,
    "vitamin_b3_mg": 16,
    "vitamin_b6_mg": 1.3,
    "vitamin_b12_mcg": 2.4,
    "folate_mcg": 400,
}

STARTER_FOODS = [
    # name, per 100 g nutrients
    {
        "name": "Chicken breast, cooked",
        "calories": 165, "protein_g": 31, "carb_g": 0, "fat_g": 3.6, "fiber_g": 0, "sugar_g": 0,
        "calcium_mg": 15, "iron_mg": 1.0, "magnesium_mg": 29, "potassium_mg": 256, "sodium_mg": 74,
        "vitamin_a_mcg": 5, "vitamin_c_mg": 0, "vitamin_d_mcg": 0, "vitamin_e_mg": 0.3,
        "vitamin_k_mcg": 0, "vitamin_b1_mg": 0.07, "vitamin_b2_mg": 0.1, "vitamin_b3_mg": 13.7,
        "vitamin_b6_mg": 0.6, "vitamin_b12_mcg": 0.3, "folate_mcg": 4,
    },
    {
        "name": "Egg, whole, boiled",
        "calories": 155, "protein_g": 13, "carb_g": 1.1, "fat_g": 11, "fiber_g": 0, "sugar_g": 1.1,
        "calcium_mg": 50, "iron_mg": 1.2, "magnesium_mg": 10, "potassium_mg": 126, "sodium_mg": 124,
        "vitamin_a_mcg": 160, "vitamin_c_mg": 0, "vitamin_d_mcg": 2, "vitamin_e_mg": 1.1,
        "vitamin_k_mcg": 0.3, "vitamin_b1_mg": 0.04, "vitamin_b2_mg": 0.5, "vitamin_b3_mg": 0.1,
        "vitamin_b6_mg": 0.12, "vitamin_b12_mcg": 1.1, "folate_mcg": 47,
    },
    {
        "name": "Greek yogurt, plain, nonfat",
        "calories": 59, "protein_g": 10, "carb_g": 3.6, "fat_g": 0.4, "fiber_g": 0, "sugar_g": 3.2,
        "calcium_mg": 110, "iron_mg": 0.1, "magnesium_mg": 11, "potassium_mg": 141, "sodium_mg": 36,
        "vitamin_a_mcg": 2, "vitamin_c_mg": 0.5, "vitamin_d_mcg": 0, "vitamin_e_mg": 0,
        "vitamin_k_mcg": 0, "vitamin_b1_mg": 0.03, "vitamin_b2_mg": 0.2, "vitamin_b3_mg": 0.1,
        "vitamin_b6_mg": 0.05, "vitamin_b12_mcg": 0.8, "folate_mcg": 7,
    },
    {
        "name": "Oats, dry",
        "calories": 389, "protein_g": 16.9, "carb_g": 66.3, "fat_g": 6.9, "fiber_g": 10.6, "sugar_g": 0.9,
        "calcium_mg": 54, "iron_mg": 4.7, "magnesium_mg": 177, "potassium_mg": 429, "sodium_mg": 2,
        "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "vitamin_d_mcg": 0, "vitamin_e_mg": 0.4,
        "vitamin_k_mcg": 2, "vitamin_b1_mg": 0.76, "vitamin_b2_mg": 0.14, "vitamin_b3_mg": 1.1,
        "vitamin_b6_mg": 0.12, "vitamin_b12_mcg": 0, "folate_mcg": 56,
    },
    {
        "name": "Banana",
        "calories": 89, "protein_g": 1.1, "carb_g": 22.8, "fat_g": 0.3, "fiber_g": 2.6, "sugar_g": 12.2,
        "calcium_mg": 5, "iron_mg": 0.3, "magnesium_mg": 27, "potassium_mg": 358, "sodium_mg": 1,
        "vitamin_a_mcg": 3, "vitamin_c_mg": 8.7, "vitamin_d_mcg": 0, "vitamin_e_mg": 0.1,
        "vitamin_k_mcg": 0.5, "vitamin_b1_mg": 0.03, "vitamin_b2_mg": 0.07, "vitamin_b3_mg": 0.7,
        "vitamin_b6_mg": 0.37, "vitamin_b12_mcg": 0, "folate_mcg": 20,
    },
    {
        "name": "Brown rice, cooked",
        "calories": 123, "protein_g": 2.7, "carb_g": 25.6, "fat_g": 1.0, "fiber_g": 1.8, "sugar_g": 0.4,
        "calcium_mg": 10, "iron_mg": 0.4, "magnesium_mg": 43, "potassium_mg": 86, "sodium_mg": 4,
        "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "vitamin_d_mcg": 0, "vitamin_e_mg": 0.1,
        "vitamin_k_mcg": 0.2, "vitamin_b1_mg": 0.1, "vitamin_b2_mg": 0.02, "vitamin_b3_mg": 1.5,
        "vitamin_b6_mg": 0.1, "vitamin_b12_mcg": 0, "folate_mcg": 9,
    },
    {
        "name": "Spinach, raw",
        "calories": 23, "protein_g": 2.9, "carb_g": 3.6, "fat_g": 0.4, "fiber_g": 2.2, "sugar_g": 0.4,
        "calcium_mg": 99, "iron_mg": 2.7, "magnesium_mg": 79, "potassium_mg": 558, "sodium_mg": 79,
        "vitamin_a_mcg": 469, "vitamin_c_mg": 28.1, "vitamin_d_mcg": 0, "vitamin_e_mg": 2.0,
        "vitamin_k_mcg": 482, "vitamin_b1_mg": 0.08, "vitamin_b2_mg": 0.19, "vitamin_b3_mg": 0.7,
        "vitamin_b6_mg": 0.2, "vitamin_b12_mcg": 0, "folate_mcg": 194,
    },
    {
        "name": "Olive oil",
        "calories": 884, "protein_g": 0, "carb_g": 0, "fat_g": 100, "fiber_g": 0, "sugar_g": 0,
        "calcium_mg": 1, "iron_mg": 0.6, "magnesium_mg": 0, "potassium_mg": 1, "sodium_mg": 2,
        "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "vitamin_d_mcg": 0, "vitamin_e_mg": 14.4,
        "vitamin_k_mcg": 60, "vitamin_b1_mg": 0, "vitamin_b2_mg": 0, "vitamin_b3_mg": 0,
        "vitamin_b6_mg": 0, "vitamin_b12_mcg": 0, "folate_mcg": 0,
    },
]

MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snack"]

# -----------------------------
# DB Utilities
# -----------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                calories REAL, protein_g REAL, carb_g REAL, fat_g REAL, fiber_g REAL, sugar_g REAL,
                calcium_mg REAL, iron_mg REAL, magnesium_mg REAL, potassium_mg REAL, sodium_mg REAL,
                vitamin_a_mcg REAL, vitamin_c_mg REAL, vitamin_d_mcg REAL, vitamin_e_mg REAL,
                vitamin_k_mcg REAL, vitamin_b1_mg REAL, vitamin_b2_mg REAL, vitamin_b3_mg REAL,
                vitamin_b6_mg REAL, vitamin_b12_mcg REAL, folate_mcg REAL
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS weights (
                dt TEXT PRIMARY KEY,
                weight_kg REAL NOT NULL
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dt TEXT NOT NULL,
                meal_type TEXT NOT NULL
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meal_id INTEGER NOT NULL,
                food_id INTEGER NOT NULL,
                quantity_g REAL NOT NULL,
                FOREIGN KEY(meal_id) REFERENCES meals(id),
                FOREIGN KEY(food_id) REFERENCES foods(id)
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dt TEXT NOT NULL,
                kind TEXT NOT NULL, -- e.g., "Push", "Pull", "Legs", "Basketball", "Cardio"
                details TEXT,       -- freeform notes or JSON
                duration_min REAL,
                calories_burned REAL
            );
            """
        )

        # Ensure settings row
        c.execute("SELECT COUNT(*) FROM settings WHERE id=1")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO settings (id, data) VALUES (1, ?)", (json.dumps(DEFAULT_TARGETS),))

        # Seed foods if empty
        c.execute("SELECT COUNT(*) FROM foods")
        if c.fetchone()[0] == 0:
            for f in STARTER_FOODS:
                cols = ["name"] + NUTRIENTS
                vals = [f.get("name")] + [f.get(k, 0) for k in NUTRIENTS]
                placeholders = ",".join(["?"] * len(vals))
                c.execute(
                    f"INSERT OR IGNORE INTO foods ({','.join(cols)}) VALUES ({placeholders})",
                    tuple(vals),
                )
        conn.commit()


# -----------------------------
# Helper functions
# -----------------------------

def get_settings() -> Dict[str, float]:
    with get_conn() as conn:
        row = conn.execute("SELECT data FROM settings WHERE id=1").fetchone()
        return json.loads(row[0]) if row else DEFAULT_TARGETS


def save_settings(data: Dict[str, float]):
    with get_conn() as conn:
        conn.execute("UPDATE settings SET data=? WHERE id=1", (json.dumps(data),))
        conn.commit()


def list_foods(search: str = "") -> pd.DataFrame:
    with get_conn() as conn:
        if search:
            df = pd.read_sql_query(
                "SELECT * FROM foods WHERE name LIKE ? ORDER BY name",
                conn,
                params=(f"%{search}%",),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM foods ORDER BY name", conn)
    return df


def upsert_food(record: Dict):
    cols = ["name"] + NUTRIENTS
    vals = [record.get("name")] + [float(record.get(k, 0) or 0) for k in NUTRIENTS]
    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO foods ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})
            ON CONFLICT(name) DO UPDATE SET
            {', '.join([f'{k}=excluded.{k}' for k in NUTRIENTS])}
            """,
            tuple(vals),
        )
        conn.commit()


def add_weight(dt: date, weight_kg: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO weights (dt, weight_kg) VALUES (?, ?)",
            (dt.isoformat(), float(weight_kg)),
        )
        conn.commit()


def get_weight_history(days: int = 60) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT dt, weight_kg FROM weights ORDER BY dt",
            conn,
            parse_dates=["dt"],
        )
    if days:
        cutoff = pd.Timestamp(date.today() - timedelta(days=days))
        df = df[df["dt"] >= cutoff]
    return df


def ensure_meal(dt: date, meal_type: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM meals WHERE dt=? AND meal_type=?",
            (dt.isoformat(), meal_type),
        ).fetchone()
        if row:
            return int(row[0])
        cur = conn.execute(
            "INSERT INTO meals (dt, meal_type) VALUES (?, ?)",
            (dt.isoformat(), meal_type),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_entry(dt: date, meal_type: str, food_id: int, quantity_g: float):
    meal_id = ensure_meal(dt, meal_type)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO entries (meal_id, food_id, quantity_g) VALUES (?, ?, ?)",
            (meal_id, food_id, float(quantity_g)),
        )
        conn.commit()


def get_daily_entries(dt: date) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT e.id as entry_id, m.meal_type, f.name, e.quantity_g, f.*
            FROM entries e
            JOIN meals m ON e.meal_id = m.id
            JOIN foods f ON e.food_id = f.id
            WHERE m.dt = ?
            ORDER BY m.meal_type, f.name
            """,
            conn,
            params=(dt.isoformat(),),
        )
    return df


def delete_entry(entry_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM entries WHERE id=?", (int(entry_id),))
        conn.commit()


def add_workout(dt: date, kind: str, details: str, duration_min: float, calories_burned: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO workouts (dt, kind, details, duration_min, calories_burned) VALUES (?, ?, ?, ?, ?)",
            (dt.isoformat(), kind, details, float(duration_min or 0), float(calories_burned or 0)),
        )
        conn.commit()


def get_workouts(dt: date) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM workouts WHERE dt=? ORDER BY id DESC",
            conn,
            params=(dt.isoformat(),),
        )
    return df


# -----------------------------
# Nutrition math
# -----------------------------

def per_quantity(nutr_row: pd.Series, qty_g: float) -> Dict[str, float]:
    factor = (qty_g or 0) / 100.0
    return {k: float(nutr_row[k]) * factor for k in NUTRIENTS}


def summarize_day(dt: date) -> Tuple[pd.Series, pd.DataFrame]:
    entries = get_daily_entries(dt)
    if entries.empty:
        zeros = pd.Series({k: 0.0 for k in NUTRIENTS})
        return zeros, entries
    # Expand nutrients by quantity
    expanded = []
    for _, row in entries.iterrows():
        nutr = per_quantity(row, row["quantity_g"])  # dict per entry
        nutr["meal_type"] = row["meal_type"]
        nutr["name"] = row["name"]
        nutr["quantity_g"] = row["quantity_g"]
        expanded.append(nutr)
    df = pd.DataFrame(expanded)
    totals = df[NUTRIENTS].sum()
    return totals, df


def remaining_targets(totals: pd.Series, targets: Dict[str, float]) -> pd.Series:
    rem = pd.Series({k: float(targets.get(k, 0)) - float(totals.get(k, 0)) for k in NUTRIENTS})
    return rem


def suggest_foods(rem: pd.Series, top_k: int = 5, avoid_sugar: bool = True) -> pd.DataFrame:
    foods = list_foods()
    if foods.empty:
        return foods
    # Score: target protein first, then calories within remaining, penalize sugar if requested
    # Compute score per 100 g. Higher is better.
    prot_need = max(rem.get("protein_g", 0), 0)
    cal_need = max(rem.get("calories", 0), 0)

    # Normalize weights to avoid domination
    w_prot, w_cal, w_fiber, w_sugar_pen, w_sodium_pen = 3.0, 1.0, 0.5, 0.5, 0.2

    scores = []
    for _, f in foods.iterrows():
        score = 0.0
        score += w_prot * min(f["protein_g"], prot_need)
        score += w_cal * min(f["calories"], cal_need) / 10.0  # scale calories
        score += w_fiber * f.get("fiber_g", 0)
        if avoid_sugar:
            score -= w_sugar_pen * f.get("sugar_g", 0)
        score -= w_sodium_pen * (f.get("sodium_mg", 0) / 100.0)
        scores.append(score)

    foods = foods.assign(suggestion_score=scores)
    foods = foods.sort_values("suggestion_score", ascending=False).head(top_k)
    return foods[["name", "suggestion_score"] + NUTRIENTS]


# -----------------------------
# UI
# -----------------------------

def header_kpis(totals: pd.Series, rem: pd.Series, targets: Dict[str, float]):
    cols = st.columns(4)
    def chip(c, label, val, targ):
        c.metric(label, f"{val:.0f}", delta=f"/ {targ}")

    chip(cols[0], "Calories", totals.get("calories", 0), targets.get("calories", 0))
    chip(cols[1], "Protein (g)", totals.get("protein_g", 0), targets.get("protein_g", 0))
    chip(cols[2], "Carbs (g)", totals.get("carb_g", 0), targets.get("carb_g", 0))
    chip(cols[3], "Fat (g)", totals.get("fat_g", 0), targets.get("fat_g", 0))

    with st.expander("Micronutrients (today)"):
        micro_cols = st.columns(3)
        micro = [
            ("Fiber (g)", "fiber_g"), ("Sugar (g)", "sugar_g"),
            ("Calcium (mg)", "calcium_mg"), ("Iron (mg)", "iron_mg"), ("Magnesium (mg)", "magnesium_mg"),
            ("Potassium (mg)", "potassium_mg"), ("Sodium (mg)", "sodium_mg"),
            ("Vit A (mcg)", "vitamin_a_mcg"), ("Vit C (mg)", "vitamin_c_mg"), ("Vit D (mcg)", "vitamin_d_mcg"),
            ("Vit E (mg)", "vitamin_e_mg"), ("Vit K (mcg)", "vitamin_k_mcg"),
            ("B1 (mg)", "vitamin_b1_mg"), ("B2 (mg)", "vitamin_b2_mg"), ("B3 (mg)", "vitamin_b3_mg"),
            ("B6 (mg)", "vitamin_b6_mg"), ("B12 (mcg)", "vitamin_b12_mcg"), ("Folate (mcg)", "folate_mcg"),
        ]
        for i, (label, key) in enumerate(micro):
            col = micro_cols[i % 3]
            col.metric(label, f"{totals.get(key, 0):.0f}", delta=f"/ {targets.get(key, 0)}")


def page_log(today: date, targets: Dict[str, float]):
    st.subheader("Daily Log")

    # Weight
    with st.container(border=True):
        st.markdown("**Weight**")
        w = st.number_input("Today's weight (kg)", min_value=20.0, max_value=300.0, value=80.0, step=0.1, key="weight")
        if st.button("Save weight"):
            add_weight(today, w)
            st.success("Saved weight.")

    # Meals
    with st.container(border=True):
        st.markdown("**Meals**")
        meal_type = st.selectbox("Meal", MEAL_TYPES)
        foods_df = list_foods(st.text_input("Search foods"))
        st.dataframe(foods_df[["id", "name", "protein_g", "carb_g", "fat_g", "calories"]], use_container_width=True, hide_index=True)
        sel = st.number_input("Food ID to add", min_value=0, step=1)
        qty = st.number_input("Quantity (g)", min_value=0.0, step=10.0)
        if st.button("Add to meal"):
            if sel > 0 and qty > 0:
                try:
                    add_entry(today, meal_type, int(sel), float(qty))
                    st.success("Added entry.")
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.warning("Select a valid Food ID and quantity.")

        # Show today's entries
        entries = get_daily_entries(today)
        if not entries.empty:
            show = entries[["entry_id", "meal_type", "name", "quantity_g"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
            del_id = st.number_input("Entry ID to delete", min_value=0, step=1)
            if st.button("Delete entry"):
                if del_id > 0:
                    delete_entry(int(del_id))
                    st.success("Deleted entry.")

    # Add / edit foods
    with st.container(border=True):
        st.markdown("**Food Library (per 100 g)**")
        with st.expander("Add or Update a Food"):
            name = st.text_input("Name")
            cols = st.columns(4)
            vals: Dict[str, float] = {}
            for i, key in enumerate(NUTRIENTS):
                vals[key] = cols[i % 4].number_input(key, value=0.0)
            if st.button("Save food") and name:
                upsert_food({"name": name, **vals})
                st.success(f"Saved: {name}")
        st.dataframe(list_foods()[["id", "name", "calories", "protein_g", "carb_g", "fat_g"]], use_container_width=True, hide_index=True)

    # Workouts
    with st.container(border=True):
        st.markdown("**Workout & Activity**")
        kind = st.selectbox("Type", ["Push", "Pull", "Legs", "Upper", "Lower", "Cardio", "Basketball", "Other"])
        details = st.text_area("Details (sets/reps or notes)")
        duration = st.number_input("Duration (min)", min_value=0.0, step=5.0)
        kcal = st.number_input("Calories burned (kcal)", min_value=0.0, step=10.0)
        if st.button("Save workout"):
            add_workout(today, kind, details, duration, kcal)
            st.success("Saved workout.")

        wk = get_workouts(today)
        if not wk.empty:
            st.dataframe(wk[["id", "kind", "duration_min", "calories_burned", "details"]], use_container_width=True, hide_index=True)



def page_overview(today: date, targets: Dict[str, float]):
    st.subheader("Overview & Suggestions")
    totals, expanded = summarize_day(today)
    rem = remaining_targets(totals, targets)
    header_kpis(totals, rem, targets)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Today's Totals**")
        st.dataframe(pd.DataFrame(totals).T, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Remaining to Hit Targets**")
        st.dataframe(pd.DataFrame(rem).T, use_container_width=True, hide_index=True)

    st.markdown("**Food Suggestions (per 100 g)**")
    sugg = suggest_foods(rem)
    if sugg is not None and not sugg.empty:
        st.dataframe(sugg, use_container_width=True, hide_index=True)
    else:
        st.info("Add foods to your library to see suggestions.")

    if expanded is not None and not expanded.empty:
        with st.expander("Breakdown by Food (today)"):
            st.dataframe(expanded, use_container_width=True, hide_index=True)


def page_trends():
    st.subheader("Trends")
    hist = get_weight_history(days=120)
    if hist.empty:
        st.info("Log weights to see a chart.")
    else:
        st.line_chart(hist.set_index("dt")["weight_kg"])

    with get_conn() as conn:
        intake = pd.read_sql_query(
            """
            SELECT m.dt, SUM(e.quantity_g * f.calories / 100.0) AS cals,
                   SUM(e.quantity_g * f.protein_g / 100.0) AS protein_g
            FROM entries e
            JOIN meals m ON e.meal_id = m.id
            JOIN foods f ON e.food_id = f.id
            GROUP BY m.dt
            ORDER BY m.dt
            """,
            conn,
        )
    if not intake.empty:
        st.area_chart(intake.set_index("dt")[["cals", "protein_g"]])


def page_settings():
    st.subheader("Settings & Targets")
    st.markdown("Update your daily nutrition targets.")
    targets = get_settings()
    cols = st.columns(4)
    new = {}
    for i, k in enumerate(NUTRIENTS):
        default = float(targets.get(k, 0))
        new[k] = cols[i % 4].number_input(k, value=default)
    if st.button("Save targets"):
        save_settings(new)
        st.success("Targets saved.")

    st.markdown("---")
    st.caption("Tips: Protein ~1.6–2.2 g/kg bodyweight; Fiber 25–35 g; Sodium < 2300 mg unless instructed otherwise.")


# -----------------------------
# AI Coach (Ollama-only)
# -----------------------------

def call_ollama(prompt: str, model: str = "llama3.1") -> str:
    """Local Ollama integration. Requires Ollama running on localhost:11434."""
    url = "http://localhost:11434/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def build_coach_prompt(selected: date, targets: Dict[str, float]) -> str:
    totals, _ = summarize_day(selected)
    rem = remaining_targets(totals, targets)
    wk = get_workouts(selected)

    def series_to_dict(s: pd.Series) -> Dict[str, float]:
        return {k: float(s.get(k, 0)) for k in s.index}

    payload = {
        "date": selected.isoformat(),
        "today_totals": series_to_dict(totals),
        "remaining_targets": series_to_dict(rem),
        "workouts_today": wk.to_dict(orient="records") if wk is not None else [],
        "instructions": (
            "Return STRICT JSON with keys: "
            "meal_suggestions (array of {food, portion_g, why}), "
            "workout_suggestions (array of {exercise, sets, reps_or_time, rest_s, why}), "
            "notes (string). Prioritize protein first, then sensible calories; add fiber; "
            "keep sodium/sugar moderate. Workouts should target hypertrophy + calorie burn."
        )
    }
    return json.dumps(payload, indent=2)

def _render_ai_output(text: str):
    st.markdown("**AI Coach Suggestions**")
    try:
        data = json.loads(text)
    except Exception:
        # Try to find a JSON object inside free text
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0)) if m else {"notes": text}

    meals = data.get("meal_suggestions") or []
    wos = data.get("workout_suggestions") or []
    notes = data.get("notes")

    if meals:
        st.markdown("### Meal Suggestions")
        st.dataframe(pd.DataFrame(meals), use_container_width=True, hide_index=True)
    if wos:
        st.markdown("### Workout Suggestions")
        st.dataframe(pd.DataFrame(wos), use_container_width=True, hide_index=True)
    if notes:
        st.markdown("### Notes")
        st.write(notes)

def page_ai_coach(selected: date, targets: Dict[str, float]):
    st.subheader("AI Coach (Local Ollama)")

    totals, _ = summarize_day(selected)
    rem = remaining_targets(totals, targets)
    wk = get_workouts(selected)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Remaining Targets**")
        st.dataframe(pd.DataFrame(rem).T, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Today’s Workouts**")
        if wk is not None and not wk.empty:
            st.dataframe(wk[["kind", "duration_min", "calories_burned", "details"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No workouts logged today.")

    st.markdown("---")
    model = st.text_input("Ollama model", value="llama3.1",
                          help="Model must exist in Ollama: try `ollama pull llama3.1`.")
    if st.button("Get Suggestions (Ollama)"):
        try:
            prompt = build_coach_prompt(selected, targets)
            resp = call_ollama(prompt, model=model)
            _render_ai_output(resp)
        except Exception as e:
            st.error(f"Ollama request failed: {e}")

# -----------------------------
# Main
# -----------------------------

def main():
    st.set_page_config(page_title="Personal Tracker", page_icon="📊", layout="wide")
    init_db()

    st.title("Personal Nutrition & Workout Tracker")
    today = st.session_state.get("selected_date", date.today())
    selected = st.date_input("Date", value=today)
    st.session_state["selected_date"] = selected

    targets = get_settings()

    tabs = st.tabs(["Overview", "Log", "Trends", "Settings", "AI Coach"])
    with tabs[0]:
        page_overview(selected, targets)
    with tabs[1]:
        page_log(selected, targets)
    with tabs[2]:
        page_trends()
    with tabs[3]:
        page_settings()
    with tabs[4]:
        page_ai_coach(selected, targets)


    st.markdown("\n\n")
    with st.expander("How suggestions work"):
        st.write(
            """
            We prioritize protein and sensible calories to hit your remaining targets. 
            The score also rewards fiber and lightly penalizes sugar and sodium.
            Suggestions are shown per 100 g; adjust quantities to fit your needs.
            """
        )


if __name__ == "__main__":
    main()

# -----------------------------
# AI Coach (Ollama-only)
# -----------------------------
import requests
import json

def call_ollama(prompt: str, model: str = "llama3.1") -> str:
    """Local Ollama integration. Requires Ollama running on localhost:11434."""
    url = "http://localhost:11434/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def build_coach_prompt(selected: date, targets: Dict[str, float]) -> str:
    totals, _ = summarize_day(selected)
    rem = remaining_targets(totals, targets)
    wk = get_workouts(selected)

    def series_to_dict(s: pd.Series) -> Dict[str, float]:
        return {k: float(s.get(k, 0)) for k in s.index}

    payload = {
        "date": selected.isoformat(),
        "today_totals": series_to_dict(totals),
        "remaining_targets": series_to_dict(rem),
        "workouts_today": wk.to_dict(orient="records") if wk is not None else [],
        "instructions": (
            "Return STRICT JSON with keys: "
            "meal_suggestions (array of {food, portion_g, why}), "
            "workout_suggestions (array of {exercise, sets, reps_or_time, rest_s, why}), "
            "notes (string). Prioritize protein first, then sensible calories; add fiber; "
            "keep sodium/sugar moderate. Workouts should target hypertrophy + calorie burn."
        )
    }
    return json.dumps(payload, indent=2)

def _render_ai_output(text: str):
    st.markdown("**AI Coach Suggestions**")
    try:
        data = json.loads(text)
    except Exception:
        # Try to find a JSON object inside free text
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        data = json.loads(m.group(0)) if m else {"notes": text}

    meals = data.get("meal_suggestions") or []
    wos = data.get("workout_suggestions") or []
    notes = data.get("notes")

    if meals:
        st.markdown("### Meal Suggestions")
        st.dataframe(pd.DataFrame(meals), use_container_width=True, hide_index=True)
    if wos:
        st.markdown("### Workout Suggestions")
        st.dataframe(pd.DataFrame(wos), use_container_width=True, hide_index=True)
    if notes:
        st.markdown("### Notes")
        st.write(notes)

def page_ai_coach(selected: date, targets: Dict[str, float]):
    st.subheader("AI Coach (Local Ollama)")

    totals, _ = summarize_day(selected)
    rem = remaining_targets(totals, targets)
    wk = get_workouts(selected)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Remaining Targets**")
        st.dataframe(pd.DataFrame(rem).T, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Today’s Workouts**")
        if wk is not None and not wk.empty:
            st.dataframe(wk[["kind", "duration_min", "calories_burned", "details"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No workouts logged today.")

    st.markdown("---")
    model = st.text_input("Ollama model", value="llama3.1",
                          help="Model must exist in Ollama: try `ollama pull llama3.1`.")
    if st.button("Get Suggestions (Ollama)"):
        try:
            prompt = build_coach_prompt(selected, targets)
            resp = call_ollama(prompt, model=model)
            _render_ai_output(resp)
        except Exception as e:
            st.error(f"Ollama request failed: {e}")
