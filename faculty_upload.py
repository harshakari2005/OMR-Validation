import streamlit as st
import pandas as pd, sqlite3, json, datetime
from parser import parse_excel_to_json

DB_PATH = "omr_system.db"

def save_to_db(exam_id, set_name, answers):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS answer_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_id TEXT,
                    set_name TEXT,
                    answers_json TEXT,
                    uploaded_on TEXT)""")
    cur.execute("INSERT INTO answer_keys (exam_id,set_name,answers_json,uploaded_on) VALUES (?,?,?,?)",
                (exam_id, set_name, json.dumps(answers), datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("Admin Dashboard — Upload Answer Keys")

exam_id = st.text_input("Enter Exam ID / Week Number (e.g., Week_5)")
uploaded = st.file_uploader("Upload Answer Key Excel", type=["xlsx","csv"])

if uploaded and exam_id:
    df = pd.read_excel(uploaded, sheet_name=None)
    parsed_keys = parse_excel_to_json(df)
    for set_name, mapping in parsed_keys.items():
        save_to_db(exam_id, set_name, mapping["answers"])
    st.success(f"✅ Uploaded {len(parsed_keys)} sets for {exam_id}")

# View uploaded keys
if st.button("Show Uploaded Keys"):
    conn = sqlite3.connect(DB_PATH)
    df_keys = pd.read_sql("SELECT exam_id,set_name,uploaded_on FROM answer_keys", conn)
    conn.close()
    st.dataframe(df_keys)
