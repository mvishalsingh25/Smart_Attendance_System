import streamlit as st
import pandas as pd
import altair as alt
import datetime
import db as db_module

def show_dashboard():
    today = datetime.date.today()
    db = db_module.get_connection()
    present_count = 0
    last_arrival, last_time = "None", "--:--"

    if db:
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance WHERE date = %s", (today,))
        present_count = cursor.fetchone()[0]
        cursor.execute("SELECT name, time FROM attendance WHERE date = %s ORDER BY time DESC LIMIT 1", (today,))
        last_row = cursor.fetchone()
        if last_row:
            last_arrival = last_row[0]
            t = last_row[1]
            if isinstance(t, datetime.timedelta): t = (datetime.datetime.min + t).time()
            last_time = t.strftime("%I:%M %p")
        db.close()

    st.subheader("✨ Today's Highlights")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Present Today", present_count)
    with m2: st.metric("Latest Arrival", last_arrival, delta=last_time)
    with m3: st.metric("System Health", "Optimal", delta="Online")

    st.markdown("### 📅 Quick Activity View")
    db = db_module.get_connection()
    if db:
        df = pd.read_sql("SELECT date, COUNT(DISTINCT name) as count FROM attendance GROUP BY date ORDER BY date DESC LIMIT 7", db)
        db.close()
        if not df.empty:
            chart = alt.Chart(df).mark_bar(color="#6366f1", cornerRadiusEnd=4).encode(
                x=alt.X('date:T', axis=alt.Axis(format="%b %d", title="Date")),
                y=alt.Y('count:Q', title="Attendees")
            ).properties(height=300)
            st.altair_chart(chart, width="stretch")
        else: st.info("No recent activity.")