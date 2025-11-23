import streamlit as st
import pandas as pd
import altair as alt
import datetime
import db as db_module

def show_analytics():
    today = datetime.date.today()
    st.subheader("📊 Data Analysis")
    
    col_type, col_date = st.columns([1, 2])
    with col_type:
        report_type = st.selectbox(
            "Select Report Type",
            ["Today's Activity", "Weekly Report", "Monthly Overview", "Specific Date", "Date Range"]
        )
    
    db = db_module.get_connection()
    if db:
        try:
            df_all = pd.read_sql("SELECT name, date, time FROM attendance", db)
            db.close()
            
            if not df_all.empty:
                # Data Cleaning
                df_all["time"] = df_all["time"].astype(str).str.replace("0 days ", "", regex=False)
                df_all["date"] = pd.to_datetime(df_all["date"])
                
                filtered_df = pd.DataFrame()
                chart = None
                
                # --- FILTERING LOGIC ---
                
                # 1. TODAY'S ACTIVITY (⛔ UNCHANGED as requested)
                if report_type == "Today's Activity":
                    filtered_df = df_all[df_all["date"].dt.date == today].copy()
                    
                    if not filtered_df.empty:
                        # Convert time -> hour
                        filtered_df["hour"] = pd.to_datetime(
                            filtered_df["time"],
                            format="%H:%M:%S",
                            errors="coerce"
                        ).dt.hour
                        
                        # Group by hour
                        grouped = (
                            filtered_df.dropna(subset=["hour"])
                            .groupby("hour")["name"]
                            .count()
                            .reset_index(name="count")
                        )
                        
                        # Create full 0–23 hour range so chart is clearer
                        all_hours = pd.DataFrame({"hour": list(range(24))})
                        chart_data = all_hours.merge(grouped, on="hour", how="left").fillna({"count": 0})
                        
                        # Make hour nicely formatted (e.g., "09:00")
                        chart_data["hour_label"] = chart_data["hour"].astype(str).str.zfill(2) + ":00"
                        
                        # Line + points chart
                        chart = (
                            alt.Chart(chart_data)
                            .mark_line(point=True, strokeWidth=2, color="#6366f1")
                            .encode(
                                x=alt.X("hour_label:O", title="Hour of Day"),
                                y=alt.Y("count:Q", title=None), # Removed title for cleaner look
                                tooltip=["hour_label", "count"]
                            )
                            .properties(height=280)
                        )

                # 2. WEEKLY REPORT (✅ MODIFIED: Lollipop Chart)
                elif report_type == "Weekly Report":
                    start = pd.to_datetime(today) - pd.Timedelta(days=7)
                    filtered_df = df_all[df_all["date"] >= start]
                    if not filtered_df.empty:
                        weekly = filtered_df.groupby("date")["name"].nunique().reset_index(name="name")
                        
                        # Base Chart
                        base = alt.Chart(weekly).encode(
                            x=alt.X("date:T", axis=alt.Axis(format="%a %d", title=None, labelAngle=0))
                        )
                        
                        # The "Stick"
                        rule = base.mark_rule(color="#10b981", size=2).encode(
                            y=alt.Y("name:Q", title=None, axis=None)
                        )
                        
                        # The "Pop" (Circle)
                        circle = base.mark_circle(size=100, color="#10b981").encode(
                            y="name:Q",
                            tooltip=["date", "name"]
                        )
                        
                        # The Label (Number on top)
                        text = base.mark_text(align='center', baseline='bottom', dy=-8, fontWeight='bold').encode(
                            y="name:Q", text="name:Q"
                        )
                        
                        chart = (rule + circle + text).properties(height=250)

                # 3. MONTHLY OVERVIEW (✅ MODIFIED: Line + Points for clarity)
                elif report_type == "Monthly Overview":
                    filtered_df = df_all[
                        (df_all["date"].dt.month == today.month) &
                        (df_all["date"].dt.year == today.year)
                    ]
                    if not filtered_df.empty:
                        monthly = filtered_df.groupby("date")["name"].nunique().reset_index(name="name")
                        
                        base = alt.Chart(monthly).encode(
                            x=alt.X("date:T", axis=alt.Axis(format="%d %b", title=None))
                        )
                        
                        # Line
                        line = base.mark_line(color="#f59e0b", strokeWidth=3).encode(
                            y=alt.Y("name:Q", axis=None)
                        )
                        
                        # Points
                        points = base.mark_circle(size=80, color="#f59e0b").encode(
                            y="name:Q", tooltip=["date", "name"]
                        )
                        
                        # Labels
                        text = base.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11).encode(
                            y="name:Q", text="name:Q"
                        )
                        
                        chart = (line + points + text).properties(height=250)

                elif report_type == "Specific Date":
                    with col_date:
                        sel_date = st.date_input("Select Date", today)
                    filtered_df = df_all[df_all["date"].dt.date == sel_date]

                elif report_type == "Date Range":
                    with col_date:
                        d_range = st.date_input(
                            "Select Range",
                            [today - datetime.timedelta(days=7), today]
                        )
                    if len(d_range) == 2:
                        filtered_df = df_all[
                            (df_all["date"].dt.date >= d_range[0]) &
                            (df_all["date"].dt.date <= d_range[1])
                        ]
                        
                # --- DISPLAY RESULTS ---
                if not filtered_df.empty:
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Total Records", len(filtered_df))
                    m2.metric("Unique Students", filtered_df["name"].nunique())
                    
                    st.divider()

                    # Show Chart
                    if chart is not None:
                        st.markdown(f"##### 📈 {report_type}")
                        st.altair_chart(
                            chart.configure_view(strokeWidth=0).configure_axis(grid=False, domain=False),
                            use_container_width=True
                        )
                    
                    # Show Table
                    st.markdown("### 📋 Detailed Records")
                    st.dataframe(
                        filtered_df[["name", "date", "time"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.download_button(
                        "Download CSV",
                        filtered_df.to_csv(index=False).encode("utf-8"),
                        "report.csv",
                        "text/csv"
                    )
                else: 
                    st.warning("No data found for selection.")
        except Exception as e: 
            st.error(f"Data Error: {e}")