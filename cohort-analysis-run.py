import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Cohort Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

def upload_data():
    st.header("📤 Upload Your Data")

    with st.container():
        st.markdown("""
        ### Data Format Requirements
        Upload a CSV file with the following columns:
        - `customer_id`: Unique identifier for each customer
        - `date`: Date of the event (YYYY-MM-DD format)
        - `event_type`: Type of event (e.g., registration, purchase)
        - `segment`: (Optional) Category/segment of the customer
        """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            required_columns = ['customer_id', 'date', 'event_type']
            if not all(col in data.columns for col in required_columns):
                st.error("❌ CSV file must contain: customer_id, date, and event_type")
                return

            data['date'] = pd.to_datetime(data['date'])
            st.session_state.data = data

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Data Preview")
                st.dataframe(data.head(), use_container_width=True)

            with col2:
                st.markdown("### 📈 Data Statistics")
                stats = {
                    "Total Records": len(data),
                    "Unique Customers": data['customer_id'].nunique(),
                    "Date Range": f"{data['date'].min().date()} to {data['date'].max().date()}",
                    "Event Types": list(data['event_type'].unique()),
                    "Segments": list(data['segment'].unique()) if 'segment' in data.columns else "No segments found"
                }
                st.json(stats)

            st.success("✅ Data uploaded successfully! Proceed to Cohort Settings.")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.session_state.data = None


def cohort_settings():
    st.header("⚙️ Cohort Settings")

    # Show explanation about cohort base vs retention event
    with st.expander("ℹ️ Understanding Cohort Base vs Retention Event", expanded=True):
        st.markdown("""
        ### Key Concepts

        #### 🎯 Cohort Base
        - This is the initial event that groups users into cohorts
        - Examples: registration date, first purchase date, installation date
        - This event marks the start of a user's journey

        #### 📊 Retention Event
        - This is the event you're tracking to measure ongoing engagement
        - Examples: subsequent logins, purchases, or any activity after the initial event
        - Used to calculate whether a user is retained in each period

        #### 📅 Cohort Type
        - Determines how users are grouped temporally
        - Daily: Most granular analysis
        - Weekly: Good for short-term patterns
        - Monthly: Best for long-term trends
        """)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    data = st.session_state.data
    event_types = data['event_type'].unique().tolist()

    col1, col2 = st.columns(2)

    with col1:
        cohort_basis = st.selectbox("Select Cohort Base", event_types,
                                    help="The event that marks the start of a user's journey")
        cohort_type = st.selectbox("Select Cohort Type",
                                   ["Daily", "Weekly", "Monthly"],
                                   help="How to group users temporally")

    with col2:
        retention_event = st.selectbox("Select Retention Event", event_types,
                                       help="The event to track for measuring ongoing engagement")
        retention_type = st.selectbox("Analysis Type",
                                      ["Retention Rate", "Churn Rate"],
                                      help="Choose whether to show retention or churn rate")

    if st.button("💾 Save Cohort Settings", use_container_width=True):
        st.session_state.cohort_settings = {
            "cohort_basis": cohort_basis,
            "cohort_type": cohort_type,
            "retention_event": retention_event,
            "retention_type": retention_type
        }
        st.success("✅ Cohort settings saved! Proceed to Date Range selection.")


def date_range_selection():
    st.header("📅 Date Range Selection")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    data = st.session_state.data
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date", min_date,
                                   min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date,
                                 min_value=min_date, max_value=max_date)

    if st.button("💾 Save Date Range", use_container_width=True):
        if start_date <= end_date:
            st.session_state.date_range = {"start": start_date, "end": end_date}
            st.success("✅ Date range saved! Proceed to Segmentation.")
        else:
            st.error("❌ End date must be after start date.")


def segmentation():
    st.header("🔍 Segmentation")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return

    data = st.session_state.data
    if 'segment' in data.columns:
        segments = data['segment'].unique().tolist()
        selected_segments = st.multiselect("Select Segments to Analyze - Up to 5",
                                           segments,
                                           default=segments[:1] if segments else None,
                                           help="Choose which segments to include in the analysis")

        if st.button("💾 Save Segments", use_container_width=True):
            st.session_state.segments = selected_segments
            st.success("✅ Segments saved! Proceed to Visualization.")
    else:
        st.info("ℹ️ No segment column found in the data. Using all data for analysis.")
        st.session_state.segments = ['All']

def create_cohort_data(data, cohort_settings, date_range, segments):
    try:
        # Filter data based on date range and cohort basis
        filtered_data = data[
            (data['date'] >= pd.Timestamp(date_range['start'])) &
            (data['date'] <= pd.Timestamp(date_range['end'])) &
            (data['event_type'] == cohort_settings['cohort_basis'])
        ]

        # Create cohorts based on the cohort type
        cohort_data = filtered_data.groupby('customer_id')['date'].min().reset_index()

        if cohort_settings['cohort_type'] == 'Daily':
            cohort_data['cohort'] = cohort_data['date'].dt.strftime('%Y-%m-%d')
        elif cohort_settings['cohort_type'] == 'Weekly':
            cohort_data['cohort'] = cohort_data['date'].dt.to_period('W').astype(str)
        else:  # Monthly
            cohort_data['cohort'] = cohort_data['date'].dt.to_period('M').astype(str)

        retention_events = data[data['event_type'] == cohort_settings['retention_event']]
        retention_data = []

        for segment in segments:
            if segment != 'All':
                segment_cohorts = cohort_data[cohort_data['customer_id'].isin(
                    data[data['segment'] == segment]['customer_id']
                )]
            else:
                segment_cohorts = cohort_data

            # Calculate retention/churn for each cohort and period
            for cohort in sorted(segment_cohorts['cohort'].unique()):
                cohort_users = segment_cohorts[segment_cohorts['cohort'] == cohort]['customer_id']
                cohort_size = len(cohort_users)

                if cohort_size == 0:
                    continue

                # Parse cohort date based on cohort type
                if cohort_settings['cohort_type'] == 'Daily':
                    cohort_start = pd.to_datetime(cohort)
                else:
                    cohort_start = pd.Period(cohort).to_timestamp()

                for period in range(13):  # 0 to 12 periods
                    # Calculate period start and end
                    if cohort_settings['cohort_type'] == 'Daily':
                        period_start = cohort_start + pd.Timedelta(days=period)
                        period_end = period_start + pd.Timedelta(days=1)
                    elif cohort_settings['cohort_type'] == 'Weekly':
                        period_start = cohort_start + pd.Timedelta(weeks=period)
                        period_end = period_start + pd.Timedelta(weeks=1)
                    else:  # Monthly
                        period_start = cohort_start + pd.DateOffset(months=period)
                        period_end = period_start + pd.DateOffset(months=1)

                    # Count retained users
                    retained_users = retention_events[
                        (retention_events['customer_id'].isin(cohort_users)) &
                        (retention_events['date'] >= period_start) &
                        (retention_events['date'] < period_end)
                    ]['customer_id'].nunique()

                    rate = retained_users / cohort_size
                    if cohort_settings['retention_type'] == 'Churn Rate':
                        rate = 1 - rate

                    retention_data.append({
                        'cohort': cohort,
                        'period': period,
                        'rate': rate,
                        'segment': segment,
                        'cohort_size': cohort_size,
                        'retained_users': retained_users
                    })

        return pd.DataFrame(retention_data)

    except Exception as e:
        st.error(f"❌ An error occurred while processing the data: {str(e)}")
        return pd.DataFrame()


def visualization():
    st.header("📊 Visualization")

    if (st.session_state.data is None or
            not st.session_state.cohort_settings or
            not st.session_state.date_range or
            not st.session_state.segments):
        st.warning("⚠️ Please complete all previous steps before visualization.")
        return

    if st.button("🔄 Run Analysis", use_container_width=True):
        with st.spinner("Processing data..."):
            retention_data = create_cohort_data(
                st.session_state.data,
                st.session_state.cohort_settings,
                st.session_state.date_range,
                st.session_state.segments
            )

        with tab1:
            rate_type = st.session_state.cohort_settings['retention_type']
            
            # Aggregate data for line chart
            aggregated_data = retention_data.groupby(['segment', 'period'])['rate'].mean().reset_index()
        
            if aggregated_data.empty:
                st.warning("Insufficient data to display the line chart.")
                return
        
            colors = px.colors.qualitative.Set2
            
            # Here's where we need to change retention_data to aggregated_data
            fig = px.line(
                aggregated_data,  # Changed from retention_data to aggregated_data
                x='period',
                y='rate',
                color='segment',
                color_discrete_sequence=colors,
                labels={
                    "period": "Period",
                    "rate": rate_type,
                    "segment": "Segment"
                },
                title=f"{rate_type} by Segment"
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                ),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    tickformat='.2%'
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            for segment in st.session_state.segments:
                segment_data = retention_data[retention_data['segment'] == segment]

                if len(segment_data) == 0:
                    st.warning(f"No sufficient data to display heatmap for segment: {segment}")
                    continue

                # Create pivot tables
                pivot_data = segment_data.pivot(
                    index='cohort',
                    columns='period',
                    values='rate'
                )

                retained_users = segment_data.pivot(
                    index='cohort',
                    columns='period',
                    values='retained_users'
                )

                cohort_sizes = segment_data.pivot(
                    index='cohort',
                    columns='period',
                    values='cohort_size'
                )

                # Create hover text matrix
                hover_text = []
                for idx in range(len(pivot_data.index)):
                    hover_row = []
                    for col in range(len(pivot_data.columns)):
                        rate = pivot_data.iloc[idx, col]
                        retained = retained_users.iloc[idx, col]
                        cohort_size = cohort_sizes.iloc[idx, 0]  # Use first column for cohort size

                        if pd.notna(rate):
                            hover_row.append(
                                f"Rate: {rate:.2%}<br>" +
                                f"Retained: {int(retained)}<br>" +
                                f"Cohort Size: {int(cohort_size)}"
                            )
                        else:
                            hover_row.append("")
                    hover_text.append(hover_row)

                fig_heatmap = px.imshow(
                    pivot_data,
                    labels=dict(x="Period", y="Cohort", color=rate_type),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale="RdYlBu_r",
                    title=f"{rate_type} Heatmap - {segment} Segment"
                )

                fig_heatmap.update_traces(
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_text
                )

                fig_heatmap.update_layout(
                    xaxis_title="Period",
                    yaxis_title="Cohort",
                    xaxis=dict(tickangle=0),
                    coloraxis_colorbar=dict(
                        title=rate_type,
                        tickformat='.2%'
                    )
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

        with tab3:
            st.markdown("### 📊 Analysis Statistics")

            stats = retention_data.groupby('segment').agg({
                'rate': ['mean', 'min', 'max'],
                'cohort_size': 'sum'
            }).round(4)

            stats.columns = ['Average Rate', 'Minimum Rate', 'Maximum Rate', 'Total Users']
            stats = stats.reset_index()

            # Format percentages in the dataframe
            for col in ['Average Rate', 'Minimum Rate', 'Maximum Rate']:
                stats[col] = stats[col].apply(lambda x: f"{x:.2%}")

            st.dataframe(stats, use_container_width=True)


def main():
    # Move title to sidebar
    st.sidebar.title("🎯 Cohort Analysis Tool")

    # Update data privacy notice in sidebar
    with st.sidebar.expander("ℹ️ Data Privacy Information"):
        st.markdown("""
            **Data Handling Information**
            - When run locally, all data processing is done on your machine.
            - If deployed on Streamlit Cloud:
              - Data is temporarily stored on Streamlit's servers during the session.
              - Data is not permanently stored or used for any other purpose.
              - Data is removed when the session ends or after inactivity.
            - We use industry-standard security practices, including HTTPS encryption.
            - We recommend not uploading sensitive personal data.
            - For full details, please refer to [Streamlit's Privacy Policy](https://streamlit.io/privacy-policy).
            """)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cohort_settings' not in st.session_state:
        st.session_state.cohort_settings = {}
    if 'date_range' not in st.session_state:
        st.session_state.date_range = {}
    if 'segments' not in st.session_state:
        st.session_state.segments = []

    # Create steps in sidebar
    st.sidebar.markdown("### 📋 Analysis Steps")
    step = st.sidebar.radio("",
                            ["1. Upload Data",
                             "2. Cohort Settings",
                             "3. Date Range",
                             "4. Segmentation",
                             "5. Visualization"
                             ])

    # Show progress
    total_steps = 5
    current_step = int(step[0])
    progress = (current_step - 1) / (total_steps - 1)
    st.sidebar.progress(progress)

    # Display current step status
    st.sidebar.markdown(f"**Current Step:** {step}")

    # Execute selected step
    if step == "1. Upload Data":
        upload_data()
    elif step == "2. Cohort Settings":
        cohort_settings()
    elif step == "3. Date Range":
        date_range_selection()
    elif step == "4. Segmentation":
        segmentation()
    elif step == "5. Visualization":
        visualization()


if __name__ == "__main__":
    main()
