import pandas as pd
import io
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------- Load Sample Data ----------
csv_data = """Name,Math,Science,English,History,Computer
Amit,78,88,90,76,95
Sara,92,81,85,89,91
John,65,72,70,68,60
Priya,85,80,78,90,88
Rahul,95,94,97,93,96
Anjali,74,69,73,70,75
Karan,60,58,64,55,59
Neha,88,90,85,92,87
Vikram,55,60,50,45,52
Riya,99,97,96,94,95
Tanmay,82,85,88,81,80
Sneha,70,75,68,72,74
Manish,48,50,52,45,43
Divya,91,89,93,90,92
Rohit,62,65,60,68,70
Kavya,77,82,79,80,83
Arjun,89,87,90,88,85
Meera,66,68,64,60,65
Siddharth,59,55,57,60,62
Isha,86,88,90,85,87"""

data_io = io.StringIO(csv_data)
df = pd.read_csv(data_io)

# ---------- Streamlit Configuration ----------
st.set_page_config(page_title="📘 Student Report Card Generator", layout="wide")
st.title("Student Report Card Generator")
st.sidebar.header("Options")

# ---------- File Upload ----------
uploaded_file = st.sidebar.file_uploader("Upload student_marks.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using sample built-in dataset")
    data_io = io.StringIO(csv_data)
    df = pd.read_csv(data_io)

subjects = df.columns[1:]

# ---------- Processing ----------
df["Total"] = df[subjects].sum(axis=1)
df["Percentage"] = df["Total"] / (len(subjects) * 100) * 100

def get_feedback(pct):
    if pct > 90:
        return "Excellent"
    elif pct >= 75:
        return "Good"
    elif pct >= 50:
        return "Average"
    else:
        return "Needs Improvement"

def get_grade(pct):
    if pct >= 95:
        return "A+"
    elif pct >= 90:
        return "A"
    elif pct >= 80:
        return "B+"
    elif pct >= 70:
        return "B"
    elif pct >= 60:
        return "C"
    elif pct >= 50:
        return "D"
    else:
        return "F"

df["Feedback"] = df["Percentage"].apply(get_feedback)
df["Grade"] = df["Percentage"].apply(get_grade)

# ---------- Class Average ----------
class_avg = df[subjects].mean().round(2)

# ---------- Top Performers ----------
top_performers = pd.concat([df[df[subject] == df[subject].max()][["Name", subject]] for subject in subjects], keys=subjects)

# ---------- Sidebar Filters ----------
selected_name = st.sidebar.selectbox("Select Student", ["All"] + df["Name"].tolist())

if selected_name == "All":
    selected_grade = st.sidebar.selectbox("Filter by Grade", ["None"] + sorted(df["Grade"].unique()))
else:
    st.sidebar.selectbox("Filter by Grade", ["None"], disabled=True, help="Available only when 'All' students are selected")
    selected_grade = "None"  # forcibly neutralize filter when individual student selected

# ---------- Apply Filters ----------
display_df = df.copy()
if selected_name != "All":
    display_df = display_df[display_df["Name"] == selected_name]
#if selected_subject != "None":
    #display_df = display_df.sort_values(by=selected_subject, ascending=False)
if selected_grade != "None":
    display_df = display_df[display_df["Grade"] == selected_grade]

# ---------- Display Data ----------
st.subheader("Report Summary")
st.dataframe(display_df.style.format({"Percentage": "{:.2f}"}), use_container_width=True)

# ---------- Plot Chart ----------
st.subheader("Interactive Marks Chart")
if selected_name != "All":
    student_row = df[df["Name"] == selected_name].iloc[0]
    fig = px.bar(x=subjects, y=student_row[subjects],
                 labels={'x': 'Subjects', 'y': 'Marks'},
                 title=f"{student_row['Name']}'s Subject-wise Marks",
                 height=400)
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)
else:
    st.info("Select a student from sidebar to view their interactive chart.")

# ---------- Class Average Comparison ----------
if selected_name != "All":
    st.subheader("Comparison with Class Average")
    student_row = df[df["Name"] == selected_name].iloc[0]
    comp_df = pd.DataFrame({
        "Subject": subjects,
        selected_name: student_row[subjects].values,
        "Class Average": class_avg.values
    })
    comp_df[selected_name] = pd.to_numeric(comp_df[selected_name], errors="coerce")
    comp_df["Class Average"] = pd.to_numeric(comp_df["Class Average"], errors="coerce")
    comp_df = comp_df.dropna()

    fig2 = px.bar(comp_df, x="Subject", y=[selected_name, "Class Average"],
                  barmode='group',
                  title=f"{selected_name} vs Class Average Comparison",
                  labels={"value": "Marks", "variable": "Legend"})
    st.plotly_chart(fig2)

# ---------- Top Performers ----------
st.sidebar.subheader("Top Scorers by Subject")
for subject in subjects:
    top_score = df[df[subject] == df[subject].max()]
    st.sidebar.markdown(f"**{subject}**: {', '.join(top_score['Name'])} ({top_score[subject].values[0]})")

# ---------- CSV Download ----------
def convert_df(download_df):
    return download_df.to_csv(index=False).encode("utf-8")

csv_output = convert_df(df)
st.sidebar.download_button("Download Report CSV", csv_output, "final_report_card.csv", "text/csv")

# ---------- PDF Export ----------
def generate_pdf(student_row):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph(f"Report Card: {student_row['Name']}", styles['Title']))
    flowables.append(Spacer(1, 12))

    data = [["Subject", "Marks"]] + [[subject, student_row[subject]] for subject in subjects]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    flowables.append(table)
    flowables.append(Spacer(1, 12))
    flowables.append(Paragraph(f"Total Marks: {student_row['Total']}", styles['Normal']))
    flowables.append(Paragraph(f"Percentage: {student_row['Percentage']:.2f}%", styles['Normal']))
    flowables.append(Paragraph(f"Grade: {student_row['Grade']}", styles['Normal']))
    flowables.append(Paragraph(f"Feedback: {student_row['Feedback']}", styles['Normal']))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

if selected_name != "All":
    student_row = df[df["Name"] == selected_name].iloc[0]
    pdf_buffer = generate_pdf(student_row)
    st.download_button(
        label="Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{student_row['Name']}_ReportCard.pdf",
        mime="application/pdf"
    )
