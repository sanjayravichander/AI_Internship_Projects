## 📘 Student Report Card Generator
This project is an interactive web application built with Streamlit that allows you to generate and visualize student report cards. You can upload your own student mark data or use the built-in sample dataset to quickly get started. The application calculates total marks, percentages, assigns grades and feedback, and provides insightful visualizations.

## ✨ Key Features
File Upload: Easily upload your student marks in a CSV format.

## Sample Data: A built-in sample dataset is available for immediate use if no file is uploaded.

Automated Calculations: Automatically calculates Total Marks and Percentage for each student.

Grading & Feedback: Assigns grades (A+, A, B+, etc.) and provides qualitative feedback (Excellent, Good, Average, Needs Improvement) based on student performance.

Interactive Data Table: View a summary of all student reports with filtering options by student name or grade.

Individual Student Performance Chart: Visualize a selected student's marks across different subjects using an interactive bar chart.

Class Average Comparison: Compare a selected student's performance against the class average for each subject.

Top Scorers by Subject: Quickly identify the highest-scoring students for each individual subject.

Data Export: Download the complete report card data as a CSV file.

PDF Report Generation: Generate and download individual student report cards as a PDF document.

## ⚙️ How It Works
The application processes student mark data through the following steps:

Data Ingestion: Reads student marks from a CSV file (either uploaded by the user or the built-in sample).

## Data Transformation:

Calculates the sum of marks for all subjects to get the Total.

Computes the Percentage based on the total marks and the maximum possible marks (assuming 100 per subject).

Applies custom logic to assign Feedback (e.g., "Excellent" for >90%) and Grade (e.g., "A+" for >=95%).

Analysis: Determines the class average for each subject and identifies top performers.

Visualization: Utilizes Plotly Express to create interactive charts for individual student performance and comparison with class averages.

PDF Generation: Employs ReportLab to dynamically create PDF documents for individual student report cards, summarizing their marks, percentage, grade, and feedback.

Streamlit Interface: All functionalities are wrapped within a Streamlit application, providing a user-friendly web interface for interaction.

## Getting Started
Prerequisites
Make sure you have Python 3.7 or higher installed on your system.

## Installation
Clone the repository (or download the script directly):

```bash

git clone https://github.com/your-username/student-report-card-generator.git
cd student-report-card-generator
```

(Note: Replace your-username/student-report-card-generator with your actual repository path if you're hosting it.)


## Install the required Python packages:
You can install all necessary libraries using pip:

```bash

pip install streamlit pandas matplotlib plotly reportlab
```

## Running the Application
Save the Python script: Save the provided Python code as, for example, report_card_app.py.

Run the Streamlit app: Open your terminal or command prompt, navigate to the directory where you saved the script, and run:

```bash

streamlit run report_card_app.py
```

Access the Dashboard: Your web browser will automatically open a new tab displaying the Streamlit application (usually at http://localhost:8501).

## Data Format
The application expects a CSV file with the following structure:

The first column must be Name (for student names).

Subsequent columns should be the subject names (e.g., Math, Science, English), containing numerical marks for each student.

## Example student_marks.csv:

## Code snippet

Name,Math,Science,English,History,Computer
Amit,78,88,90,76,95
Sara,92,81,85,89,91
John,65,72,70,68,60

## 📚 Libraries Used
streamlit: For building the interactive web application.

pandas: For data manipulation and analysis.

matplotlib.pyplot: For basic plotting (though Plotly is used for interactive charts).

plotly.express: For creating interactive and visually appealing charts.

reportlab: For generating PDF documents.

io: For handling in-memory file operations.

