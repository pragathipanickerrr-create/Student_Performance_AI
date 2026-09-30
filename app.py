from flask import Flask, request, render_template_string, redirect, url_for, session, abort, flash
import sqlite3
import pandas as pd
import joblib
import os
import secrets
import csv
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


def csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)
    return session["_csrf_token"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        expected = session.get("_csrf_token")
        submitted = request.form.get("csrf_token")
        if not expected or not submitted or not secrets.compare_digest(expected, submitted):
            abort(400, "The form expired. Refresh the page and try again.")


def init_auth_database():
    """Create the account table without changing existing student records."""
    connection = sqlite3.connect("student_database.db")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('student', 'teacher', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS engagement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            participation INTEGER,
            online_activity INTEGER,
            feedback TEXT,
            sentiment TEXT
        )
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(engagement)")}
    if "created_at" not in columns:
        connection.execute("ALTER TABLE engagement ADD COLUMN created_at TEXT")
    connection.commit()
    connection.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def roles_required(*allowed_roles):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if session.get("user_role") not in allowed_roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


init_auth_database()

# ============================================================
# LOAD TRAINED AI MODEL
# ============================================================

model = joblib.load("student_performance_model.pkl")


# ============================================================
# SAVE STUDENT DATA
# ============================================================

def save_student(
    name,
    email,
    study_hours,
    attendance,
    previous_score,
    assignments_completed,
    sleep_hours,
    predicted_score,
    risk_level
):

    connection = sqlite3.connect("student_database.db")
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO students
        (
            name,
            email,
            study_hours,
            attendance,
            previous_score,
            assignments_completed,
            sleep_hours,
            predicted_score,
            risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        email,
        study_hours,
        attendance,
        previous_score,
        assignments_completed,
        sleep_hours,
        predicted_score,
        risk_level
    ))

    connection.commit()
    connection.close()


# ============================================================
# SAVE ENGAGEMENT DATA
# ============================================================

def save_engagement(
    student_id,
    participation,
    online_activity,
    feedback,
    sentiment
):

    connection = sqlite3.connect("student_database.db")
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO engagement
        (
            student_id,
            participation,
            online_activity,
            feedback,
            sentiment,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        participation,
        online_activity,
        feedback,
        sentiment,
        datetime.now().isoformat(timespec="minutes")
    ))

    connection.commit()
    connection.close()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(feedback):

    positive_words = [
        "good",
        "great",
        "happy",
        "interesting",
        "helpful",
        "excellent",
        "love",
        "enjoy"
    ]

    negative_words = [
        "bad",
        "boring",
        "difficult",
        "sad",
        "confusing",
        "poor",
        "hate",
        "hard"
    ]

    text = feedback.lower()

    positive_count = 0
    negative_count = 0

    for word in positive_words:

        if word in text:
            positive_count += 1

    for word in negative_words:

        if word in text:
            negative_count += 1

    if positive_count > negative_count:

        return "Positive"

    elif negative_count > positive_count:

        return "Negative"

    else:

        return "Neutral"


# ============================================================
# STUDENT SUPPORT CHATBOT
# ============================================================

def chatbot_response(question):

    question = question.lower()

    if "hello" in question or "hi" in question:

        return (
            "Hello! I am your Student Support Assistant. "
            "How can I help you?"
        )

    elif "attendance" in question:

        return (
            "Try to attend classes regularly and "
            "maintain good attendance."
        )

    elif "score" in question or "marks" in question:

        return (
            "To improve your score, increase your study time, "
            "complete assignments, and review weak topics."
        )

    elif "study" in question:

        return (
            "Create a simple daily study schedule and "
            "study consistently."
        )

    elif "assignment" in question:

        return (
            "Try to complete assignments on time and "
            "review them before submission."
        )

    elif "sleep" in question:

        return (
            "Maintain a regular sleep schedule so you can "
            "stay focused during your studies."
        )

    elif "risk" in question:

        return (
            "The system identifies students as High Risk, "
            "Medium Risk, or Low Risk based on the predicted score."
        )

    else:

        return (
            "I can help with attendance, marks, study habits, "
            "assignments, sleep, and student risk levels."
        )


# ============================================================
# GET STUDENTS
# ============================================================

def get_students(email=None):

    connection = sqlite3.connect("student_database.db")
    cursor = connection.cursor()

    query = """
        SELECT
            id,
            name,
            email,
            study_hours,
            attendance,
            previous_score,
            assignments_completed,
            sleep_hours,
            predicted_score,
            risk_level
        FROM students
    """
    params = ()
    if email:
        query += " WHERE email = ?"
        params = (email,)
    query += " ORDER BY id DESC"
    cursor.execute(query, params)

    students = cursor.fetchall()

    connection.close()

    return students


# ============================================================
# DASHBOARD DATA
# ============================================================

def get_dashboard_data(email=None):

    connection = sqlite3.connect("student_database.db")
    cursor = connection.cursor()

    where_clause = " WHERE email = ?" if email else ""
    params = (email,) if email else ()
    cursor.execute("SELECT COUNT(*) FROM students" + where_clause, params)

    total_students = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(predicted_score) FROM students" + where_clause, params)

    average_score = cursor.fetchone()[0]

    if average_score is None:

        average_score = 0

    else:

        average_score = round(
            average_score,
            1
        )

    risk_where = " AND risk_level = ?" if email else " WHERE risk_level = ?"
    cursor.execute("SELECT COUNT(*) FROM students" + where_clause + risk_where,
                   params + ("High Risk",))

    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students" + where_clause + risk_where,
                   params + ("Medium Risk",))

    medium_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students" + where_clause + risk_where,
                   params + ("Low Risk",))

    low_risk = cursor.fetchone()[0]

    connection.close()

    return (
        total_students,
        average_score,
        high_risk,
        medium_risk,
        low_risk
    )


def get_recent_high_risk_students(limit=5):
    connection = sqlite3.connect("student_database.db")
    students = connection.execute("""
        SELECT id, name, email, predicted_score
        FROM students
        WHERE risk_level = 'High Risk'
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    connection.close()
    return students


def get_engagement_analytics(email=None):
    connection = sqlite3.connect("student_database.db")
    query = """
        SELECT e.id, e.participation, e.online_activity, e.sentiment, e.created_at
        FROM engagement e
        JOIN students s ON s.id = e.student_id
    """
    params = ()
    if email:
        query += " WHERE lower(s.email) = lower(?)"
        params = (email,)
    query += " ORDER BY e.id DESC LIMIT 20"
    rows = connection.execute(query, params).fetchall()

    count_query = """
        SELECT e.sentiment, COUNT(*)
        FROM engagement e
        JOIN students s ON s.id = e.student_id
    """
    if email:
        count_query += " WHERE lower(s.email) = lower(?)"
    count_query += " GROUP BY e.sentiment"
    sentiment_rows = connection.execute(count_query, params).fetchall()
    connection.close()

    rows.reverse()
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for sentiment, count in sentiment_rows:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] = count

    labels = [
        row[4][:10] if row[4] else f"Earlier record {index + 1}"
        for index, row in enumerate(rows)
    ]
    return {
        "labels": labels,
        "participation": [row[1] for row in rows],
        "online_activity": [row[2] for row in rows],
        "sentiment_counts": [sentiment_counts["Positive"], sentiment_counts["Neutral"], sentiment_counts["Negative"]],
        "total": sum(sentiment_counts.values()),
        "positive": sentiment_counts["Positive"],
        "neutral": sentiment_counts["Neutral"],
        "negative": sentiment_counts["Negative"],
    }


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    study_hours,
    attendance,
    previous_score,
    assignments_completed,
    sleep_hours
):

    recommendations = []

    if study_hours < 2:

        recommendations.append(
            "Try increasing daily study time gradually."
        )

    if attendance < 75:

        recommendations.append(
            "Try to improve attendance and attend classes regularly."
        )

    if previous_score < 50:

        recommendations.append(
            "Review previous topics and focus on weak subjects."
        )

    if assignments_completed < 7:

        recommendations.append(
            "Complete pending assignments on time."
        )

    if sleep_hours < 6:

        recommendations.append(
            "Try to maintain a more consistent sleep schedule."
        )

    if not recommendations:

        recommendations.append(
            "Keep maintaining the current study habits."
        )

    return recommendations


# ============================================================
# AI FACTOR IMPORTANCE
# ============================================================

def get_feature_importance():

    features = [
        "Study Hours",
        "Attendance",
        "Previous Score",
        "Assignments Completed",
        "Sleep Hours"
    ]

    importance = model.feature_importances_

    factors = list(
        zip(
            features,
            importance
        )
    )

    factors.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return factors


def get_model_evaluation():
    """Evaluate the saved regressor on the same held-out split used for training."""
    feature_names = [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed",
        "sleep_hours",
    ]
    try:
        data = pd.read_csv("student_data.csv")
        if not set(feature_names + ["final_score"]).issubset(data.columns):
            return {"available": False, "message": "The training data is missing required columns."}
        data = data.dropna(subset=feature_names + ["final_score"])
        if len(data) < 5:
            return {"available": False, "message": "At least five complete data rows are needed to make a held-out evaluation."}
        inputs = data[feature_names]
        target = data["final_score"]
        _, test_inputs, _, test_target = train_test_split(
            inputs, target, test_size=0.2, random_state=42
        )
        predictions = model.predict(test_inputs)
        mae = float(mean_absolute_error(test_target, predictions))
        r_squared = float(r2_score(test_target, predictions)) if len(test_target) > 1 else None
        return {
            "available": True,
            "mae": round(mae, 2),
            "r_squared": round(r_squared, 3) if r_squared is not None else None,
            "test_count": int(len(test_target)),
            "data_count": int(len(data)),
        }
    except (OSError, ValueError, KeyError) as error:
        return {"available": False, "message": f"Evaluation could not be calculated: {error}"}


# ============================================================
# WEBSITE PAGE
# ============================================================

PAGE = """

<!doctype html>

<html>

<head>

    <title>Student Performance AI</title>

    <link
        rel="stylesheet"
        href="{{ url_for('static', filename='style.css') }}"
    >

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

</head>


<body>

<nav class="site-nav" aria-label="Main navigation">
    <a href="#prediction">Prediction</a>
    <a href="#engagement">Engagement</a>
    <a href="#engagement-trends">Engagement trends</a>
    <a href="#chatbot">Support chatbot</a>
    <a href="#dashboard">Dashboard</a>
    <a href="#records">Student records</a>
    <a href="#model-evaluation">Model evaluation</a>
    <a href="{{ url_for('reports') }}">Reports</a>
    {% if current_role == "admin" %}<a href="#accounts">Accounts</a>{% endif %}
    <span class="nav-user">{{ current_name }} · {{ current_role|title }}</span>
    <form method="post" action="{{ url_for('logout') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><button type="submit" class="nav-logout">Sign out</button></form>
</nav>

{% with messages = get_flashed_messages() %}{% if messages %}<div class="site-notices">{% for notice in messages %}<p>{{ notice }}</p>{% endfor %}</div>{% endif %}{% endwith %}


<!-- ========================================================
     MAIN TITLE
========================================================= -->

<h1>Student Performance AI</h1>

<h2>Student Performance Prediction System</h2>

<p>
    AI-powered student performance analysis
    and early warning system.
</p>


<hr>


<!-- ========================================================
     PERFORMANCE FORM
========================================================= -->

{% if current_role != "student" %}
<h2 id="prediction">Add Student Performance Data</h2>


<form method="post">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">

    <input
        type="hidden"
        name="performance_form"
        value="yes"
    >


    <label>Student Name:</label>

    <input
        type="text"
        name="name"
        required
    >

    <br><br>


    <label>Student Email:</label>

    <input
        type="email"
        name="email"
        required
    >

    <br><br>


    <label>Study hours per day:</label>

    <input
        type="number"
        name="study_hours"
        min="0"
        max="24"
        step="0.5"
        required
    >

    <br><br>


    <label>Attendance:</label>

    <input
        type="number"
        name="attendance"
        min="0"
        max="100"
        required
    >

    <br><br>


    <label>Previous score:</label>

    <input
        type="number"
        name="previous_score"
        min="0"
        max="100"
        required
    >

    <br><br>


    <label>Assignments completed:</label>

    <input
        type="number"
        name="assignments_completed"
        min="0"
        max="10"
        required
    >

    <br><br>


    <label>Sleep hours:</label>

    <input
        type="number"
        name="sleep_hours"
        min="0"
        max="24"
        step="0.5"
        required
    >

    <br><br>


    <button type="submit">
        Predict Performance
    </button>

</form>
{% endif %}


<!-- ========================================================
     PREDICTION RESULT
========================================================= -->

{% if prediction is not none %}

<hr>

<h2>Prediction Result</h2>


<div class="prediction-box">

    <h3>
        Student:
        {{ student_name }}
    </h3>


    <div class="score-display">

        <span class="score-label">
            Predicted Score
        </span>

        <span class="score-number">
            {{ prediction }}
        </span>

        <span class="score-total">
            / 100
        </span>

    </div>


    {% if risk_level == "High Risk" %}

        <div class="risk-box high-risk">

            <div class="risk-icon">
                
            </div>

            <div>

                <div class="risk-title">
                    HIGH RISK
                </div>

                <div class="risk-description">
                    This student may need extra academic support.
                </div>

            </div>

        </div>


    {% elif risk_level == "Medium Risk" %}

        <div class="risk-box medium-risk">

            <div class="risk-icon">
                
            </div>

            <div>

                <div class="risk-title">
                    MEDIUM RISK
                </div>

                <div class="risk-description">
                    This student is making progress.
                    Continued support is recommended.
                </div>

            </div>

        </div>


    {% else %}

        <div class="risk-box low-risk">

            <div class="risk-icon">
                
            </div>

            <div>

                <div class="risk-title">
                    LOW RISK
                </div>

                <div class="risk-description">
                    This student is performing well.
                    Keep up the good work!
                </div>

            </div>

        </div>

    {% endif %}


    <p>
        {{ message }}
    </p>


    <!-- RECOMMENDATIONS -->

    <h3>
        Personalized Recommendations
    </h3>


    <ul>

        {% for recommendation in recommendations %}

        <li>
            {{ recommendation }}
        </li>

        {% endfor %}

    </ul>


    <!-- STUDENT FACTORS -->

    <h3>
        Student Factors
    </h3>


    <ul>

        <li>
            Study Hours:
            {{ study_hours }}
        </li>

        <li>
            Attendance:
            {{ attendance }}%
        </li>

        <li>
            Previous Score:
            {{ previous_score }}
        </li>

        <li>
            Assignments Completed:
            {{ assignments_completed }}/10
        </li>

        <li>
            Sleep Hours:
            {{ sleep_hours }}
        </li>

    </ul>

</div>

{% endif %}


<!-- ========================================================
     ENGAGEMENT ANALYSIS
========================================================= -->

<hr>

{% if current_role != "student" %}<h2 id="engagement">Student Engagement Analysis</h2>{% endif %}

<p>
    Analyze participation, online activity
    and student feedback.
</p>


<form method="post">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">

    <input
        type="hidden"
        name="engagement_form"
        value="yes"
    >


    <label>Student ID:</label>

    <input
        type="number"
        name="student_id"
        min="1"
        value="{{ current_student_id if current_role == 'student' else '' }}"
        {% if current_role == "student" %}readonly{% endif %}
        required
    >

    <br><br>


    <label>Participation Level:</label>

    <input
        type="number"
        name="participation"
        min="0"
        max="100"
        required
    >

    <br><br>


    <label>Online Activity:</label>

    <input
        type="number"
        name="online_activity"
        min="0"
        max="100"
        required
    >

    <br><br>


    <label>Student Feedback:</label>

    <input
        type="text"
        name="feedback"
        required
    >

    <br><br>


    <button type="submit">
        Analyze Engagement
    </button>

</form>


{% if sentiment %}

<div class="engagement-result">

    <h3>
        Engagement Result
    </h3>

    <p>
        <strong>Participation:</strong>
        {{ engagement_participation }}%
    </p>

    <p>
        <strong>Online Activity:</strong>
        {{ engagement_online_activity }}%
    </p>

    <p>
        <strong>Feedback:</strong>
        {{ engagement_feedback }}
    </p>

    <p>
        <strong>Feedback Sentiment:</strong>
        {{ sentiment }}
    </p>

</div>

{% endif %}


<hr>
<h2 id="engagement-trends">Student Engagement Trends</h2>
<p>Review recent participation and online activity scores, together with feedback sentiment.</p>
<div class="charts engagement-trend-grid">
    <div class="chart-box engagement-chart-box">
        <h3>Participation and Online Activity</h3>
        {% if engagement_analytics.total %}<div class="engagement-trend-container"><canvas id="engagementTrendChart"></canvas></div>
        {% else %}<p class="report-empty">Engagement records will appear here after an analysis is saved.</p>{% endif %}
    </div>
    <div class="chart-box engagement-chart-box">
        <h3>Feedback Sentiment</h3>
        {% if engagement_analytics.total %}<div class="sentiment-container"><canvas id="sentimentChart"></canvas></div>
        {% else %}<p class="report-empty">No feedback sentiment records are available yet.</p>{% endif %}
        <div class="sentiment-counts"><span>Positive <strong>{{ engagement_analytics.positive }}</strong></span><span>Neutral <strong>{{ engagement_analytics.neutral }}</strong></span><span>Negative <strong>{{ engagement_analytics.negative }}</strong></span></div>
    </div>
</div>


<!-- ========================================================
     CHATBOT
========================================================= -->

<hr>


<h2 id="chatbot">
    Student Support Chatbot
</h2>


<p>
    Ask the chatbot questions about your studies,
    attendance, assignments, marks, sleep, or risk level.
</p>


<form method="post" action="/#chatbot">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">

    <input
        type="hidden"
        name="chatbot_form"
        value="yes"
    >


    <label>Your Question:</label>

    <input
        type="text"
        name="question"
        placeholder="Ask your question..."
        required
    >

    <br><br>


    <button type="submit">
        Ask Chatbot
    </button>

</form>


{% if chatbot_answer %}

<div class="chatbot-answer">

    <h3>
        Chatbot Response
    </h3>

    <p>
        {{ chatbot_answer }}
    </p>

</div>

{% endif %}


<!-- ========================================================
     DASHBOARD
========================================================= -->

{% if current_role == "admin" %}
<hr>
<h2 id="accounts">Account Management</h2>
<p>Create student and teacher accounts. Student accounts can only view records tied to their email address.</p>
<form method="post" action="{{ url_for('create_account') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <label>Full name:</label><input type="text" name="name" required><br><br>
    <label>Email address:</label><input type="email" name="email" required><br><br>
    <label>Temporary password:</label><input type="password" name="password" minlength="10" required><br><br>
    <label>Account type:</label><select name="role" required><option value="student">Student</option><option value="teacher">Teacher</option></select><br><br>
    <button type="submit">Create account</button>
</form>
{% endif %}

<hr>

<h2 id="dashboard">Performance Dashboard</h2>

{% if current_role in ("admin", "teacher") %}
<section class="early-warning" aria-labelledby="early-warning-title">
    <div class="early-warning-heading">
        <div><p class="alert-eyebrow">STAFF FOLLOW-UP</p><h3 id="early-warning-title">Early Warning Alerts</h3></div>
        <span class="alert-count">{{ high_risk_alerts|length }} recent</span>
    </div>
    {% if high_risk_alerts %}
    <p>These students are currently classified as high risk. Review their records and consider appropriate academic support.</p>
    <div class="alert-list">
        {% for alert in high_risk_alerts %}
        <article class="alert-item">
            <div><strong>{{ alert[1] }}</strong><span>Student ID {{ alert[0] }} · {{ alert[2] }}</span></div>
            <div class="alert-score"><strong>{{ alert[3] }}</strong><span>Predicted score</span></div>
        </article>
        {% endfor %}
    </div>
    {% else %}
    <p class="alert-clear">No high-risk student records are currently listed.</p>
    {% endif %}
</section>
{% endif %}


<div class="dashboard-cards">


    <div class="card">

        <h3>
            Total Students
        </h3>

        <p>
            {{ total_students }}
        </p>

    </div>


    <div class="card">

        <h3>
            Average Score
        </h3>

        <p>
            {{ average_score }}
        </p>

    </div>


    <div class="card">

        <h3>
             High Risk
        </h3>

        <p>
            {{ high_risk }}
        </p>

    </div>


    <div class="card">

        <h3>
             Medium Risk
        </h3>

        <p>
            {{ medium_risk }}
        </p>

    </div>


    <div class="card">

        <h3>
             Low Risk
        </h3>

        <p>
            {{ low_risk }}
        </p>

    </div>


</div>


<!-- ========================================================
     GRAPHICAL ANALYSIS
========================================================= -->

<h2>Graphical Analysis</h2>


<div class="charts">


    <div class="chart-box pie-box">

        <h3>
            Risk Level Distribution
        </h3>


        <div class="pie-container">

            <canvas id="riskChart"></canvas>

        </div>

    </div>


    <div class="chart-box factor-box">

        <h3>
            AI Factor Importance
        </h3>


        <div class="factor-container">

            <canvas id="factorChart"></canvas>

        </div>

    </div>


</div>


<!-- ========================================================
     STUDENT RECORDS
========================================================= -->

<hr>

<h2 id="records">Student Records</h2>


{% if students %}

<table>

    <tr>

        <th>ID</th>

        <th>Name</th>

        <th>Email</th>

        <th>Attendance</th>

        <th>Previous Score</th>

        <th>Predicted Score</th>

        <th>Risk Level</th>

    </tr>


    {% for student in students %}

    <tr>

        <td>
            {{ student[0] }}
        </td>

        <td>
            {{ student[1] }}
        </td>

        <td>
            {{ student[2] }}
        </td>

        <td>
            {{ student[4] }}%
        </td>

        <td>
            {{ student[5] }}
        </td>

        <td>
            {{ student[8] }}
        </td>

        <td>
            {{ student[9] }}
        </td>

    </tr>

    {% endfor %}

</table>


{% else %}

<p>
    No student records available yet.
</p>

{% endif %}


<!-- ========================================================
     AI FACTOR ANALYSIS
========================================================= -->

<hr>

<h2>AI Factor Analysis</h2>


<p>
    The trained Random Forest model identifies
    the relative importance of the input features
    used for prediction.
</p>


<table>

    <tr>

        <th>
            Factor
        </th>

        <th>
            Importance
        </th>

    </tr>


    {% for factor, importance in factors %}

    <tr>

        <td>
            {{ factor }}
        </td>

        <td>
            {{ "%.2f"|format(importance * 100) }}%
        </td>

    </tr>

    {% endfor %}

</table>


<hr>
<h2 id="model-evaluation">Model Evaluation</h2>
<p>Evaluation uses the held-out test split from the project training dataset.</p>
{% if model_evaluation.available %}
<div class="model-evaluation-cards">
    <article><span>Mean absolute error</span><strong>{{ model_evaluation.mae }} points</strong><small>Average distance between predicted and actual scores. Lower is better.</small></article>
    <article><span>Held-out records</span><strong>{{ model_evaluation.test_count }}</strong><small>Records used to evaluate this model.</small></article>
    <article><span>Training dataset</span><strong>{{ model_evaluation.data_count }} records</strong><small>Complete rows in student_data.csv.</small></article>
    <article><span>R² score</span><strong>{{ model_evaluation.r_squared if model_evaluation.r_squared is not none else "Not available" }}</strong><small>How much score variation the model explains; it can be negative.</small></article>
</div>
<p class="evaluation-note">These metrics describe this dataset's held-out split, not guaranteed performance for every student or institution.</p>
{% else %}
<div class="model-evaluation-unavailable"><strong>Evaluation is not available yet.</strong><p>{{ model_evaluation.message }}</p></div>
{% endif %}


<!-- ========================================================
     CHART JAVASCRIPT
========================================================= -->

<script>


const riskChart = new Chart(

    document.getElementById("riskChart"),

    {

        type: "doughnut",

        data: {

            labels: [

                "High Risk",
                "Medium Risk",
                "Low Risk"

            ],

            datasets: [{

                backgroundColor: [
                    "#c85a52",
                    "#d8a33e",
                    "#4c9a70"
                ],
                borderColor: "#ffffff",
                borderWidth: 3,

                data: [

                    {{ high_risk }},
                    {{ medium_risk }},
                    {{ low_risk }}

                ]

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {

                    position: "bottom"

                }

            }

        }

    }

);


const factorChart = new Chart(

    document.getElementById("factorChart"),

    {

        type: "bar",

        data: {

            labels: [

                {% for factor, importance in factors %}

                    "{{ factor }}",

                {% endfor %}

            ],

            datasets: [{

                label: "Importance (%)",

                backgroundColor: [
                    "#244f8f",
                    "#3869aa",
                    "#5682ba",
                    "#789bc9",
                    "#9cb5d9"
                ],
                borderRadius: 7,
                borderSkipped: false,

                data: [

                    {% for factor, importance in factors %}

                        {{ importance * 100 }},

                    {% endfor %}

                ]

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    }

);

{% if engagement_analytics.total %}
const engagementData = {{ engagement_analytics|tojson }};
new Chart(document.getElementById("engagementTrendChart"), {
    type: "line",
    data: {
        labels: engagementData.labels,
        datasets: [
            {label: "Participation (%)", data: engagementData.participation, borderColor: "#3869aa", backgroundColor: "rgba(56,105,170,.12)", fill: true, tension: .3},
            {label: "Online activity (%)", data: engagementData.online_activity, borderColor: "#4c9a70", backgroundColor: "rgba(76,154,112,.1)", fill: true, tension: .3}
        ]
    },
    options: {responsive: true, maintainAspectRatio: false, scales: {y: {beginAtZero: true, max: 100}}, plugins: {legend: {position: "bottom"}}}
});

new Chart(document.getElementById("sentimentChart"), {
    type: "doughnut",
    data: {labels: ["Positive", "Neutral", "Negative"], datasets: [{data: engagementData.sentiment_counts, backgroundColor: ["#4c9a70", "#8a9bb1", "#c85a52"], borderColor: "#fff", borderWidth: 3}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: "bottom"}}}
});
{% endif %}


</script>


</body>

</html>

"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    if not session.get("user_id"):
        return redirect(url_for("login"))

    current_email = session.get("user_email")
    current_role = session.get("user_role")

    prediction = None

    message = ""

    risk_level = ""

    student_name = ""

    recommendations = []

    study_hours = ""

    attendance = ""

    previous_score = ""

    assignments_completed = ""

    sleep_hours = ""


    # Engagement

    sentiment = ""

    engagement_participation = ""

    engagement_online_activity = ""

    engagement_feedback = ""


    # Chatbot

    chatbot_answer = ""


    # ========================================================
    # HANDLE POST REQUEST
    # ========================================================

    if request.method == "POST":


        # ====================================================
        # PERFORMANCE FORM
        # ====================================================

        if "performance_form" in request.form:

            if current_role == "student":
                abort(403)

            name = request.form["name"]

            email = request.form["email"]

            student_name = name


            study_hours = float(
                request.form["study_hours"]
            )

            attendance = float(
                request.form["attendance"]
            )

            previous_score = float(
                request.form["previous_score"]
            )

            assignments_completed = float(
                request.form["assignments_completed"]
            )

            sleep_hours = float(
                request.form["sleep_hours"]
            )


            # Create input for AI model

            student = pd.DataFrame([{

                "study_hours": study_hours,

                "attendance": attendance,

                "previous_score": previous_score,

                "assignments_completed":
                    assignments_completed,

                "sleep_hours": sleep_hours

            }])


            # AI prediction

            prediction = round(

                float(
                    model.predict(student)[0]
                ),

                1

            )


            # =================================================
            # RISK CLASSIFICATION
            # =================================================

            if prediction < 50:

                message = (
                    "This student may need extra support."
                )

                risk_level = "High Risk"


            elif prediction < 75:

                message = (
                    "This student is making progress. "
                    "Keep encouraging them."
                )

                risk_level = "Medium Risk"


            else:

                message = (
                    "This student is performing well. "
                    "Keep it up!"
                )

                risk_level = "Low Risk"


            # =================================================
            # PERSONALIZED RECOMMENDATIONS
            # =================================================

            recommendations = generate_recommendations(

                study_hours,

                attendance,

                previous_score,

                assignments_completed,

                sleep_hours

            )


            # =================================================
            # SAVE STUDENT
            # =================================================

            save_student(

                name,

                email,

                study_hours,

                attendance,

                previous_score,

                assignments_completed,

                sleep_hours,

                prediction,

                risk_level

            )


        # ====================================================
        # ENGAGEMENT FORM
        # ====================================================

        elif "engagement_form" in request.form:

            student_id = int(
                request.form["student_id"]
            )

            if current_role == "student":
                connection = sqlite3.connect("student_database.db")
                owns_record = connection.execute(
                    "SELECT 1 FROM students WHERE id = ? AND email = ?",
                    (student_id, current_email)
                ).fetchone()
                connection.close()
                if not owns_record:
                    abort(403)


            engagement_participation = float(
                request.form["participation"]
            )


            engagement_online_activity = float(
                request.form["online_activity"]
            )


            engagement_feedback = request.form[
                "feedback"
            ]


            # Sentiment analysis

            sentiment = analyze_sentiment(

                engagement_feedback

            )


            # Save engagement

            save_engagement(

                student_id,

                engagement_participation,

                engagement_online_activity,

                engagement_feedback,

                sentiment

            )


        # ====================================================
        # CHATBOT FORM
        # ====================================================

        elif "chatbot_form" in request.form:

            question = request.form["question"]


            chatbot_answer = chatbot_response(

                question

            )


    # ========================================================
    # GET DATA FOR DASHBOARD
    # ========================================================

    students = get_students(current_email if current_role == "student" else None)


    (
        total_students,
        average_score,
        high_risk,
        medium_risk,
        low_risk

    ) = get_dashboard_data(current_email if current_role == "student" else None)


    # ========================================================
    # GET AI FACTORS
    # ========================================================

    factors = get_feature_importance()

    high_risk_alerts = (
        get_recent_high_risk_students()
        if current_role in ("admin", "teacher") else []
    )
    engagement_analytics = get_engagement_analytics(
        current_email if current_role == "student" else None
    )
    model_evaluation = get_model_evaluation()


    # ========================================================
    # RENDER PAGE
    # ========================================================

    return render_template_string(

        PAGE,

        prediction=prediction,

        message=message,

        risk_level=risk_level,

        student_name=student_name,

        recommendations=recommendations,

        study_hours=study_hours,

        attendance=attendance,

        previous_score=previous_score,

        assignments_completed=
            assignments_completed,

        sleep_hours=sleep_hours,

        sentiment=sentiment,

        engagement_participation=
            engagement_participation,

        engagement_online_activity=
            engagement_online_activity,

        engagement_feedback=
            engagement_feedback,

        chatbot_answer=chatbot_answer,

        students=students,

        total_students=total_students,

        average_score=average_score,

        high_risk=high_risk,

        medium_risk=medium_risk,

        low_risk=low_risk,

        factors=factors,
        current_role=current_role,
        current_name=session.get("user_name"),
        current_student_id=(students[0][0] if current_role == "student" and students else ""),
        high_risk_alerts=high_risk_alerts,
        engagement_analytics=engagement_analytics,
        model_evaluation=model_evaluation

    )


LOGIN_PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ page_title }} | Student Performance AI</title><link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}"></head>
<body class="auth-page"><main class="auth-card"><p class="auth-kicker">STUDENT PERFORMANCE AI</p><h1>{{ page_title }}</h1>
<p class="auth-intro">{{ intro }}</p>
{% with messages = get_flashed_messages() %}{% if messages %}<div class="auth-message">{% for message in messages %}<p>{{ message }}</p>{% endfor %}</div>{% endif %}{% endwith %}
<form method="post" class="auth-form"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
{% if setup %}<label>Administrator name<input name="name" required autocomplete="name"></label>{% endif %}
<label>Email address<input name="email" type="email" required autocomplete="username"></label>
<label>Password<input name="password" type="password" required autocomplete="{{ 'new-password' if setup else 'current-password' }}"></label>
{% if setup %}<label>Confirm password<input name="confirm_password" type="password" required autocomplete="new-password"></label>{% endif %}
<button type="submit">{{ 'Create administrator account' if setup else 'Sign in' }}</button></form>
{% if not setup %}<p class="auth-footnote">Accounts are provided by your project administrator.</p>{% endif %}</main></body></html>
"""


@app.route("/setup", methods=["GET", "POST"])
def setup():
    connection = sqlite3.connect("student_database.db")
    account_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    connection.close()
    if account_count:
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 10:
            flash("Enter your name and email, and use a password with at least 10 characters.")
        elif password != request.form.get("confirm_password"):
            flash("The passwords do not match.")
        else:
            connection = sqlite3.connect("student_database.db")
            connection.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, 'admin')",
                (name, email, generate_password_hash(password))
            )
            connection.commit()
            connection.close()
            flash("Administrator account created. Please sign in.")
            return redirect(url_for("login"))
    return render_template_string(LOGIN_PAGE, page_title="Set up your site", intro="Create the first administrator account to secure this site.", setup=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    connection = sqlite3.connect("student_database.db")
    account_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    connection.close()
    if account_count == 0:
        return redirect(url_for("setup"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        connection = sqlite3.connect("student_database.db")
        account = connection.execute(
            "SELECT id, name, email, password_hash, role FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        connection.close()
        if account and check_password_hash(account[3], password):
            session.clear()
            session["user_id"] = account[0]
            session["user_name"] = account[1]
            session["user_email"] = account[2]
            session["user_role"] = account[4]
            return redirect(url_for("home"))
        flash("Email or password was not recognized.")
    return render_template_string(LOGIN_PAGE, page_title="Sign in", intro="Sign in to continue to the student performance dashboard.", setup=False)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/accounts", methods=["POST"])
@roles_required("admin")
def create_account():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role = request.form.get("role", "")
    if not name or not email or len(password) < 10 or role not in ("student", "teacher"):
        flash("Enter all account details and use a password with at least 10 characters.")
        return redirect(url_for("home") + "#accounts")
    connection = sqlite3.connect("student_database.db")
    if role == "student":
        match = connection.execute("SELECT 1 FROM students WHERE lower(email) = ? LIMIT 1", (email,)).fetchone()
        if not match:
            connection.close()
            flash("Add a student record with this email before creating its student account.")
            return redirect(url_for("home") + "#accounts")
    try:
        connection.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (name, email, generate_password_hash(password), role)
        )
        connection.commit()
        flash(f"{role.title()} account created.")
    except sqlite3.IntegrityError:
        flash("An account already uses that email address.")
    finally:
        connection.close()
    return redirect(url_for("home") + "#accounts")


REPORT_PAGE = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Student Performance Report</title><link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}"></head>
<body class="report-page"><header class="report-heading"><p class="auth-kicker">STUDENT PERFORMANCE AI</p>
<h1>Student Performance Report</h1><p>Generated {{ generated_at }} · Prepared for {{ scope_label }}</p></header>
<nav class="report-actions"><a href="{{ url_for('home') }}">Return to dashboard</a><a href="{{ url_for('download_report') }}">Download CSV</a><button type="button" onclick="window.print()">Print or save as PDF</button></nav>
<section class="report-metrics" aria-label="Summary statistics">
<article><span>Students included</span><strong>{{ total_students }}</strong></article>
<article><span>Average predicted score</span><strong>{{ average_score }} / 100</strong></article>
<article class="report-high"><span>High risk</span><strong>{{ high_risk }}</strong></article>
<article class="report-medium"><span>Medium risk</span><strong>{{ medium_risk }}</strong></article>
<article class="report-low"><span>Low risk</span><strong>{{ low_risk }}</strong></article>
</section>
<h2>Student records</h2>
{% if students %}<div class="report-table-wrap"><table><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Study hours</th><th>Attendance</th><th>Previous score</th><th>Assignments</th><th>Sleep hours</th><th>Predicted score</th><th>Risk</th></tr></thead><tbody>
{% for student in students %}<tr><td>{{ student[0] }}</td><td>{{ student[1] }}</td><td>{{ student[2] }}</td><td>{{ student[3] }}</td><td>{{ student[4] }}%</td><td>{{ student[5] }}</td><td>{{ student[6] }}/10</td><td>{{ student[7] }}</td><td>{{ student[8] }}</td><td>{{ student[9] }}</td></tr>{% endfor %}
</tbody></table></div>{% else %}<p class="report-empty">No student records are available yet.</p>{% endif %}
<footer class="report-footnote">Scores are model predictions for academic support and should be reviewed alongside educator judgment.</footer>
</body></html>
"""


def report_rows_for_current_user():
    if session.get("user_role") == "student":
        return get_students(session.get("user_email"))
    return get_students()


@app.route("/reports")
@login_required
def reports():
    students = report_rows_for_current_user()
    email_scope = session.get("user_email") if session.get("user_role") == "student" else None
    total, average, high, medium, low = get_dashboard_data(email_scope)
    scope_label = "My records" if email_scope else "All student records"
    return render_template_string(
        REPORT_PAGE,
        students=students,
        total_students=total,
        average_score=average,
        high_risk=high,
        medium_risk=medium,
        low_risk=low,
        generated_at=datetime.now().strftime("%d %b %Y, %I:%M %p"),
        scope_label=scope_label,
    )


@app.route("/reports.csv")
@login_required
def download_report():
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "Student ID", "Name", "Email", "Study hours per day", "Attendance (%)",
        "Previous score", "Assignments completed", "Sleep hours", "Predicted score", "Risk level"
    ])
    for student in report_rows_for_current_user():
        cells = list(student)
        for index in (1, 2, 9):
            value = str(cells[index])
            if value.startswith(("=", "+", "-", "@")):
                cells[index] = "'" + value
        writer.writerow(cells)
    response = app.response_class(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=student-performance-report-{datetime.now():%Y%m%d}.csv"},
    )
    response.headers["Cache-Control"] = "no-store"
    return response


# ============================================================
# START FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)
