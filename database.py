import sqlite3


def create_database():

    connection = sqlite3.connect("student_database.db")

    cursor = connection.cursor()


    # Student performance table

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            study_hours REAL,
            attendance REAL,
            previous_score REAL,
            assignments_completed INTEGER,
            sleep_hours REAL,
            predicted_score REAL,
            risk_level TEXT
        )
    """)


    # Student engagement table

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            participation INTEGER,
            online_activity INTEGER,
            feedback TEXT,
            sentiment TEXT
        )
    """)


    connection.commit()

    connection.close()

    print("Student database created successfully!")


if __name__ == "__main__":

    create_database()