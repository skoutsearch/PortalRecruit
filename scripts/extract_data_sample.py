import csv
import os
import sqlite3

DB_PATH = os.path.join(os.getcwd(), "data/skout.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # try to get a mix of positions
    pos_groups = ["G", "F", "C", "F/C", "G/F"]
    rows = []
    for pos in pos_groups:
        cur.execute(
            "SELECT position, height_in, weight_lb, full_name, team_id FROM players WHERE position = ? LIMIT 2",
            (pos,),
        )
        rows.extend(cur.fetchall())
        if len(rows) >= 10:
            break

    # fallback if not enough
    if len(rows) < 10:
        cur.execute("SELECT position, height_in, weight_lb, full_name, team_id FROM players LIMIT 10")
        rows = cur.fetchall()

    conn.close()

    writer = csv.writer(os.sys.stdout)
    writer.writerow(["true_position", "height_in", "weight_lb", "text"])
    for pos, h, w, name, team in rows[:10]:
        text = f"{name} ({team})" if team else name
        writer.writerow([pos, h, w, text])


if __name__ == "__main__":
    main()
