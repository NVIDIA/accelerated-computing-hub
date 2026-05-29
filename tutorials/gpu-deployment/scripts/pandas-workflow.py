import pandas as pd
import time
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "nyc_parking_violations_2022.parquet"
)

df = pd.read_parquet(
    DATA_PATH,
    columns=[
        "Registration State",
        "Violation Code",
        "Vehicle Body Type",
        "Vehicle Make",
        "Violation Time",
        "Violation County",
        "Vehicle Year",
        "Violation Description",
        "Issue Date",
        "Summons Number",
    ],
)


start = time.time()
result_1 = (
    df[["Registration State", "Violation Description"]]  # get only these two columns
    .value_counts()  # get the count of violations per state and per type of offence
    .groupby("Registration State")  # group by state
    .head(1)  # get the first row in each group (the type of violation with the largest count)
    .sort_index()  # sort by state name
    .reset_index()
)
end = time.time()
print(f"Operation 1 (value_counts + groupby + head): {end - start:.4f} seconds")


start = time.time()
result_2 = (
    df.groupby(["Vehicle Body Type"])
    .agg({"Summons Number": "count"})
    .rename(columns={"Summons Number": "Count"})
    .sort_values(["Count"], ascending=False)
)
end = time.time()
print(f"Operation 2 (groupby + agg + sort): {end - start:.4f} seconds")


weekday_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

start = time.time()
df["Issue Date"] = df["Issue Date"].astype("datetime64[ms]")
df["issue_weekday"] = df["Issue Date"].dt.weekday.map(weekday_names)
result_3 = (
    df.groupby(["issue_weekday"])["Summons Number"]
    .count()
    .sort_values(ascending=False)
)
end = time.time()
print(f"Operation 3 (weekday violation counts): {end - start:.4f} seconds")


start = time.time()
result_5 = (
    df.groupby("Violation County").size().sort_values(ascending=False).head(10)
)
end = time.time()
print(f"Operation 5 (groupby county + size + head): {end - start:.4f} seconds")


start = time.time()
df.count(axis=0)
end = time.time()
print(f"Operation 6 (df.count axis=0): {end - start:.4f} seconds")

start = time.time()
df.count(axis=1)
end = time.time()
print(f"Operation 7 (df.count axis=1): {end - start:.4f} seconds")
