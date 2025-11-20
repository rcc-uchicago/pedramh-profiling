from datetime import datetime, timedelta

# Parameters
start_year = 112
end_year = 121
n_samples = 512

# Split samples
n_jja = n_samples // 2
n_djf = n_samples - n_jja  # in case n_samples is odd

# --- Helper: build all days between two dates (inclusive) ---
def collect_days(start, end):
    days = []
    delta = (end - start).days + 1
    for i in range(delta):
        days.append(start + timedelta(days=i))
    return days

# --- Build JJA days: June–July–August for each year ---
all_jja = []
for year in range(start_year, end_year + 1):
    all_jja += collect_days(datetime(year, 6, 1), datetime(year, 8, 31))

# --- Build DJF days ---
# DJF = Dec(year) + Jan(year+1) + Feb(year+1)
all_djf = []
for year in range(start_year, end_year + 1):
    # December of current year
    all_djf += collect_days(datetime(year, 12, 1), datetime(year, 12, 31))
    # # January and February of next year
    # next_year = year + 1
    all_djf += collect_days(datetime(year, 1, 1), datetime(year, 2, 28))

# --- Function to choose evenly spaced dates ---
def evenly_spaced_days(days, n):
    total = len(days)
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [days[i] for i in indices]

# Select dates
jja_selected = evenly_spaced_days(all_jja, n_jja)
djf_selected = evenly_spaced_days(all_djf, n_djf)

# Combine
selected_days = jja_selected + djf_selected

# Add constant time 18:00:00
datetimes = [d.replace(hour=18, minute=0, second=0) for d in selected_days]

# Print YAML block
print("init_datetimes: [")
for i in range(0, len(datetimes), 10):
    line = ', '.join(f'"{dt.strftime("%04Y-%m-%d %H:%M:%S")}"' for dt in datetimes[i:i+10])
    print(f"  {line},")
print("]")

# Check total count
dates = [d.strftime("%04Y-%m-%d %H:%M:%S") for d in datetimes]
print(len(dates))