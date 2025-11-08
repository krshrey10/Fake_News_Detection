import csv, random

real_templates = [
  "Government announces new {topic} initiative in {place}.",
  "Official report shows {topic} improved in {year}.",
  "Local authorities schedule {topic} drive next week.",
  "University researchers publish peer-reviewed study on {topic}.",
  "Ministry issues guidelines for {topic} across {place}."
]
fake_templates = [
  "Aliens confirm plan to visit {place} for {topic}.",
  "Celebrity claims {topic} cures all diseases instantly.",
  "Time traveler warns of {topic} disaster tomorrow.",
  "Secret lab creates {topic} machine to control minds.",
  "Dragon spotted in {place} during {topic} festival."
]
topics = ["education","health","transport","climate","finance","vaccination","safety","energy"]
places = ["Delhi","Bengaluru","Mumbai","Kolkata","Chennai","Hyderabad","Pune","Jaipur"]
years  = [str(y) for y in range(2015, 2026)]

rows = []
for _ in range(120):
    t = random.choice(real_templates)
    rows.append((t.format(topic=random.choice(topics), place=random.choice(places), year=random.choice(years)), "REAL"))
for _ in range(120):
    t = random.choice(fake_templates)
    rows.append((t.format(topic=random.choice(topics), place=random.choice(places), year=random.choice(years)), "FAKE"))

random.shuffle(rows)

with open("data/train.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text","label"])
    w.writerows(rows)

print("Wrote data/train.csv with", len(rows), "rows")
