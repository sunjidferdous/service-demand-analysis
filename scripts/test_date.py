from datetime import datetime

date_text = "2018-07-07 22:09:11"

date_object = datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S")

print("Original:", date_text)
print("Parsed:", date_object)
print("Year:", date_object.year)
print("Month:", date_object.month)
print("Day:", date_object.day)