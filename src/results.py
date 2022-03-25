import csv


# Intentionally not using pandas here, so it will work with whatever values are
def add_to_output(path: str, values: list):
    with open(path, 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(values)
